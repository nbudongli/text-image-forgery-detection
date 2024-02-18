import torch.nn as nn
import torch.nn.functional as F
import torch

# ~~~~~~~~~~ U-Net ~~~~~~~~~~

class U_double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(U_double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = U_double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class U_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(U_down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            U_double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class U_up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(U_up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = U_double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffY, 0,
                        diffX, 0))
        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)
        return x


# ~~~~~~~~~~ RU-Net ~~~~~~~~~~

class RU_double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RU_double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        x = self.conv(x)
        return x


class RU_first_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RU_first_down, self).__init__()
        self.conv = RU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(ft1 + self.res_conv(x))

        return r1


class RU_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RU_down, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = RU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        x = self.maxpool(x)
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(ft1 + self.res_conv(x))

        return r1


class RU_up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(RU_up, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        #  nn.Upsample hasn't weights to learn, but nn.ConvTransposed2d has weights to learn.
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = RU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_ch))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffY, 0,
                        diffX, 0))
        x = torch.cat([x2, x1], dim=1)

        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(self.res_conv(x) + ft1)

        return r1


# ~~~~~~~~~~ RRU-Net ~~~~~~~~~~

class RRU_double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RRU_double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2),
            nn.GroupNorm(32, out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class RRU_first_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RRU_first_down, self).__init__()
        self.conv = RRU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_ch)
        )
        self.res_conv_back = nn.Sequential(
            nn.Conv2d(out_ch, in_ch, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(ft1 + self.res_conv(x))
        # the second ring conv
        ft2 = self.res_conv_back(r1)
        x = torch.mul(1 + torch.sigmoid(ft2), x)
        # the third ring conv
        ft3 = self.conv(x)
        r3 = self.relu(ft3 + self.res_conv(x))

        return r3


class RRU_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RRU_down, self).__init__()
        self.conv = RRU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(32, out_ch))
        self.res_conv_back = nn.Sequential(
            nn.Conv2d(out_ch, in_ch, kernel_size=1, bias=False))

    def forward(self, x):
        x = self.pool(x)
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(ft1 + self.res_conv(x))
        # the second ring conv
        ft2 = self.res_conv_back(r1)
        x = torch.mul(1 + torch.sigmoid(ft2), x)
        # the third ring conv
        ft3 = self.conv(x)
        r3 = self.relu(ft3 + self.res_conv(x))

        return r3


class RRU_up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(RRU_up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2),
                nn.GroupNorm(32, in_ch // 2))

        self.conv = RRU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_ch))
        self.res_conv_back = nn.Sequential(
            nn.Conv2d(out_ch, in_ch, kernel_size=1, bias=False))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffY, 0,
                        diffX, 0))

        x = self.relu(torch.cat([x2, x1], dim=1))

        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(self.res_conv(x) + ft1)
        # the second ring conv
        ft2 = self.res_conv_back(r1)
        x = torch.mul(1 + torch.sigmoid(ft2), x)
        # the third ring conv
        ft3 = self.conv(x)
        r3 = self.relu(ft3 + self.res_conv(x))

        return r3


# !!!!!!!!!!!! Universal functions !!!!!!!!!!!!

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Unet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = U_down(64, 128)
        self.down2 = U_down(128, 256)
        self.down3 = U_down(256, 512)
        self.down4 = U_down(512, 512)
        self.up1 = U_up(1024, 256)
        self.up2 = U_up(512, 128)
        self.up3 = U_up(256, 64)
        self.up4 = U_up(128, 64)
        self.out = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x


class Res_Unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Res_Unet, self).__init__()
        self.down = RU_first_down(n_channels, 32)
        self.down1 = RU_down(32, 64)
        self.down2 = RU_down(64, 128)
        self.down3 = RU_down(128, 256)
        self.down4 = RU_down(256, 256)
        self.up1 = RU_up(512, 128)
        self.up2 = RU_up(256, 64)
        self.up3 = RU_up(128, 32)
        self.up4 = RU_up(64, 32)
        self.out = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.down(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x


class Ringed_Res_Unet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(Ringed_Res_Unet, self).__init__()
        self.down = RRU_first_down(n_channels, 32)
        self.down1 = RRU_down(32, 64)
        self.down2 = RRU_down(64, 128)
        self.down3 = RRU_down(128, 256)
        self.down4 = RRU_down(256, 256)
        self.up1 = RRU_up(512, 128)
        self.up2 = RRU_up(256, 64)
        self.up3 = RRU_up(128, 32)
        self.up4 = RRU_up(64, 32)
        self.out = outconv(32, n_classes)

    def forward(self, x):
        x_ = x
        x1 = self.down(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x#,x_