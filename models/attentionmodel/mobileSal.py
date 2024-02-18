import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
import math
from .MobileNetV2 import mobilenet_v2
import segmentation_models_pytorch as smp
from models.attention.DANet import DAModule
from models.attention.CBAM import CBAMBlock
from models.attention.PSA import PSA
from models.attention.SKAttention import SKAttention
from models.attention.BAM import BAMBlock
from models.attention.ShuffleAttention import ShuffleAttention
from models.create_featex import Featex_vgg16_base
from models.create_featex import load_featex_weights
from models.salientmodel.u2net import RSU4F
import numpy as np


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "{})".format(self.weight.shape[0])
        return s


class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            bias=True, use_relu=True, leaky_relu=False, use_bn=True, frozen=False, spectral_norm=False, prelu=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad, \
                              dilation=dilation, groups=groups, bias=bias)
        if use_bn:
            if frozen:
                self.bn = FrozenBatchNorm2d(nOut)
            elif spectral_norm:
                self.bn = SpectralNorm(nOut)
            else:
                self.bn = BatchNorm2d(nOut)
        else:
            self.bn = None
        if use_relu:
            if leaky_relu is True:
                self.act = nn.LeakyReLU(0.1, inplace=True)
            elif prelu is True:
                self.act = nn.PReLU(nOut)
            else:
                self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)

        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            bias=True, use_relu=True, use_bn=True, frozen=False):
        super(ResidualConvBlock, self).__init__()
        self.conv = ConvBNReLU(nIn, nOut, ksize=ksize, stride=stride, pad=pad,
                               dilation=dilation, groups=groups, bias=bias,
                               use_relu=use_relu, use_bn=use_bn, frozen=frozen)
        self.residual_conv = ConvBNReLU(nIn, nOut, ksize=1, stride=stride, pad=0,
                               dilation=1, groups=groups, bias=bias,
                               use_relu=False, use_bn=use_bn, frozen=frozen)

    def forward(self, x):
        x = self.conv(x) + self.residual_conv(x)
        return x
    


class ReceptiveConv(nn.Module):
    def __init__(self, inplanes, planes, baseWidth=24, scale=4, dilation=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: basic width of conv3x3
            scale: number of scale.
        """
        super(ReceptiveConv, self).__init__()
        assert scale >= 1, 'The input scale must be a positive value'

        self.width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, self.width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.width*scale)
        #self.nums = 1 if scale == 1 else scale - 1
        self.nums = scale

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        dilation = [1] * self.nums if dilation is None else dilation
        for i in range(self.nums):
            self.convs.append(nn.Conv2d(self.width, self.width, kernel_size=3, \
                    padding=dilation[i], dilation=dilation[i], bias=False))
            self.bns.append(nn.BatchNorm2d(self.width))

        self.conv3 = nn.Conv2d(self.width*scale, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.scale = scale

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            out = sp if i == 0 else torch.cat((out, sp), 1)
        #if self.scale > 1:
        #    out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out += x
        out = self.relu(out)

        return out



class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4, residual=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        if self.stride == 1 and inp == oup:
            self.use_res_connect = residual
        else:
            self.use_res_connect = False

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, ksize=1, pad=0))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
    


class SEnet(nn.Module):
    def __init__(self,channels,ratio=16):
        super(SEnet, self).__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        # 经过两次全连接层，一次较小，一次还原
        self.fc=nn.Sequential(
            nn.Linear(channels,channels//ratio,False),
            nn.ReLU(),
            nn.Linear(channels//ratio, channels, False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_,_=x.size() #取出batch size和通道数
        # b,c,w,h->b,c,1,1->b,c 以便进行全连接
        avg=self.avgpool(x).view(b,c)
        #b,c->b,c->b,c,1,1 以便进行线性加权
        fc=self.fc(avg).view(b,c,1,1) 
        return fc*x


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        if use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class MFAB(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, reduction=16):
        # MFAB is just a modified version of SE-blocks, one for skip, one for input
        super(MFAB, self).__init__()
        self.hl_conv = nn.Sequential(
            Conv2dReLU(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            ),
            Conv2dReLU(
                in_channels,
                skip_channels,
                kernel_size=1,
                use_batchnorm=use_batchnorm,
            )
        )
        reduced_channels = max(1, skip_channels // reduction)
        self.SE_ll = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(skip_channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, skip_channels, 1),
            nn.Sigmoid(),
        )
        self.SE_hl = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(skip_channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, skip_channels, 1),
            nn.Sigmoid(),
        )
        self.conv1 = Conv2dReLU(
            skip_channels + skip_channels,  # we transform C-prime form high level to C from skip connection
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None):
        x = self.hl_conv(x)
        x = F.interpolate(x, scale_factor=4, mode="nearest")
        attention_hl = self.SE_hl(x)
        if skip is not None:
            attention_ll = self.SE_ll(skip)
            attention_hl = attention_hl + attention_ll
            x = x * attention_hl
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        #x = self.conv2(x)
        return x

class Conv2dBnRelu(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1,padding=0,dilation=1,bias=True):
        super(Conv2dBnRelu,self).__init__()
		
        self.conv = nn.Sequential(
		nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding,dilation=dilation,bias=bias),
		nn.BatchNorm2d(out_ch, eps=1e-3),
		nn.ReLU(inplace=True)
	)

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, inplanes, planes, upsample=False):
        super(UpBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.will_ups = upsample

    def forward(self, x):
        if self.will_ups:
            x = nn.functional.interpolate(x,scale_factor=2, mode="bilinear", align_corners=True)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MobileSal(nn.Module):
    def __init__(self, pretrained=True, use_carafe=True,
                 enc_channels=[40, 32, 48, 136, 384],   # [16, 24, 32, 96, 320]  # [3,16,24,32,96,1280]
                 dec_channels=[40, 32, 48, 136, 384]):  # [40, 32, 48, 136, 384]
        super(MobileSal, self).__init__()



        #self.backbone = mobilenet_v2(pretrained)mobilenet_v2 # timm-efficientnet-b3
        self.backbone = smp.PAN(encoder_name='timm-efficientnet-b3',encoder_weights="imagenet")
        self.depth_fuse = DepthFuseNet(inchannels=384)
        self.fpn = CPRDecoder(enc_channels, dec_channels)
        self.in_chals = 384+12
        self.out_chals = 16

        self.cls6 = nn.Conv2d(self.out_chals, 1, 3, stride=1, padding=0)
        self.cls1 = nn.Conv2d(dec_channels[0], 1, 1, stride=1, padding=0)
        self.cls2 = nn.Conv2d(dec_channels[1], 1, 1, stride=1, padding=0)
        self.cls3 = nn.Conv2d(dec_channels[2], 1, 1, stride=1, padding=0)
        self.cls4 = nn.Conv2d(dec_channels[3], 1, 1, stride=1, padding=0)
        self.cls5 = nn.Conv2d(dec_channels[4], 1, 1, stride=1, padding=0)
        self.senet = SEnet(self.in_chals)
        self.danet=DAModule(d_model=5,kernel_size=3,H=7,W=7)
        self.cbam = CBAMBlock(channel=384+12,reduction=16,kernel_size=7)
        self.ska = SKAttention(channel=5,reduction=2)
        self.bam = BAMBlock(channel=384,reduction=16,dia_val=2)
        self.mfab = MFAB(in_channels=self.in_chals, skip_channels=32, out_channels=self.out_chals)

        self.featex = Featex_vgg16_base()
        self.featex = load_featex_weights('/share/home/dongli/Liang/DL_code/myseg-project/models/ManTraNet_Ptrain4.h5',self.featex)
        self.apn = RSU4F(self.in_chals,64,self.out_chals)

        self.decoder =nn.Sequential(
                UpBlock(self.in_chals, 256, upsample=True), 
                UpBlock(256, 128, upsample=True), 
                UpBlock(128, 64, upsample=True),
                UpBlock(64, 16, upsample=True),
                nn.Conv2d(16, 1, 1, 1),
                #nn.LeakyReLU()
                )


    def forward(self, input, depth=None, test=True):
        _,conv1, conv2, conv3, conv4, conv5 = self.backbone(input)
        #print(conv2.shape)
        #print(conv5.shape)
        #exit()
        srm_out,bayar_out = self.featex(input)
        srm_out = F.interpolate(srm_out,scale_factor=1/16.,mode='bilinear',align_corners=False)
        bayar_out = F.interpolate(bayar_out,scale_factor=1/16.,mode='bilinear',align_corners=False)
        feature_tensor = torch.cat([conv5,srm_out,bayar_out], dim=1)
        features_ = self.cbam(feature_tensor) # c:(384+12)*32*32
        features_ = self.mfab(features_,conv2)
        out_ = self.cls6(features_)
        out_ = F.interpolate(out_,input.shape[2:],mode='bilinear',align_corners=False)

        return out_


       
        #print(features_.shape)
        #exit()
        #features_ = self.cbam(feature_tensor) # c:(384+12)*32*32
        #features_ = self.block(feature_tensor)
        #features_ = self.apn(feature_tensor)

class DepthFuseNet(nn.Module):
    def __init__(self, inchannels=384):
        super(DepthFuseNet, self).__init__()
        self.d_conv1 = InvertedResidual(inchannels, inchannels, residual=True)
        self.d_linear = nn.Sequential(
            nn.Linear(inchannels, inchannels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(inchannels, inchannels, bias=True),
        )
        self.d_conv2 = InvertedResidual(inchannels, inchannels, residual=True)

    def forward(self, x, x_d):
        x_f = self.d_conv1(x * x_d)
        x_d1 = self.d_linear(x.mean(dim=2).mean(dim=2)).unsqueeze(dim=2).unsqueeze(dim=3)
        x_f1 = self.d_conv2(torch.sigmoid(x_d1) * x_f * x_d)
        return x_f1
    


class CPR(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4, dilation=[1,2,3], residual=True):
        super(CPR, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        if self.stride == 1 and inp == oup:
            self.use_res_connect = residual
        else:
            self.use_res_connect = False

        self.conv1 = ConvBNReLU(inp, hidden_dim, ksize=1, pad=0, prelu=False)

        self.hidden_conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation[0], groups=hidden_dim, dilation=dilation[0])
        self.hidden_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation[1], groups=hidden_dim, dilation=dilation[1])
        self.hidden_conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation[2], groups=hidden_dim, dilation=dilation[2])
        self.hidden_bnact = nn.Sequential(nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True))
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        m = self.conv1(x)
        m = self.hidden_conv1(m) + self.hidden_conv2(m) + self.hidden_conv3(m)
        m = self.hidden_bnact(m)
        if self.use_res_connect:
            return x + self.out_conv(m)
        else:
            return self.out_conv(m)
    


class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, input_num=2):
        super(Fusion, self).__init__()
        if input_num == 2:
            self.channel_att = nn.Sequential(nn.Linear(in_channels, in_channels),
                                             nn.ReLU(),
                                             nn.Linear(in_channels, in_channels),
                                             nn.Sigmoid()
                                             )
        self.fuse = nn.Sequential(CPR(in_channels, in_channels, expand_ratio=expansion, residual=True),
                                      ConvBNReLU(in_channels, in_channels, ksize=1, pad=0, stride=1)
                                      )


    def forward(self, low, high=None):
        if high is None:
            final = self.fuse(low)
        else:
            high_up = F.interpolate(high, size=low.shape[2:], mode='bilinear', align_corners=False)
            fuse = torch.cat((high_up, low), dim=1)
            final = self.channel_att(fuse.mean(dim=2).mean(dim=2)).unsqueeze(dim=2).unsqueeze(dim=2) * self.fuse(fuse)
        return final



class CPRDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, teacher=False):
        super(CPRDecoder, self).__init__()
        #assert in_channels[-1] == out_channels[-1]
        self.inners_a = nn.ModuleList()
        self.inners_b = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.inners_a.append(ConvBNReLU(in_channels[i], out_channels[i] // 2, ksize=1, pad=0))
            self.inners_b.append(ConvBNReLU(out_channels[i + 1], out_channels[i] // 2, ksize=1, pad=0))
        self.inners_a.append(ConvBNReLU(in_channels[-1], out_channels[-1], ksize=1, pad=0))

        self.fuse = nn.ModuleList()
        for i in range(len(in_channels)):
            if i == len(in_channels) - 1:
                self.fuse.append(Fusion(out_channels[i], out_channels[i], input_num=1))
            else:
                self.fuse.append(
                    ConvBNReLU(out_channels[i], out_channels[i]) if teacher else Fusion(out_channels[i], out_channels[i])
                    )

    def forward(self, features, att=None):
        stage_result = self.fuse[-1](self.inners_a[-1](features[-1]))
        results = [stage_result]
        for idx in range(len(features) - 2, -1, -1):
            #inner_top_down = F.interpolate(self.inners_b[idx](stage_result),
            #                               size=features[idx].shape[2:],
            #                               mode='bilinear',
            #                               align_corners=False)
            inner_top_down = self.inners_b[idx](stage_result)
            inner_lateral = self.inners_a[idx](features[idx])
            stage_result = self.fuse[idx](inner_lateral, inner_top_down)#(torch.cat((inner_top_down, inner_lateral), dim=1))
            results.insert(0, stage_result)
        return results

    
class FPNDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPNDecoder, self).__init__()
        #assert in_channels[-1] == out_channels[-1]
        self.inners = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.inners.append(ConvBNReLU(in_channels[i], out_channels[i], ksize=1, pad=0))

        self.fuse = nn.ModuleList()
        for i in range(len(out_channels)):
            self.fuse.append(
                ConvBNReLU(out_channels[i], out_channels[i]),
            )

    def forward(self, features, att=None):
        stage_result = self.fuse[-1](self.inners[-1](features[-1]))
        results = [stage_result]
        for idx in range(len(features) - 2, -1, -1):
            inner_top_down = F.interpolate(self.inners[idx](stage_result),
                                           size=features[idx].shape[2:],
                                           mode='bilinear',
                                           align_corners=False)
            inner_lateral = self.inners[idx-1](features[idx])
            stage_result = self.fuse[idx](inner_top_down + inner_lateral)
            results.insert(0, stage_result)
        return results
