
import torch.optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn as nn
import math
import torch.nn.functional as F

class LSTMEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(LSTMEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(3, 16, 5, padding=2, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 1, 5, padding=2, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=3, batch_first=True)
        self.lstm.flatten_parameters()

    def forward(self, x):
        bs = x.size(0)

        x = self.relu1(self.conv1(x))  # bsx16x64x64

        y = self.relu2(self.conv2(x))  # bsx1x64x64

        # split x to 8x8 blocks
        y_list = y.split(8, dim=3)  # 8x[(bsx1x8x64)]
        xy_list = [x.split(8, dim=2) for x in y_list]  # 8x8x( bsx 1x 8 x 8)

        xy = [item for items in xy_list for item in items]
        xy = torch.cat(xy, 1)  # bsx64x(8x8)

        b,c,h,w = xy.shape
        #print(xy.shape)
        #exit()
        xy = xy.view(bs, c, 64)  # bs x 64 x 64

        self.lstm.flatten_parameters()
        # 8x8 list
        outputs, (ht, ct) = self.lstm(xy)

        return outputs


class Segmentation(nn.Module):
    def __init__(self, patch_size):
        super(Segmentation, self).__init__()

        self.sqrt_patch_size = int(math.sqrt(patch_size))
        self.patch_size = patch_size

        self.conv3 = nn.Conv2d(96, 32, 5, padding=2, bias=False)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(32, 2, 5, padding=2, bias=False)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(2, 1, 5, padding=2, bias=False)
        self.relu5 = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(2, stride=2)

        #self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, outputs_lstm):
        bs = outputs_lstm.size(0)
        # outputs = [bs,64,256]
        b,c,n = outputs_lstm.shape
        # print(outputs_lstm.shape)
        # print(c//self.patch_size)
        # exit()
        #print(outputs_lstm.shape)
        #print(self.sqrt_patch_size)
        #exit()
        # outputs_lstm = outputs_lstm.contiguous().view(bs, c,self.sqrt_patch_size * 2, self.sqrt_patch_size * 2).permute(0, 1,
        #                                                                                                           3, 2,
        #                                                                                         4).contiguous()
        # 24,4096,8,8
        # bs,8*16,8*16
        outputs_lstm = outputs_lstm.reshape(bs, c//self.patch_size, self.patch_size, self.patch_size * 4)

        # bs x 32 x 96x96
        x = self.relu3(self.conv3(outputs_lstm))

        x = self.max_pool(x)

        x = self.relu4(self.conv4(x))

        x = self.conv5(x)
        x = F.interpolate(x, size=[768,512], mode="bicubic",align_corners=False)
        #print(x.shape)
        #exit()
        #output_mask = torch.sigmoid(self.conv5(x))

        return x


class Classification(nn.Module):
    """docstring for Classification"""

    def __init__(self, hidden_dim, number_of_class=2):
        super(Classification, self).__init__()
        self.hidden_dim = hidden_dim
        self.number_of_class = number_of_class

        self.linear = nn.Linear(self.hidden_dim, self.number_of_class)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, outputs_lstm):
        outputs = outputs_lstm[:, -1, :]  # bsx256

        y = self.softmax(self.linear(outputs))

        return y

class HybridN(nn.Module):
    def __init__(self):
        super(HybridN, self).__init__()

        patch_size = 64
        hidden_dim = 256
        number_of_class = 1
        self.encode_lstm = LSTMEncoder(patch_size, hidden_dim)
        self.classification = Classification(hidden_dim, number_of_class)
        self.Segmentation = Segmentation(patch_size)

    def forward(self, x):
        local_feature = self.encode_lstm(x)
        #classification = self.classification(local_feature)
        seg = self.Segmentation(local_feature)
        return seg#,classification


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)

def hybrid_network():
    model = HybridN()
    model.apply(weights_init)
    return model