from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

# Author of LayerNorm & ConvNeXt block :
# https://github.com/facebookresearch/ConvNeXt/blob/main/semantic_segmentation/backbone/convnext.py

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXtEncoderBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, in_channels, features, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding="same", groups=in_channels) # depthwise conv
        self.norm = LayerNorm(in_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channels, 4 * in_channels) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * in_channels, in_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((in_channels)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
     
        self.convFinal = nn.Conv2d(in_channels, features * 2, padding='same', kernel_size=7)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        x = self.convFinal(x)

        return x



class ResUNetEncoderBlock(nn.Module):

    def __init__(self, in_channels, features, kernel_size=3):
        super(ResUNetEncoderBlock, self).__init__()

        self.convPath1 = nn.Conv2d(in_channels, features, padding='same', kernel_size=kernel_size)
        self.batchNormPath1 = nn.BatchNorm2d(features)
        self.reluPath1 = nn.ReLU()

        self.convPath2 = nn.Conv2d(features, features, padding='same', kernel_size=kernel_size)
        self.batchNormPath2 = nn.BatchNorm2d(features)

        self.convShortcut = nn.Conv2d(in_channels, features, padding='same', kernel_size=1)
        self.batchNormShortcut = nn.BatchNorm2d(features)

        self.reluAddition = nn.ReLU()

    def forward(self, x):

        path = self.convPath1(x)
        path = self.batchNormPath1(path)
        path = self.reluPath1(path)

        path = self.convPath2(path)
        path = self.batchNormPath2(path)

        shortcut = self.convShortcut(x)
        shortcut = self.batchNormShortcut(shortcut)

        addition = torch.cat((path, shortcut), dim=1)

        out = self.reluAddition(addition)

        return out





class ConvNeXtUNet(nn.Module):
    def __init__(self, in_channels, out_classes, features=16):
        super(ConvNeXtUNet, self).__init__()

        # Encoder
        self.encoder_1 = ConvNeXtEncoderBlock(in_channels, features)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_2 = ConvNeXtEncoderBlock(features * 2, features * 2)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_4 = ConvNeXtEncoderBlock(features * 2 * 2, features * 4)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_8 = ConvNeXtEncoderBlock(features * 2 * 4, features * 8)
        self.pool_8 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_16 = ConvNeXtEncoderBlock(features * 2 * 8, features * 16)

        # Decoder
        self.upconv_8 = nn.ConvTranspose2d(features * 2 * 16, features * 8, kernel_size=2, stride=2)
        self.decoder_8 = ConvNeXtEncoderBlock((features * 8) * 3, features * 8)

        self.upconv_4 = nn.ConvTranspose2d(features * 2 * 8, features * 4, kernel_size=2, stride=2)
        self.decoder_4 = ConvNeXtEncoderBlock((features * 4) * 3, features * 4)

        self.upconv_2 = nn.ConvTranspose2d(features * 2 * 4, features * 2, kernel_size=2, stride=2)
        self.decoder_2 = ConvNeXtEncoderBlock((features * 2) * 3, features * 2)

        self.upconv_1 = nn.ConvTranspose2d(features * 2 * 2, features, kernel_size=2, stride=2)
        self.decoder_1 = ConvNeXtEncoderBlock(features * 3, features)

        # Classifier
        self.convClassifier1 = nn.Conv2d(features * 2, out_classes * 2, padding="same", kernel_size=3)
        self.batchNormClassifier = nn.BatchNorm2d(out_classes * 2)
        self.reluClassifier = nn.ReLU()

        self.convClassifier2 = nn.Conv2d(out_classes * 2, out_classes, padding="same", kernel_size=1)

    def forward(self, x):

        # Encoder
        encoder_1 = self.encoder_1(x)
        pool_1 = self.pool_1(encoder_1)

        encoder_2 = self.encoder_2(pool_1)
        pool_2 = self.pool_2(encoder_2)

        encoder_4 = self.encoder_4(pool_2)
        pool_4 = self.pool_4(encoder_4)

        encoder_8 = self.encoder_8(pool_4)
        pool_8 = self.pool_8(encoder_8)

        encoder_16 = self.encoder_16(pool_8)

        # Decoder
        decoder_8 = self.upconv_8(encoder_16)

        decoder_8 = torch.cat((decoder_8, encoder_8), dim=1)
        decoder_8 = self.decoder_8(decoder_8)

        decoder_4 = self.upconv_4(decoder_8)
        decoder_4 = torch.cat((decoder_4, encoder_4), dim=1)
        decoder_4 = self.decoder_4(decoder_4)

        decoder_2 = self.upconv_2(decoder_4)
        decoder_2 = torch.cat((decoder_2, encoder_2), dim=1)
        decoder_2 = self.decoder_2(decoder_2)

        decoder_1 = self.upconv_1(decoder_2)
        decoder_1 = torch.cat((decoder_1, encoder_1), dim=1)
        decoder_1 = self.decoder_1(decoder_1)

        # Classifier
        classifier = self.convClassifier1(decoder_1)
        classifier = self.batchNormClassifier(classifier)
        classifier = self.reluClassifier(classifier)

        classifier = self.convClassifier2(classifier)

        return classifier


if __name__ == "__main__":
    model = ConvNeXtUNet(3, 17)
    from torchinfo import summary
    summary(model, input_size=(5, 3, 512, 512))