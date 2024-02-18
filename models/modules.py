import sys

import matplotlib.pyplot as plt
import PIL
from PIL import Image
import json
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm import create_model

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

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
        return x

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
class _Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, in_channels, out_channels, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3) # depthwise conv
        dim = out_channels
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

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
        return x

class UpBlock(nn.Module):
    """
    Upsampling block for UNeXt
    """
    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None):
        super().__init__()
        if up_conv_in_channels is None:
          up_conv_in_channels = in_channels
        if up_conv_out_channels is None:
          up_conv_out_channels = out_channels
        self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        self.block = Block(out_channels)
    
    def forward(self, x, res):
        x = self.upsample(x, output_size=res.shape)
        x = torch.cat([x, res], 1)
        x = self.block(x)
        return x


class Encoder(torch.nn.Module):
    '''
    Encoder for UNeXt, making use of pre-trained ConvNeXt.
    '''
    def __init__(self, model):
        super().__init__()
        self.input = model.stem
        self.layers = model.stages

    def forward(self, x):
        pre_pools = []
        pre_pools.append(x)
        x = self.input(x)
        pre_pools.append(x)
        for layer in self.layers:
          x = layer(x)
          pre_pools.append(x)
        pre_pools = list(reversed(pre_pools[:-1]))
        return x, pre_pools

class Bridge(nn.Module):
    '''
    Bridge to connect Encoder to Decoder.
    '''
    def __init__(self, nc):
        super().__init__()
        self.block1 = Block(nc)
        self.block2 = Block(nc)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

class Head(nn.Module):
    '''
    Head for UNeXt Decoder.
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels//2, 1, 1)
        self.down = nn.Conv2d(in_channels, in_channels//4, 3, 1, 1)
        self.norm1 = LayerNorm(in_channels//4, eps=1e-6)
        self.up1 = nn.ConvTranspose2d(in_channels//4, in_channels//4, 4, 4)
        self.norm2 = LayerNorm(in_channels//4, eps=1e-6)
        self.block = nn.Sequential(nn.Conv2d(in_channels//4, out_channels, 3, 1, 1))
        # self.block = nn.Sequential(nn.Conv2d(in_channels//4, out_channels, 3, 1, 1), nn.Sigmoid())
    
    def forward(self, x, pools):
        res = pools[0]
        x = self.upsample(x, output_size=res.shape)
        x = torch.cat([x, res], 1)
        x = self.down(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        res = pools[2]
        x = self.up1(x, output_size=res.shape)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.block(x)
        return x
class Decoder(nn.Module):
    '''
    Decoder for UNeXt.
    '''
    def __init__(self, in_channels=2048, out_channels=3):
        super().__init__()
        up_blocks = []
        up_blocks.append(UpBlock(in_channels, in_channels, up_conv_out_channels=in_channels//2))
        up_blocks.append(UpBlock(in_channels, in_channels//2, up_conv_out_channels=in_channels//4))
        up_blocks.append(UpBlock(in_channels//2, in_channels//4, up_conv_out_channels=in_channels//8))
        self.blocks = up_blocks
        self.head = Head(in_channels//4, out_channels)

    def forward(self, x, pre_pools):
        for i, block in enumerate(self.blocks):
          x = block(x, pre_pools[i])
        x = self.head(x, pre_pools[-3:])
        return x

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        up_blocks = [] 
        for b in self.blocks:
            up_blocks.append(b.to(*args, **kwargs))
        self.blocks = up_blocks
        return self