import os
import sys
from functools import partial
import torch
from torch import nn
from torch.utils import model_zoo
import math
from models.senet import SENet
from models import resnet
from torch.nn import BatchNorm2d

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

pretrained_settings = {
    'senet154':
        {
            'imagenet':
                {
                    'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
                    'input_space': 'RGB',
                    'input_size': [3, 224, 224],
                    'input_range': [0, 1],
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225],
                    'num_classes': 1000,
                }
        },
    'se_resnet50':
        {
            'imagenet':
                {
                    'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
                    'input_space': 'RGB',
                    'input_size': [3, 224, 224],
                    'input_range': [0, 1],
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225],
                    'num_classes': 1000,
                }
        },
    'se_resnet101':
        {
            'imagenet':
                {
                    'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
                    'input_space': 'RGB',
                    'input_size': [3, 224, 224],
                    'input_range': [0, 1],
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225],
                    'num_classes': 1000,
                }
        },
    'se_resnet152':
        {
            'imagenet':
                {
                    'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
                    'input_space': 'RGB',
                    'input_size': [3, 224, 224],
                    'input_range': [0, 1],
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225],
                    'num_classes': 1000,
                }
        },
    'se_resnext50_32x4d':
        {
            'imagenet':
                {
                    'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
                    'input_space': 'RGB',
                    'input_size': [3, 224, 224],
                    'input_range': [0, 1],
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225],
                    'num_classes': 1000,
                }
        },
    'se_resnext101_32x4d':
        {
            'imagenet':
                {
                    'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
                    'input_space': 'RGB',
                    'input_size': [3, 224, 224],
                    'input_range': [0, 1],
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225],
                    'num_classes': 1000,
                }
        },
}

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    '3x3 convolution with padding'
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def initialize_pretrained_model(model, num_classes, settings):
    assert (num_classes == settings['num_classes']
           ), 'num_classes should be {}, but is {}'.format(settings['num_classes'], num_classes)
    # model.load_state_dict(model_zoo.load_url(settings['url']), strict=False)
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']

class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """

    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(
            planes * 2,
            planes * 4,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

def senet154(num_classes=1000, pretrained='imagenet'):
    pretrained = 'imagenet'
    model = SENet(
        SEBottleneck,
        [3, 8, 36, 3],
        groups=64,
        reduction=16,
        dropout_p=0.2,
        num_classes=num_classes,
    )
    if pretrained is not None:
        settings = pretrained_settings['senet154'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model

class SEModule(nn.Module):
    def __init__(self, channels, reduction, concat=False):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SCSEModule(nn.Module):
    # according to https://arxiv.org/pdf/1808.08127.pdf concat is better
    def __init__(self, channels, reduction=16, mode='concat'):
        super(SCSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.spatial_se = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )
        self.mode = mode

    def forward(self, x):
        module_input = x

        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        chn_se = self.sigmoid(x)
        chn_se = chn_se * module_input

        spa_se = self.spatial_se(module_input)
        spa_se = module_input * spa_se
        if self.mode == 'concat':
            return torch.cat([chn_se, spa_se], dim=1)
        elif self.mode == 'maxout':
            return torch.max(chn_se, spa_se)
        else:
            return chn_se + spa_se



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        print(x.size())

        x = self.layer1(x)
        print(x.size())

        x = self.layer2(x)
        print(x.size())

        x = self.layer3(x)
        print(x.size())

        x = self.layer4(x)
        print(x.size())

        return x


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


encoder_params = {
    'senet154':
        {
            'filters': [128, 256, 512, 1024, 2048],
            'decoder_filters': [64, 128, 256, 384],
            'init_op': senet154,
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
        },
    'resnet34':
        {
            'filters': [64, 64, 128, 256, 512],
            'decoder_filters': [64, 128, 256, 256],
            'last_upsample': 64,
            'init_op': partial(resnet.resnet34, in_channels=3),
            'url': resnet.model_urls['resnet34'],
        },
        'resnet50':
        {
            'filters': [64, 256, 512, 1024, 2048],
            'decoder_filters': [64, 128, 256, 256],
            'last_upsample': 64,
            'init_op': partial(resnet.resnet50, in_channels=3),
            'url': resnet.model_urls['resnet50'],
        },
}


class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def initialize_encoder(self, model, model_url, num_channels_changed=False):
        if os.path.isfile(model_url):
            pretrained_dict = torch.load(model_url)
        else:
            pretrained_dict = model_zoo.load_url(model_url)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if num_channels_changed:
            model.state_dict()[self.first_layer_params_name +
                               '.weight'][:, :3, ...] = pretrained_dict[self.first_layer_params_name + '.weight'].data
            skip_layers = [
                self.first_layer_params_name,
                self.first_layer_params_name + '.weight',
            ]
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if not any(k.startswith(s) for s in skip_layers)
            }
        model.load_state_dict(pretrained_dict, strict=False)

    @property
    def first_layer_params_name(self):
        return 'conv1'


class EncoderDecoder(AbstractModel):
    def __init__(self, num_classes, num_channels=3, encoder_name='resnet34'):
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = False
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck

        self.filters = encoder_params[encoder_name]['filters']
        self.decoder_filters = encoder_params[encoder_name].get('decoder_filters', self.filters[:-1])
        self.last_upsample_filters = encoder_params[encoder_name].get('last_upsample', self.decoder_filters[0] // 2)

        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.bottlenecks = nn.ModuleList(
            [
                self.bottleneck_type(self.filters[-i - 2] + f, f)
                for i, f in enumerate(reversed(self.decoder_filters[:]))
            ]
        )

        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))])

        if self.first_layer_stride_two:
            self.last_upsample = self.decoder_block(
                self.decoder_filters[0],
                self.last_upsample_filters,
                self.last_upsample_filters,
            )

        self.final = self.make_final_classifier(
            self.last_upsample_filters if self.first_layer_stride_two else self.decoder_filters[0],
            num_classes,
        )

        self._initialize_weights()

        encoder = encoder_params[encoder_name]['init_op'](pretrained=False)
        self.encoder_stages = nn.ModuleList([self.get_encoder(encoder, idx) for idx in range(len(self.filters))])
        if encoder_params[encoder_name]['url'] is not None:
            self.initialize_encoder(encoder, encoder_params[encoder_name]['url'], num_channels != 3)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())

        last_dec_out = enc_results[-1]
        # size = last_dec_out.size(2)
        # last_dec_out = torch.cat([last_dec_out, F.upsample(angles, size=(size, size), mode="nearest")], dim=1)
        x = last_dec_out
        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = -(idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        if self.first_layer_stride_two:
            x = self.last_upsample(x)

        f = self.final(x)

        return f#,x

    def get_decoder(self, layer):
        in_channels = (
            self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[layer + 1]
        )
        return self.decoder_block(
            in_channels,
            self.decoder_filters[layer],
            self.decoder_filters[max(layer, 0)],
        )

    def make_final_classifier(self, in_filters, num_classes):
        return nn.Sequential(nn.Conv2d(in_filters, num_classes, 1, padding=0), nn.Sigmoid())

    def get_encoder(self, encoder, layer):
        raise NotImplementedError

    @property
    def first_layer_params(self):
        return _get_layers_params([self.encoder_stages[0]])

    @property
    def layers_except_first_params(self):
        layers = get_slice(self.encoder_stages, 1, -1) + [
            self.bottlenecks,
            self.decoder_stages,
            self.final,
        ]
        return _get_layers_params(layers)


def _get_layers_params(layers):
    return sum((list(l.parameters()) for l in layers), [])


def get_slice(features, start, end):
    if end == -1:
        end = len(features)
    return [features[i] for i in range(start, end)]


class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class UnetConvTransposeDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class Resnet(EncoderDecoder):
    def __init__(self, seg_classes, backbone_arch):
        self.first_layer_stride_two = True
        super().__init__(seg_classes, 4, backbone_arch)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        elif layer == 1:
            return nn.Sequential(encoder.maxpool, encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4


class ConvTransposeResnetUnet(EncoderDecoder):
    def __init__(self, seg_classes, backbone_arch):
        self.first_layer_stride_two = True
        self.decoder_block = UnetConvTransposeDecoderBlock
        super().__init__(seg_classes, 3, backbone_arch)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        elif layer == 1:
            return nn.Sequential(encoder.maxpool, encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4




class SEUnet(EncoderDecoder):
    def __init__(self, seg_classes=1, backbone_arch='senet154'):
        self.first_layer_stride_two = True
        super().__init__(seg_classes, num_channels=3, encoder_name=backbone_arch)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return encoder.layer0
        elif layer == 1:
            return nn.Sequential(encoder.pool, encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4

    @property
    def first_layer_params_name(self):
        return 'layer0.conv1'


class ConvSCSEBottleneckNoBn(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2):
        # print('bottleneck ', in_channels, out_channels)
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            SCSEModule(out_channels, reduction=reduction, mode='maxout'),
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class SCSEUnet(SEUnet):
    def __init__(self, seg_classes=1, backbone_arch='seresnext50'):
        self.bottleneck_type = ConvSCSEBottleneckNoBn
        super().__init__(seg_classes, backbone_arch=backbone_arch)


setattr(sys.modules[__name__], 'resnet_unet', partial(Resnet))
setattr(sys.modules[__name__], 'convt_resnet_unet', partial(ConvTransposeResnetUnet))

setattr(sys.modules[__name__], 'se_unet', partial(SEUnet))
setattr(sys.modules[__name__], 'scse_unet', partial(SCSEUnet))


if __name__ == '__main__':
    d = DensenetUnet(1, backbone_arch='densenet121')
    d.eval()
    import numpy as np

    with torch.no_grad():
        images = torch.from_numpy(np.zeros((16, 3, 256, 256), dtype='float32'))
        i = d(images)
        print(i.shape)

    print(d)
