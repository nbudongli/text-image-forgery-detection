from models.modules import *
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm import create_model



class UNeXt(nn.Module):
    '''
    UNeXt module, a ConvNeXt based U-Net architecture. 
    '''
    def __init__(self, noc, model_name="convnext_tiny_in22ft1k", in_channels=3, device=None):
        super().__init__()
        if device is None:
           device = 'cuda' if torch.cuda.is_available() else 'cpu'
        convnext = create_model(model_name, pretrained=True).to(device)
        print('load pretrain weight successful!')      
        # shapes = {
        #     'convnext_xlarge_in22k': [256, 512, 1024, 2048],
        #     'convnext_large_22k': [192, 384, 768, 1536],
        #     'convnext_base_22k': [128, 256, 512, 1024],
        #     'convnext_small_22k': [96, 192, 384, 768],
        #     'convnext_tiny_22k': [96, 192, 384, 768]
        # }
        self.input = nn.Conv2d(in_channels, 3, 3, padding=1)
        self.encoder = Encoder(convnext)
        dim = convnext.head.norm.weight.shape[0]
        self.bridge = Bridge(dim)
        self.decoder = Decoder(dim, noc)
        self.device = device
    def forward(self, x):
        x = self.input(x)
        enc, pools = self.encoder(x)
        bridge_out = self.bridge(enc)
        out = self.decoder(bridge_out, pools)
        return out
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.encoder = self.encoder.to(*args, **kwargs)
        self.bridge = self.bridge.to(*args, **kwargs)
        self.decoder = self.decoder.to(*args, **kwargs)
        return self

if __name__ == '__main__':
    # model_name = "convnext_xlarge_in22k"
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print("device = ", device)
    # # create a ConvNeXt model : https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py
    # model = create_model(model_name, pretrained=True).to(device)
    # from timm.data.constants import \
    # IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    # NORMALIZE_MEAN = IMAGENET_DEFAULT_MEAN
    # NORMALIZE_STD = IMAGENET_DEFAULT_STD
    # SIZE = 256

    # # Here we resize smaller edge to 256, no center cropping
    # transforms = [
    #               T.Resize(SIZE, interpolation=T.InterpolationMode.BICUBIC),
    #               T.ToTensor(),
    #               T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    #               ]

    # transforms = T.Compose(transforms)
    # imagenet_labels = json.load(open('label_to_words.json'))
    # img = PIL.Image.open('test.jpeg')
    # img_tensor = transforms(img).unsqueeze(0).to(device)
    unext = UNeXt(3)
    unext = unext.to(unext.device)
    # out = unext(img_tensor)
    # print(out.shape)
    print(unext)
    # torchvision.utils.save_image(out.squeeze().detach().cpu(), 'test.png')
