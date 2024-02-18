import re
import segmentation_models_pytorch as smp

def smp_model(type = 'unet++'):
    model = smp.UnetPlusPlus(encoder_name='resnet34')
    return model