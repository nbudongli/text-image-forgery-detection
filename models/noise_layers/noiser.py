import numpy as np
import torch.nn as nn
from models.noise_layers.identity import Identity
from models.noise_layers.jpeg_compression import JpegCompression
from models.noise_layers.quantization import Quantization


class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self):
        super(Noiser, self).__init__()
        #self.noise_layers = [Identity()]
        self.noise_layers = []
        #self.noise_layers.append(JpegCompression())
        self.noise_layers.append(Quantization())

        # self.noise_layers = nn.Sequential(*noise_layers)

    def forward(self, encoded_and_cover):
        random_noise_layer = np.random.choice(self.noise_layers, 3)[0]
        #print(random_noise_layer)
        #exit()
        return random_noise_layer(encoded_and_cover)

