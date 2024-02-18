from models.sym_padding import *
import torch
import torch.nn.functional as F
import h5py
from torchvision.utils import save_image

class Featex_vgg16_base(nn.Module):
    def __init__(self, type=1, mid_output = False):
        super(Featex_vgg16_base, self).__init__()
        base = 32
        self.type = type
        self.mid_output = mid_output
        # block 1
        in_channels = 3
        out_channels = base # 32
        self.b1c1 = CombinedConv2D(in_channels, 16) # relu
        self.b1c2 = Conv2DSymPadding(16, out_channels, (3, 3)) # relu
        # block 2
        in_channels = out_channels
        out_channels = 2 * base # 64
        self.b2c1 = Conv2DSymPadding(in_channels, out_channels, (3, 3)) # relu
        self.b2c2 = Conv2DSymPadding(out_channels, out_channels, (3, 3)) # relu
        # block 3
        in_channels = out_channels
        out_channels = 4 * base # 128??
        self.b3c1 = Conv2DSymPadding(in_channels, out_channels, (3, 3)) # relu
        self.b3c2 = Conv2DSymPadding(out_channels, out_channels, (3, 3)) # relu
        self.b3c3 = Conv2DSymPadding(out_channels, out_channels, (3, 3)) # relu
        # block 4
        in_channels = out_channels
        out_channels = 8 * base # 256??
        self.b4c1 = Conv2DSymPadding(in_channels, out_channels, (3, 3)) # relu
        self.b4c2 = Conv2DSymPadding(out_channels, out_channels, (3, 3)) # relu
        self.b4c3 = Conv2DSymPadding(out_channels, out_channels, (3, 3)) # relu
        # block 5/bottle-neck
        self.b5c1 = Conv2DSymPadding(out_channels, out_channels, (3, 3)) # relu
        self.b5c2 = Conv2DSymPadding(out_channels, out_channels, (3, 3)) # relu
        self.transform = Conv2DSymPadding(out_channels, out_channels, (3, 3)) # tanh if type >= 1 else None
        # l2_norm
        

    def forward(self, x):
        # block 
        #print(x.shape)
        #save_image(x,'/share/home/dongli/Liang/DL_code/myseg-project/srm/x.jpg')
        srm_out, bayar_out = self.b1c1(x)
        # for i in range(len(srm_out[0])):
        #     save_image(srm_out[:,i:i+1,:,:],'/share/home/dongli/Liang/DL_code/myseg-project/srm/srm_{}.jpg'.format(i))
        # for i in range(len(bayar_out[0])):
        #     save_image(bayar_out[:,i:i+1,:,:],'/share/home/dongli/Liang/DL_code/myseg-project/srm/bayar_out_{}.jpg'.format(i))

        return srm_out,bayar_out

def _load_Featex_weights(model, f):
    for k in f.keys():
        layer_name = k[:-2]
        layer = getattr(model, layer_name) # b1c1, b1c2, etc.
        for sub_k in f[k].keys():
            param_name = sub_k[:-2]
            if param_name == 'kernel':
                param_name = 'weight'
            param = getattr(layer, param_name) # weight or bias
            weight = f[k][sub_k][:]
            # transposing weights for conv kernel
            if 'kernel' in sub_k:
                weight = weight.transpose(3, 2, 0, 1)
            assert  weight.shape == param.data.shape, \
                    f"Featex.{layer_name}.{param_name}: Shape doesn't match. Got {param.data.shape} but needs {weight.shape}."
            param.data = torch.from_numpy(weight)
    return model

def load_featex_weights(weight_filepath, model):
    f = h5py.File(weight_filepath, 'r')
    # Featex
    model = _load_Featex_weights(model, f['Featex'])
    return model