import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn import init

def _pad_symmetric(input, padding):
    # padding is left, right, top, bottom
    in_sizes = input.size()

    x_indices = [i for i in range(in_sizes[-1])]  # [0, 1, 2, 3, ...]
    left_indices = [i for i in range(padding[0] - 1, -1, -1)]  # e.g. [3, 2, 1, 0]
    right_indices = [-(i + 1) for i in range(padding[1])]  # e.g. [-1, -2, -3]
    x_indices = torch.tensor(left_indices + x_indices + right_indices)

    y_indices = [i for i in range(in_sizes[-2])]
    top_indices = [i for i in range(padding[2] - 1, -1, -1)]
    bottom_indices = [-(i + 1) for i in range(padding[3])]
    y_indices = torch.tensor(top_indices + y_indices + bottom_indices)

    ndim = input.ndim
    if ndim == 3:
        return input[:, y_indices[:, None], x_indices[None, :]]
    elif ndim == 4:
        return input[:, :, y_indices[:, None], x_indices[None, :]]
    else:
        raise RuntimeError("Symmetric padding of N-D tensors are not supported yet")


class Conv2DSymPadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=(1, 1), dilation=(1, 1), bias=True):

        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size
        else:
            kh = kw = kernel_size
        self.ph, self.pw = kh//2, kw//2

        super(Conv2DSymPadding, self).__init__(in_channels, out_channels, kernel_size,
                                               stride=stride, dilation=dilation, bias=bias, 
                                               padding=(self.ph, self.pw))

    def _conv_forward(self, input, weight):
        temp_conv = _pad_symmetric(input, (self.pw, self.pw, self.ph, self.ph))
        #print(temp_conv.shape)
        #print(weight.shape)
        #print(weight)
        #print('==================')
        
        #exit()
        #print('==================')
        #exit()
        return F.conv2d(temp_conv,weight, padding=_pair(0))

class BayarConstraint() :
    def __init__(self):
        self.mask = None
    def _initialize_mask( self, w):
        out_channels, in_channels, kernel_height, kernel_width = w.size()
        m = np.zeros((out_channels, in_channels, kernel_height, kernel_width)).astype('float32')
        m[:, :, kernel_height//2, kernel_width//2] = 1.
        device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mask = torch.tensor(m, device=device)
        return
    def __call__(self, w) :
        if self.mask is None :
            self._initialize_mask(w)
        w *= (1-self.mask)
        rest_sum = torch.sum( w, dim=(-1, -2), keepdim=True)
        w /= rest_sum + 1e-7 #K.epsilon() = 1e-7
        w -= self.mask
        return w

class CombinedConv2D(Conv2DSymPadding):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5),
                 stride=(1, 1), dilation=(1, 1), bias=False):
        super(CombinedConv2D, self).__init__(in_channels, out_channels, kernel_size,
                                             stride=stride, dilation=dilation, bias=bias)
        del self.weight
        self._build_all_kernel()
        self.bayar_constraint = BayarConstraint()
    
    def _get_srm_list(self):
        # srm kernel 1                                                                                                                                
        srm1 = np.zeros([5,5]).astype('float32')
        srm1[1:-1,1:-1] = np.array([[-1, 2, -1],
                                    [2, -4, 2],
                                    [-1, 2, -1]] )
        srm1 /= 4.
        # srm kernel 2                                                                                                                                
        srm2 = np.array([[-1, 2, -2, 2, -1],
                         [2, -6, 8, -6, 2],
                         [-2, 8, -12, 8, -2],
                         [2, -6, 8, -6, 2],
                         [-1, 2, -2, 2, -1]]).astype('float32')
        srm2 /= 12.
        # srm kernel 3                                                                                                                                
        srm3 = np.zeros([5,5]).astype('float32')
        srm3[2,1:-1] = np.array([1,-2,1])
        srm3 /= 2.
        return [ srm1, srm2, srm3 ]
    
    # torch -> C_out, C_in, kH, kW
    def _build_SRM_kernel(self):
        kernel = []
        srm_list = self._get_srm_list()
        for idx, srm in enumerate(srm_list):
            for ch in range(3):
                this_ch_kernel = np.zeros([3,5,5]).astype('float32')
                this_ch_kernel[ch,:,:] = srm
                kernel.append(this_ch_kernel)
        kernel = np.stack(kernel, axis=0)
        srm_kernel = nn.Parameter(torch.tensor(kernel), requires_grad=False)
        return srm_kernel

    def _build_all_kernel(self):
        # 1. regular conv kernels, fully trainable
        out_channels = self.out_channels - 9 - 3
        if out_channels >= 1:
            regular_kernel_shape = (out_channels, self.in_channels) + self.kernel_size
            self.regular_kernel = nn.Parameter(torch.ones(regular_kernel_shape))
            nn.init.xavier_uniform_(self.regular_kernel.data)
        else:
            self.regular_kernel = None
        # 2. SRM kernels, not trainable
        self.srm_kernel = self._build_SRM_kernel()
        # 3. bayar kernels, trainable but under constraint
        bayar_kernel_shape = (3, self.in_channels) + self.kernel_size
        self.bayar_kernel = nn.Parameter(torch.ones(bayar_kernel_shape))
        nn.init.xavier_uniform_(self.bayar_kernel.data)
        # 4. collect all kernels
        if self.regular_kernel is not None:
            all_kernels = [ self.regular_kernel,
                            self.srm_kernel,
                            self.bayar_kernel]
        else:
            all_kernels = [ self.srm_kernel,
                            self.bayar_kernel]
    
    def apply_bayar_constraint(self):
        self.bayar_kernel.data = self.bayar_constraint(self.bayar_kernel.data)
        return
    
    def forward(self, input):
        if self.regular_kernel is not None:
            regular_out = super(CombinedConv2D, self)._conv_forward(input, self.regular_kernel)
        srm_out = super(CombinedConv2D, self)._conv_forward(input, self.srm_kernel)
        #print(srm_out.shape)
        bayar_out = super(CombinedConv2D, self)._conv_forward(input, self.bayar_kernel)
        #print(bayar_out.shape)
        #exit()
        return srm_out,bayar_out


