
import torch.nn.functional as F
import torch.nn as nn
import segmentation_models_pytorch.losses as sml
from segmentation_models_pytorch.losses import LovaszLoss
######## WeightedBCE Loss ###########
class WeightedBCE(nn.Module):
    def __init__(self, weights=[0.2, 0.8]):
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, logit_pixel, truth_pixel):
        logit = logit_pixel.reshape(-1)
        truth = truth_pixel.reshape(-1)
        assert(logit.shape==truth.shape)
        loss = F.binary_cross_entropy(logit, truth, reduction='mean')
        pos = (truth>=0.35).float()
        neg = (truth<0.35).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (self.weights[0]*pos*loss/pos_weight + self.weights[1]*neg*loss/neg_weight).sum()
        return loss



######## WeightedDice Loss ###########
class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.5]): # W_pos=0.8, W_neg=0.2
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights

    def forward(self, logit, truth, smooth=1e-5):
        batch_size = len(logit)
        logit = logit.reshape(batch_size,-1)
        truth = truth.reshape(batch_size,-1)
        assert(logit.shape==truth.shape)
        p = logit.view(batch_size,-1)
        t = truth.view(batch_size,-1)

        w = truth.detach()
        w = w*(self.weights[1]-self.weights[0])+self.weights[0]
        p = w*(p)
        t = w*(t)
        intersection = (p * t).sum(-1)
        union =  (p * p).sum(-1) + (t * t).sum(-1)
        dice  = 1 - (2*intersection + smooth) / (union +smooth)
        loss = dice.mean()
        return loss


######## Total Loss  = WeightedDice Loss  + WeightedBCE###########
class WeightedDiceBCE(nn.Module):
    def __init__(self,dice_weight=1,BCE_weight=1):
        super(WeightedDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.8, 0.2])
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.lovasz_loss = sml.LovaszLoss(mode='binary')
        self.BCE_weight = BCE_weight
        self.lovasz_weight = 0
        self.dice_weight = dice_weight
        
    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        #lovasz = self.lovasz_loss(inputs, targets)
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE# + lovasz * self.lovasz_weight
        return dice_BCE_loss

######## Total Loss  = WeightedDice Loss  + WeightedBCE###########
class WeightedLovaszBCE(nn.Module):
    def __init__(self,dice_weight=1,BCE_weight=1):
        super(WeightedLovaszBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.8, 0.2])
        self.lovasz_loss = sml.LovaszLoss(mode='binary')
        self.BCE_weight = BCE_weight
        self.lovasz_weight = dice_weight
        
    def forward(self, inputs, targets):
        lovasz = self.lovasz_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        dice_BCE_loss = self.lovasz_weight * lovasz + self.BCE_weight * BCE
        return dice_BCE_loss