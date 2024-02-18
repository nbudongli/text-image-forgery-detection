
import torch.nn as nn

crossentropyloss = nn.CrossEntropyLoss()

def similar_loss(b, cls, fea, alpha, m):
    y = (cls.unsqueeze(0) != cls.unsqueeze(1)).float().view(-1)
    dist = ((fea.unsqueeze(0) - fea.unsqueeze(1)) ** 2)
    dist = dist.sum(dim=2).view(-1)
    dist_p = (1-y)/2*(dist-float(m/3)).clamp(min=0)
    dist_n = y / 2 * (m - dist).clamp(min=0)
    loss = dist_p + dist_n
    return crossentropyloss(b, cls)+alpha * loss.mean()