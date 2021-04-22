import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import numpy as np

class Central_Learning(nn.Module):

    def __init__(self, cfg):
        super(Central_Learning, self).__init__()

        self.n = cfg.DATALOADER.IMS_PER_BATCH
        self.k = cfg.DATALOADER.NUM_INSTANCE
        self.p = self.n // self.k
        self.s = 20

        self.p_m = 0.7
        self.n_m = 1 - self.p_m
        self.e_m = cfg.OPTIMIZER.E_MARGIN
    
    def euclidean_dist(self, x, y):
        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist = torch.addmm(input=dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
        # dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist 
    
    def score_func(self, dist):
        return torch.exp(-dist).sqrt()

    def forward(self, x_face, x_context, labels):
        
        self.n = x_face.shape[0]
        self.p = self.n // self.k

        center_labels = torch.from_numpy(np.repeat(range( self.p ), self.k )).long().cuda()
        center_mask = torch.zeros((self.n, self.p)).cuda()
        center_mask.scatter_(1, center_labels.view(-1, 1).long(), 1)
        
        # define dominant sample
        center = x_face.contiguous().view(self.p, self.k, -1).mean(dim=1)

        cc_score = self.score_func(self.euclidean_dist(center,  center))
        xc_dist  = self.euclidean_dist(x_context, center)
        if self.e_m > 0:
            xc_dist  = center_mask * (xc_dist + self.e_m) + (1 - center_mask) * xc_dist
        xc_score = self.score_func(xc_dist)
        
        s = self.s
        tgt_mask  = (1 - torch.eye(self.p)).cuda().bool()
        orth_loss = (cc_score[tgt_mask]).norm(p=2) / (self.p**2) * s
        center_pos_loss = F.relu(self.p_m - xc_score[center_mask.bool()]).contiguous().view(self.n, -1)
        center_neg_loss = F.relu(xc_score[(1 - center_mask).bool()] - (self.n_m)).contiguous().view(self.n, -1)
        central_loss    = torch.cat([center_pos_loss, center_neg_loss], dim=1)
        central_loss    = (central_loss).norm(p=2) / (self.p * self.n ) * s
        
        ce_loss = F.cross_entropy(xc_score * s, center_labels)
        loss = central_loss + orth_loss + ce_loss
        return loss
