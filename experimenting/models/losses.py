import torch
import torch.nn as nn
from operator import mul
from .metrics import MPJPE
from functools import reduce

__all__ = ['HeatmapLoss', 'PixelWiseLoss']

class HeatmapLoss(nn.Module):
    """
    loss for detection heatmap
    from https://github.com/princeton-vl/pytorch_stacked_hourglass/blob/master/task/loss.py
    """
    def __init__(self, reduction='mean'):
        super(HeatmapLoss, self).__init__()
        self.reduction = _get_reduction(reduction)
        self.channel_position = 1 #Channel_first for pytorch

    def forward(self, pred, gt):

        l = ((pred - gt)**2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        
        return self.reduction(l) ## l of dim bsize

class PixelWiseLoss(nn.Module):
    """
    from https://github.com/anibali/margipose/blob/a9dbe5c3151d7a7e071df6275d5702c07ef5152d/src/margipose/models/margipose_model.py#L202
    """
    def __init__(self, reduction='mean'):
        super(PixelWiseLoss, self).__init__()
        self.sigma = 1.0
        self.divergence = _js
        self.mpjpe = MPJPE()

    
    def forward(self, pred, gt):
        ndims = 2
        n_joints = pred.shape[1]
        loss = self.divergence(pred, gt, ndims)
        loss += self.mpjpe(pred, gt)
        mask = gt.view(gt.size()[0], -1, n_joints).sum(1) >0
        loss = loss * mask
        denom = mask.sum()
        return loss.sum() / denom        
        

def _kl(p, q, ndims):
    eps = 1e-24
    unsummed_kl = p * ((p + eps).log() - (q + eps).log())
    kl_values = reduce(lambda t, _: t.sum(-1, keepdim=False), range(ndims), unsummed_kl)
    return kl_values


def _js(p, q, ndims):
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m, ndims) + 0.5 * _kl(q, m, ndims)

def _get_reduction(reduction_type):
    switch = {'mean': torch.mean, 'sum': torch.sum}
    
    return switch[reduction_type]



