import torch
import torch.nn as nn

__all__ = ['HeatmapLoss']

class HeatmapLoss(nn.Module):
    """
    loss for detection heatmap
    from https://github.com/princeton-vl/pytorch_stacked_hourglass/blob/master/task/loss.py
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        l = ((pred - gt)**2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l ## l of dim bsize
