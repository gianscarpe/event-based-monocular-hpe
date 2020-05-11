import torch
import torch.nn as nn

__all__ = ['HeatmapLoss']

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
        l = l.mean(dim=self.channel_position).mean(dim=-1).mean(dim=-1)
        
        return self.reduction(l) ## l of dim bsize

def _get_reduction(reduction_type):
    switch = {'mean': torch.mean, 'sum': torch.sum}
    
    return switch[reduction_type]
                  
