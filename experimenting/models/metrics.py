import numpy as np
import torch
from torch import nn
from ..utils import get_heatmap_max

class BaseMetric(nn.Module):
    pass


class MPJPE(BaseMetric):

    def __init__(self, confidence=0, n_joints=13, **kwargs):
        super().__init__(**kwargs)
        self.confidence = confidence
        self.n_joints = n_joints
        
    def forward(self, y_pr, y_gt, mask=None):
        """

        y_pr = heatmap obtained with CNN
        y_gt = 2d points of joints, in order
        """
        gt_mask = y_gt.view(y_gt.size()[0], -1, self.n_joints).sum(1) > 0
        gt_coords, _ = get_heatmap_max(y_gt)
        gt_coords = gt_coords.type(torch.float)

        p_coords_max, confidence = get_heatmap_max(y_pr)
        
        # where mask is 0, set gt back to NaN

        gt_coords[gt_mask==0] = float('nan')
        dist_2d = torch.norm((gt_coords - p_coords_max), dim=-1)

        mpjpe = nanmean(dist_2d)
        return mpjpe

def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
        is_nan = torch.isnan(v)
        v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)
    
def _get_max_indices(self, x):
    """
        https://stackoverflow.com/questions/53212507/how-to-efficiently-retrieve-the-indices-of-maximum-values-in-a-torch-tensor
        """
    n = torch.tensor(x.shape[0])
    d = torch.tensor(x.shape[-1])
    m = x.view(n, -1).argmax(1)
    indices = torch.cat(((m / d).view(-1, 1), (m % d).view(-1, 1)), dim=1)
    return indices
