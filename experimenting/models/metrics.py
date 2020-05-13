import numpy as np
import torch
from torch import nn
from ..utils import  get_joints_from_heatmap
from .dsnt import euclidean_losses
from .soft_argmax import SoftArgmax2D

class BaseMetric(nn.Module):
    pass


class MPJPE(BaseMetric):

    def __init__(self, reduction=None, confidence=0, n_joints=13, **kwargs):
        super().__init__(**kwargs)
        self.confidence = confidence
        self.n_joints = n_joints
        self.reduction = reduction
        self.soft_argmax = SoftArgmax2D()
        
    def forward(self, y_pr, y_gt, mask=None):
        """

        y_pr = heatmap obtained with CNN
        y_gt = 2d points of joints, in order
        """

        p_coords_max = self.soft_argmax(y_pr)
        gt_coords = self.soft_argmax(y_gt)

        dist_2d = euclidean_losses(p_coords_max, gt_coords)
        
        if self.reduction:
            # To apply a reduction method (e.g. mean) we need a mask of gt
            # joints
            
            mask = y_gt.view(y_gt.size()[0], -1, self.n_joints).sum(1) >0
            dist_2d = self.reduction(dist_2d, mask)
        return dist_2d
