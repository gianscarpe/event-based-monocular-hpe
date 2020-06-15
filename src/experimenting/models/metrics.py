import torch
from torch import nn


class BaseMetric(nn.Module):
    pass


class MPJPE(BaseMetric):
    def __init__(self, reduction=None, confidence=0, n_joints=13, **kwargs):
        super().__init__(**kwargs)
        self.confidence = confidence
        self.n_joints = n_joints
        self.reduction = reduction

    def forward(self, y_pr, points_gt, gt_mask=None):
        """

        y_pr = heatmap obtained with CNN
        y_gt = 2d points of joints, in order
        """
        points_gt[~gt_mask] = 0
        dist_2d = torch.norm((points_gt - y_pr), dim=-1)

        if self.reduction:
            # To apply a reduction method (e.g. mean) we need a mask of gt
            # joints

            dist_2d = self.reduction(dist_2d, gt_mask)
        return dist_2d
