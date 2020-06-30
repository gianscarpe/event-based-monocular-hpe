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


class PCK(BaseMetric):
    def __init__(self, reduction, threshold=150, **kwargs):
        super().__init__(**kwargs)
        self.thr = threshold
        self.reduction = reduction

    def forward(self, y_pr, points_gt, gt_mask=None):
        points_gt[~gt_mask] = 0
        dist_2d = (torch.norm((points_gt - y_pr), dim=-1) < self.thr).double()
        if self.reduction:
            # To apply a reduction method (e.g. mean) we need a mask of gt
            # joints
            dist_2d = self.reduction(dist_2d, gt_mask)
        return dist_2d


class AUC(BaseMetric):
    def __init__(self, reduction, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def forward(self, y_pr, points_gt, gt_mask=None):
        # This range of thresholds mimics `mpii_compute_3d_pck.m`, which is provided as part of the
        # MPI-INF-3DHP test data release.

        thresholds = torch.linspace(0, 150, 31).tolist()

        pck_values = torch.DoubleTensor(len(thresholds))
        for i, threshold in enumerate(thresholds):
            _pck = PCK(self.reduction, threshold=threshold)
            pck_values[i] = _pck(y_pr, points_gt, threshold=threshold)

        if self.reduction:
            # To apply a reduction method (e.g. mean) we need a mask of gt
            # joints
            pck_values = self.reduction(pck_values, gt_mask)

        return pck_values
