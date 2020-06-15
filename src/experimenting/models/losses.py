from functools import reduce

import torch
import torch.nn as nn

from kornia.geometry import render_gaussian2d, spatial_expectation2d

from ..utils import SoftArgmax2D, average_loss, get_joints_from_heatmap, predict_xyz
from .metrics import MPJPE

__all__ = ['HeatmapLoss', 'PixelWiseLoss', 'MultiPixelWiseLoss']


class HeatmapLoss(nn.Module):
    """
    from https://github.com/anibali/margipose
    """
    def __init__(self, reduction='mask_mean', n_joints=13):
        """
        Args:
         reduction (String, optional): only "mask" methods allowed
        """
        super(HeatmapLoss, self).__init__()
        self.divergence = _js
        self.reduction = _get_reduction(reduction)
        self.soft_argmax = SoftArgmax2D(window_fn="Uniform")
        self.n_joints = n_joints

    def _mpjpe(self, y_pr, y_gt, reduce=False):
        """
        y_pr = heatmap obtained with CNN
        y_gt = 2d points of joints, in order
        """

        p_coords_max = self.soft_argmax(y_pr)
        gt_coords, _ = get_joints_from_heatmap(y_gt)

        dist_2d = torch.norm((gt_coords - p_coords_max), dim=-1)
        if self.reduction:
            # To apply a reduction method (e.g. mean) we need a mask of gt
            # joints
            gt_mask = y_gt.view(y_gt.size()[0], -1, self.n_joints).sum(1) > 0
            dist_2d = self.reduction(dist_2d, gt_mask)
        return dist_2d

    def forward(self, pred, gt):
        ndims = 2
        n_joints = pred.shape[1]

        loss = torch.add(self._mpjpe(pred, gt),
                         self.divergence(pred, gt, ndims))
        gt_mask = gt.view(gt.size()[0], -1, n_joints).sum(1) > 0

        return self.reduction(loss, gt_mask)


class PixelWiseLoss(nn.Module):
    """
    from https://github.com/anibali/margipose/
    """
    def __init__(self, reduction='mask_mean'):
        super(PixelWiseLoss, self).__init__()
        self.divergence = _divergence
        self.mpjpe = MPJPE()
        self.reduction = _get_reduction(reduction)
        self.sigma = 1

    def forward(self, pred_hm, gt_joints, gt_mask=None):
        if type(pred_hm) == tuple:
            pred_hm = pred_hm[0]
        gt_joints = gt_joints.narrow(-1, 0, 2)
        pred_joints = spatial_expectation2d(pred_hm)
        loss = torch.add(self.mpjpe(pred_joints, gt_joints, gt_mask),
                         self.divergence(pred_hm, gt_joints, self.sigma))

        return self.reduction(loss, gt_mask)


class MultiPixelWiseLoss(PixelWiseLoss):
    """
    from https://github.com/anibali/margipose
    """
    def __init__(self, reduction='mask_mean'):
        """
        Args:
         reduction (String, optional): only "mask" methods allowed
        """
        super(MultiPixelWiseLoss, self).__init__(reduction)

    def forward(self, pred_hm, gt_joints, gt_mask=None):
        loss = 0
        ndims = 2

        pred_xy_hm, pred_zy_hm, pred_xz_hm = pred_hm
        target_xy = gt_joints.narrow(-1, 0, 2)
        target_zy = torch.cat(
            [gt_joints.narrow(-1, 2, 1),
             gt_joints.narrow(-1, 1, 1)], -1)
        target_xz = torch.cat(
            [gt_joints.narrow(-1, 0, 1),
             gt_joints.narrow(-1, 2, 1)], -1)

        pred_joints = predict_xyz(pred_hm)

        loss += self.divergence(pred_xy_hm, target_xy, ndims)
        loss += self.divergence(pred_zy_hm, target_zy, ndims)
        loss += self.divergence(pred_xz_hm, target_xz, ndims)

        loss += self.mpjpe(pred_joints, gt_joints, gt_mask)
        result = self.reduction(loss, gt_mask)

        return result


def _divergence(pred_hm, gt_joints, sigma):
    ndims = 2
    sigma = torch.tensor([sigma, sigma],
                         dtype=gt_joints.dtype,
                         device=pred_hm.device)
    hm_dim = (pred_hm.shape[2], pred_hm.shape[3])

    gt_hm = render_gaussian2d(gt_joints, sigma, hm_dim)
    divergence = _js(pred_hm, gt_hm, ndims)
    return divergence


def _kl(p, q, ndims):
    eps = 1e-24
    unsummed_kl = p * ((p + eps).log() - (q + eps).log())
    kl_values = reduce(lambda t, _: t.sum(-1, keepdim=False), range(ndims),
                       unsummed_kl)
    return kl_values


def _js(p, q, ndims):
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m, ndims) + 0.5 * _kl(q, m, ndims)


def _get_reduction(reduction_type):
    switch = {'mean': torch.mean, 'mask_mean': average_loss, 'sum': torch.sum}

    return switch[reduction_type]
