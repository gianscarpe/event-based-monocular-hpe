from functools import reduce

import torch
import torch.nn as nn

from kornia.geometry import render_gaussian2d, spatial_expectation2d

from ..utils import SoftArgmax2D, average_loss, get_joints_from_heatmap
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
        self.divergence = _js
        self.mpjpe = MPJPE()
        self.reduction = _get_reduction(reduction)
        self.sigma = 0.001

    def forward(self, pred_hm, gt_joints, gt_mask=None):
        ndims = 2
        sigma = torch.tensor([self.sigma, self.sigma], device=pred_hm.device)
        n_joints = pred_hm.shape[1]
        hm_dim = (pred_hm.shape[2], pred_hm.shape[3])

        gt_hm = render_gaussian2d(gt_joints, sigma, hm_dim)
        pred_joints = spatial_expectation2d(pred_hm)

        if gt_mask is None:
            gt_mask = gt_joints.view(gt_joints.size()[0], -1,
                                     n_joints).sum(1) != 0

        loss = torch.add(self.mpjpe(pred_joints, gt_joints, gt_mask),
                         self.divergence(pred_hm, gt_hm, ndims))
        return self.reduction(loss, gt_mask)


class MultiPixelWiseLoss(nn.Module):
    """
    from https://github.com/anibali/margipose
    """
    def __init__(self, reduction='mask_mean'):
        """
        Args:
         reduction (String, optional): only "mask" methods allowed
        """
        super(MultiPixelWiseLoss, self).__init__()
        self.divergence = _js
        self.mpjpe = MPJPE()
        self.reduction = _get_reduction(reduction)
        self.sigma = 0.001

    def forward(self, pred_hm, gt_joints, gt_mask=None):
        loss = 0
        ndims = 2
        n_joints = pred_hm[0].shape[1]
        hm_dim = (pred_hm[0].shape[2], pred_hm[0].shape[3])

        target_xy = gt_joints.narrow(-1, 0, 2)
        target_zy = torch.cat(
            [gt_joints.narrow(-1, 2, 1),
             gt_joints.narrow(-1, 1, 1)], -1)
        target_xz = torch.cat(
            [gt_joints.narrow(-1, 0, 1),
             gt_joints.narrow(-1, 2, 1)], -1)

        device_xy = target_xy.device
        device_zy = target_zy.device
        device_xz = target_xz.device
        dtype = target_xy.dtype
        sigma = torch.tensor([self.sigma, self.sigma], dtype=dtype)
        
        gt_xy_hm = render_gaussian2d(target_xy, sigma.to(device_xy), hm_dim)
        gt_zy_hm = render_gaussian2d(target_zy, sigma.to(device_zy), hm_dim)
        gt_xz_hm = render_gaussian2d(target_xz, sigma.to(device_xz), hm_dim)

        pred_xy_hm, pred_zy_hm, pred_xz_hm = pred_hm
        xy = spatial_expectation2d(pred_xy_hm)
        zy = spatial_expectation2d(pred_zy_hm)
        xz = spatial_expectation2d(pred_xz_hm)
        x, y = xy.split(1, -1)
        z = 0.5 * (zy[:, :, 0:1] + xz[:, :, 1:2])

        pred_joints = torch.cat([x, y, z], -1)

        if gt_mask is None:
            gt_mask = gt_joints.view(gt_joints.size()[0], -1,
                                     n_joints).sum(1) != 0
        loss += self.divergence(pred_xy_hm, gt_xy_hm, ndims)
        loss += self.divergence(pred_zy_hm, gt_zy_hm, ndims)
        loss += self.divergence(pred_xz_hm, gt_xz_hm, ndims)

        loss += self.mpjpe(pred_joints, gt_joints, gt_mask)

        return self.reduction(loss, gt_mask)


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
