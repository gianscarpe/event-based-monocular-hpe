import torch
from torch import nn

from kornia.geometry import spatial_softmax2d

__all__ = ['average_loss', 'nanmean', 'FlatSoftmax']


def average_loss(losses, mask=None):
    """Calculate the average of per-location losses.

    Args:
        losses (Tensor): Predictions (B x L)
        mask (Tensor, optional): Mask of points to include in the loss calculation
            (B x L), defaults to including everything
    """
    
    if mask is not None:
        assert mask.size() == losses.size(), 'mask must be the same size as losses'
        losses = losses * mask
        denom = mask.sum()
    else:
        denom = losses.numel()

    # Prevent division by zero
    if isinstance(denom, int):
        denom = max(denom, 1)
    else:
        denom = denom.clamp(1)

    return losses.sum() / denom


def nanmean(v,  *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
        is_nan = torch.isnan(v)
        v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


class FlatSoftmax(nn.Module):
    def __init__(self):
        super(FlatSoftmax, self).__init__()

    def forward(self, inp):
        return spatial_softmax2d(inp)
