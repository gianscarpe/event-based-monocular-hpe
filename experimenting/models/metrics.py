import numpy as np
import torch
from torch import nn

class BaseMetric(nn.Module):
    pass


class MPJPE(BaseMetric):

    def __init__(self, confidence=0, **kwargs):
        super().__init__(**kwargs)
        self.confidence = confidence
        
    def forward(self, y_pr, y_gt):
        confidence = torch.zeros(y_gt.shape[0]) # confidence is not used in this
        batch_size = y_gt.shape[0]
        n_joints = y_gt.shape[1]

        gt_mask = y_gt.view(y_gt.size()[0], -1, n_joints).sum(1) > 0

        p_coords_max = torch.zeros_like(y_gt)
        for b in range(batch_size):
            for j in range(n_joints):
                p_coords_max[b][j] = y_pr[b][j] == torch.max(y_pr[b][j])
                # Confidence of the joint
                #confidence[j_idx] = np.max(y_pr)
        
        # where mask is 0, set gt back to NaN

        y_gt[gt_mask==0] = float('nan')
        dist_2d = torch.norm((y_gt - p_coords_max), dim=-1)

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
