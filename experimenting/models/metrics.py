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
        breakpoint()
        p_coords_max = torch.zeros((*y_gt.shape[:2], 2))
        confidence = torch.zeros(y_gt.shape[0]) # confidence is not used in this
        n_joints = y_gt.shape[1]

        gt_mask = y_gt.view(y_gt.size()[0], -1, n_joints).sum(1) > 0
        for j_idx in range(y_gt.shape[0]):
            pred_j_map = y_pr[:, j_idx]
            # Predicted max value for each heatmap. Keep only the first one if
            # more are present.

            p_coords_max[j_idx] = pred_j_map[pred_j_map == torch.max(pred_j_map)]
            # Confidence of the joint
            #confidence[j_idx] = np.max(y_pr)
        
        # where mask is 0, set gt back to NaN
        y_gt[gt_mask==0] = torch.nan
        dist_2d = np.linalg.norm((y_gt - p_coords_max), axis=-1)
        mpjpe = np.nanmean(dist_2d)
        return mpjpe

    def _get_max_indices(self, x):
        """
        https://stackoverflow.com/questions/53212507/how-to-efficiently-retrieve-the-indices-of-maximum-values-in-a-torch-tensor
        """
        n = torch.tensor(x.shape[0])
        d = torch.tensor(x.shape[-1])
        m = x.view(n, -1).argmax(1)
        indices = torch.cat(((m / d).view(-1, 1), (m % d).view(-1, 1)), dim=1)
        return indices
