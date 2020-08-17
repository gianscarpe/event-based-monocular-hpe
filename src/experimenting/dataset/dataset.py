"""
Task dataset implementations.
Provided:
- classification: frame + label
- heatmap: frame + 2d gaussian heatmap
- joints: frame + projected 2d joints
- 3djoints: frame + 3d joints + camera information
- autoencoder: frame

"""

import numpy as np
import torch
from kornia import geometry
from pose3d_utils.camera import CameraIntrinsics
from pose3d_utils.skeleton_normaliser import SkeletonNormaliser
from torch.utils.data import Dataset

__all__ = [
    'ClassificationDataset', 'HeatmapDataset', 'JointsDataset',
    'Joints3DDataset', 'AutoEncoderDataset'
]


class BaseDataset(Dataset):
    def __init__(self,
                 dataset,
                 indexes=None,
                 transform=None,
                 augment_label=False):

        self.dataset = dataset
        self.x_indexes = indexes
        self.transform = transform
        self.augment_label = augment_label

    def __len__(self):
        return len(self.x_indexes)

    def _get_x(self, idx):
        """
        Basic get_x utility. Load frame from dataset given frame index

        Args:
          idx: Input index (from 0 to len(self))

        Returns:
          Input loaded from dataset
        """
        x = self.dataset.load_frame_from_id(idx)
        return x

    def __getitem__(self, idx):
        idx = self.x_indexes[idx]
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self._get_x(idx)
        y = self._get_y(idx)

        if self.transform:
            if self.augment_label:
                augmented = self.transform(image=x, mask=y)
                x = augmented['image']
                y = augmented['mask']
                y = torch.squeeze(y.transpose(0, -1))
            else:
                augmented = self.transform(image=x)
                x = augmented['image']
        return x, y


class ClassificationDataset(BaseDataset):
    def __init__(self, dataset, indexes=None, transform=None):

        x_indexes = indexes if indexes is not None else np.arange(
            len(self.x_paths))

        super(ClassificationDataset, self).__init__(dataset, x_indexes,
                                                    transform, False)

    def _get_y(self, idx):
        return self.dataset.get_label_from_id(idx)


class AutoEncoderDataset(BaseDataset):
    def __init__(self, dataset, indexes=None, transform=None):

        x_indexes = indexes if indexes is not None else np.arange(
            len(self.x_paths))
        super(AutoEncoderDataset, self).__init__(dataset, x_indexes, transform,
                                                 False)

    def __getitem__(self, idx):
        idx = self.x_indexes[idx]
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self._get_x(idx)

        if self.transform:
            augmented = self.transform(image=x)
            x = augmented['image']
        return x


class HeatmapDataset(BaseDataset):
    def __init__(self, dataset, indexes=None, transform=None):

        super(HeatmapDataset, self).__init__(dataset, indexes, transform, True)

    def _get_y(self, idx):
        return self.dataset.get_heatmap_from_id(idx)


class JointsDataset(BaseDataset):
    def __init__(self, dataset, indexes=None, transform=None):

        super(JointsDataset, self).__init__(dataset,
                                            indexes,
                                            transform,
                                            augment_label=False)

        self.max_h = dataset.max_h
        self.max_w = dataset.max_w

    def _get_y(self, idx):
        joints_file = self.dataset.get_joint_from_id(idx)

        joints = torch.tensor(joints_file['joints'])
        mask = torch.tensor(joints_file['mask']).type(torch.bool)
        return geometry.normalize_pixel_coordinates(joints, self.max_h,
                                                    self.max_w), mask

    def __getitem__(self, idx):
        idx = self.x_indexes[idx]
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self._get_x(idx)
        y, mask = self._get_y(idx)

        if self.transform:
            augmented = self.transform(image=x)
            x = augmented['image']

        return x, y, mask


class Joints3DDataset(BaseDataset):
    def __init__(self, dataset, in_shape, indexes=None, transform=None):

        super(Joints3DDataset, self).__init__(dataset, indexes, transform,
                                              False)

        self.n_joints = dataset.n_joints
        self.normalizer = SkeletonNormaliser()
        self.height = in_shape[0]
        self.width = in_shape[1]

    def _get_y(self, idx):
        joints_file = self.dataset.get_joint_from_id(idx)

        joints = torch.tensor(joints_file['xyz_cam'].swapaxes(0, 1))
        xyz = torch.tensor(joints_file['xyz'].swapaxes(0, 1))
        mask = ~torch.isnan(joints[:, 0])
        joints[~mask] = 0
        xyz[~mask] = 0
        skeleton = torch.cat(
            [joints,
             torch.ones((self.n_joints, 1), dtype=joints.dtype)],
            axis=1)

        z_ref = joints[4][2]
        camera = torch.tensor(joints_file['camera'])
        extrinsic_matrix = torch.tensor(joints_file['M'])
        # TODO: select a standard format for joints (better 3xnum_joints)

        normalized_skeleton = self.normalizer.normalise_skeleton(
            skeleton, z_ref, CameraIntrinsics(camera), self.height,
            self.width).narrow(-1, 0, 3)

        normalized_skeleton[~mask] = 0
        if torch.isnan(normalized_skeleton).any():
            breakpoint()

        label = {
            'xyz': xyz,
            'skeleton': joints,
            'normalized_skeleton': normalized_skeleton,
            'z_ref': z_ref,
            'M': extrinsic_matrix,
            'camera': camera,
            'mask': mask
        }
        return label
