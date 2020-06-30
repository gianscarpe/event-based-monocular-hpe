"""
DHP19 dataset implementations (classification, heatmap, joints)

"""

from os.path import basename, join

import numpy as np
import torch
from torch.utils.data import Dataset

from kornia import geometry
from pose3d_utils.camera import CameraIntrinsics
from pose3d_utils.skeleton_normaliser import SkeletonNormaliser

from ..utils import (
    MAX_CAM_HEIGHT,
    MAX_CAM_WIDTH,
    MOVEMENTS_PER_SESSION,
    N_JOINTS,
    _retrieve_2hm_files,
    get_label_from_filename,
    load_frame,
    load_heatmap,
)

__all__ = [
    'DHP19ClassificationDataset', 'DHPHeatmapDataset', 'DHPJointsDataset',
    'DHP3DJointsDataset'
]


class DHP19BaseDataset(Dataset):
    def __init__(self,
                 file_paths,
                 labels=None,
                 indexes=None,
                 transform=None,
                 augment_label=False):

        self.x_paths = file_paths
        self.x_indexes = indexes
        self.labels = labels
        self.transform = transform
        self.augment_label = augment_label

    def __len__(self):
        return len(self.x_indexes)

    def _get_x(self, idx):
        img_name = self.x_paths[idx]
        x = load_frame(img_name)
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


class DHP19ClassificationDataset(DHP19BaseDataset):
    def __init__(self,
                 file_paths,
                 labels=None,
                 indexes=None,
                 transform=None,
                 movements_per_session=MOVEMENTS_PER_SESSION):

        x_indexes = indexes if indexes is not None else np.arange(
            len(self.x_paths))
        self.movements_per_session = movements_per_session
        labels = labels if labels is not None else [
            get_label_from_filename(x_path, movements_per_session)
            for x_path in file_paths]

        super(DHP19ClassificationDataset,
              self).__init__(file_paths, labels, x_indexes, transform, False)

    def _get_y(self, idx):
        return self.labels[idx]


class DHPHeatmapDataset(DHP19BaseDataset):
    def __init__(self,
                 file_paths,
                 labels_dir,
                 indexes=None,
                 transform=None,
                 n_joints=N_JOINTS):

        labels = _retrieve_2hm_files(file_paths=file_paths,
                                     labels_dir=labels_dir)

        super(DHPHeatmapDataset, self).__init__(file_paths, labels, indexes,
                                                transform, True)

        self.n_joints = n_joints
        self.augment_label = True

    def _get_y(self, idx):
        joints_file = self.labels[idx]

        return load_heatmap(joints_file, self.n_joints)


class DHPJointsDataset(DHP19BaseDataset):
    def __init__(self,
                 file_paths,
                 labels_dir,
                 max_h=MAX_CAM_HEIGHT,
                 max_w=MAX_CAM_WIDTH,
                 indexes=None,
                 transform=None,
                 n_joints=N_JOINTS):

        labels = _retrieve_2hm_files(file_paths=file_paths,
                                     labels_dir=labels_dir)

        super(DHPJointsDataset, self).__init__(file_paths,
                                               labels,
                                               indexes,
                                               transform,
                                               augment_label=False)

        self.n_joints = n_joints
        self.max_h = max_h
        self.max_w = max_w

    def _retrieve_2hm_files(labels_dir, file_paths):
        labels_hm = [
            join(labels_dir,
                 basename(x).split('.')[0] + '_2dhm.npz') for x in file_paths
        ]
        return labels_hm

    def _get_y(self, idx):
        joints_file = np.load(self.labels[idx])

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


class DHP3DJointsDataset(DHP19BaseDataset):
    def __init__(self,
                 file_paths,
                 labels_dir,
                 height,
                 width,
                 indexes=None,
                 transform=None,
                 n_joints=N_JOINTS):

        labels = _retrieve_2hm_files(file_paths=file_paths,
                                     labels_dir=labels_dir)

        super(DHP3DJointsDataset, self).__init__(file_paths, labels, indexes,
                                                 transform, False)

        self.n_joints = n_joints
        self.normalizer = SkeletonNormaliser()
        self.height = height
        self.width = width

    def _get_y(self, idx):
        joints_file = np.load(self.labels[idx])

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
        M = torch.tensor(joints_file['M'])
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
            'M': M,
            'camera': camera,
            'mask': mask
        }
        return label
