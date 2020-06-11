"""
DHP19 dataset implementations (classification, heatmap, joints)

"""

from os.path import basename, join

import numpy as np
import torch
from torch.utils.data import Dataset

from kornia import geometry

from ..utils import (
    MAX_CAM_HEIGHT,
    MAX_CAM_WIDTH,
    MAX_X,
    MAX_Y,
    MAX_Z,
    MOVEMENTS_PER_SESSION,
    N_JOINTS,
    get_label_from_filename,
    load_frame,
    load_heatmap,
)

__all__ = [
    'DHP19ClassificationDataset', 'DHPHeatmapDataset', 'DHPJointsDataset', 'DHP3DJointsDataset'
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
            for x_path in file_paths
        ]

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

        labels = DHPHeatmapDataset._retrieve_joints_files(
            file_paths=file_paths, labels_dir=labels_dir)

        super(DHPHeatmapDataset, self).__init__(file_paths, labels, indexes,
                                                transform, True)

        self.n_joints = n_joints
        self.augment_label = True

    def _retrieve_joints_files(labels_dir, file_paths):
        labels_hm = [
            join(labels_dir,
                 basename(x).split('.')[0] + '_2dhm.npy') for x in file_paths
        ]
        return labels_hm

    def _get_y(self, idx):
        joints_file = self.labels[idx]

        return load_heatmap(joints_file, self.n_joints)

    def __getitem__(self, idx):
        idx = self.x_indexes[idx]
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self._get_x(idx)
        y = self._get_y(idx)

        if self.transform:
            augmented = self.transform(image=x, mask=y)
            y = augmented['mask']
            y = torch.squeeze(y.transpose(0, -1))
            x = augmented['image']

        return x, y


class DHPJointsDataset(DHP19BaseDataset):
    def __init__(self,
                 file_paths,
                 labels_dir,
                 max_h=MAX_Y,
                 max_w=MAX_X,
                 indexes=None,
                 transform=None,
                 n_joints=N_JOINTS):

        labels = DHPJointsDataset._retrieve_2hm_files(file_paths=file_paths,
                                                      labels_dir=labels_dir)

        super(DHPJointsDataset, self).__init__(file_paths, labels, indexes,
                                               transform, True)

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

        joints = torch.tensor(joints_file['xyz'].swapaxes(0, 1))[:, 0:2]
        mask = torch.tensor(joints_file['mask']).type(torch.bool)
        joints[~mask] = 0
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
                 max_x=MAX_X,
                 max_y=MAX_Y,
                 max_z=MAX_Z,
                 indexes=None,
                 transform=None,
                 n_joints=N_JOINTS):

        labels = DHPJointsDataset._retrieve_2hm_files(file_paths=file_paths,
                                                      labels_dir=labels_dir)

        super(DHP3DJointsDataset, self).__init__(file_paths, labels, indexes,
                                                 transform, True)

        self.n_joints = n_joints
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z

    def _retrieve_2hm_files(labels_dir, file_paths):
        labels_hm = [
            join(labels_dir,
                 basename(x).split('.')[0] + '_2dhm.npz') for x in file_paths
        ]
        return labels_hm

    def _get_y(self, idx):
        joints_file = np.load(self.labels[idx])

        joints = torch.tensor(joints_file['xyz'].swapaxes(0, 1))
        mask = ~torch.isnan(joints[:, 0])
        joints[~mask] = 0
        return geometry.normalize_pixel_coordinates3d(joints, self.max_z,
                                                      self.max_y,
                                                      self.max_z), mask

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
