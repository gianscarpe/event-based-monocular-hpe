"""
Core dataset implementation. BaseCore may be inherhit to create a new
DatasetCore. DHP19 and NTU cores are provided
"""

import os
from abc import ABC, abstractmethod

import numpy as np
from scipy import io

from ..utils import get_file_paths, load_heatmap


class BaseCore(ABC):
    """
    Base class for dataset cores. Each core should implement get_frame_info and
    load_frame_from_id for base functionalities. Labels, heatmaps, and joints
    loading may be implemented as well to use the relative task implementations
    """
    def __init__(self, hparams_dataset):
        self._set_partition_function(hparams_dataset)

    def _set_partition_function(self, hparams_dataset):
        partition_param = hparams_dataset.partition
        if (partition_param is None):
            partition_param = 'cross-subject'

        if (partition_param == 'cross-subject'):
            self.partition_function = self.get_cross_subject_partition_function(
            )
        else:
            self.partition_function = self.get_cross_view_partition_function()

    @staticmethod
    @abstractmethod
    def get_frame_info(path):
        """
        Get frame attributes given the path

        Args:
          path: frame path

        Returns:
          Frame attributes as a subscriptable object
        """

    def get_cross_subject_partition_function(self):
        """
        Get partition function for cross-subject evaluation method

        Note:
          Core class must implement get_test_subjects
          get_frame_info must provide frame's subject
        """
        return lambda x: type(self).get_frame_info(x)[
            'subject'] in self.get_test_subjects()

    def get_cross_view_partition_function(self):
        """
        Get partition function for cross-view evaluation method

        Note:
          Core class must implement get_test_view
          get_frame_info must provide frame's cam
        """

        return lambda x: type(self).get_frame_info(x)[
            'cam'] in self.get_test_view()

    def get_test_subjects(self):
        raise NotImplementedError()

    def get_test_view(self):
        raise NotImplementedError()

    def get_frame_from_id(self, idx):
        raise NotImplementedError()

    def get_label_from_id(self, idx):
        raise NotImplementedError()

    def get_joint_from_id(self, idx):
        raise NotImplementedError()

    def get_heatmap_from_id(self, idx):
        raise NotImplementedError()

    def get_train_test_split(self, split_at=0.8):
        """
        Get train, val, and test indexes accordingly to partition function

        Args:
            split_at: Split train/val according to a given percentage
        Returns:
            Train, val, and test indexes as numpy vectors
        """
        data_indexes = np.arange(len(self.file_paths))
        test_subject_indexes_mask = [
            self.partition_function(x) for x in self.file_paths
        ]

        test_indexes = data_indexes[test_subject_indexes_mask]
        data_index = data_indexes[~np.in1d(data_indexes, test_indexes)]
        train_indexes, val_indexes = _split_set(data_index, split_at=split_at)
        return train_indexes, val_indexes, test_indexes


class DHP19Core(BaseCore):
    """
    DHP19 dataset core class. It provides implementation to load frames,
    heatmaps, 2D joints, 3D joints
    """

    MOVEMENTS_PER_SESSION = {1: 8, 2: 6, 3: 6, 4: 6, 5: 7}
    MAX_WIDTH = 346
    MAX_HEIGHT = 260
    N_JOINTS = 13
    DEFAULT_TEST_SUBJECTS = [1, 2, 3, 4, 5]
    DEFAULT_TEST_VIEW = [1, 2]

    def __init__(self, hparams_dataset):
        super(DHP19Core, self).__init__(hparams_dataset)
        self.file_paths = DHP19Core._get_file_paths_with_cam_and_mov(
            hparams_dataset.data_dir, hparams_dataset.cams,
            hparams_dataset.movements)

        self.labels_dir = hparams_dataset.labels_dir
        self.classification_labels = [
            DHP19Core.get_label_from_filename(x_path)
            for x_path in self.file_paths
        ]

        self.joints = self._retrieve_2hm_files(hparams_dataset.joints_dir,
                                               'npz')
        self.heatmaps = self._retrieve_2hm_files(hparams_dataset.hm_dir, 'npy')

        if hparams_dataset.test_subjects is None:
            self.subjects = DHP19Core.DEFAULT_TEST_SUBJECTS
        else:
            self.subjects = hparams_dataset.test_subjects

        if hparams_dataset.test_cams is None:
            self.view = DHP19Core.DEFAULT_TEST_VIEW
        else:
            self.view = hparams_dataset.test_cams

    @staticmethod
    def get_standard_path(subject, session, movement, frame, cam, postfix=""):
        return "S{}_session_{}_mov_{}_frame_{}_cam_{}{}.npy".format(
            subject, session, movement, frame, cam, postfix)

    @staticmethod
    def load_frame(path):
        ext = os.path.splitext(path)[1]
        if ext == '.mat':
            x = DHP19Core._load_matlab_frame(path)
        elif ext == '.npy':
            x = np.load(path) / 255.
            if len(x.shape) == 2:
                x = np.expand_dims(x, -1)
        return x

    def _load_matlab_frame(path):
        """
        Matlab files contain voxelgrid frames and must be loaded properly.
        Information is contained respectiely in attributes: V1n, V2n, V3n, V4n

        Examples:
          S1_.mat

        """
        info = DHP19Core.get_frame_info(path)
        x = np.swapaxes(io.loadmat(path)[f'V{info["cam"] + 1}n'], 0, 1)
        return x

    def get_frame_from_id(self, idx):
        return DHP19Core.load_frame(self.file_paths[idx])

    def get_label_from_id(self, idx):
        return self.classification_labels[idx]

    def get_joint_from_id(self, idx):
        joints_file = np.load(self.joints[idx])
        return joints_file

    def get_heatmap_from_id(self, idx):
        hm_path = self.heatmaps[idx]
        return load_heatmap(hm_path, self.N_JOINTS)

    def _get_file_paths_with_cam_and_mov(data_dir, cams=None, movs=None):
        if cams is None:
            cams = [3]

        file_paths = np.array(
            get_file_paths(data_dir, extensions=['.npy', '.mat']))
        cam_mask = np.zeros(len(file_paths))

        for c in cams:
            cam_mask += [f'cam_{c}' in x for x in file_paths]

        file_paths = file_paths[cam_mask > 0]
        if movs is not None:
            mov_mask = [
                DHP19Core.get_label_from_filename(x) in movs
                for x in file_paths
            ]

            file_paths = file_paths[mov_mask]

        return file_paths

    @staticmethod
    def get_frame_info(filename):
        filename = os.path.splitext(os.path.basename(filename))[0]

        result = {
            'subject':
            int(filename[filename.find('S') + 1:filename.find('S') +
                         4].split('_')[0]),
            'session':
            int(DHP19Core._get_info_from_string(filename, 'session')),
            'mov':
            int(DHP19Core._get_info_from_string(filename, 'mov')),
            'cam':
            int(DHP19Core._get_info_from_string(filename, 'cam')),
            'frame':
            DHP19Core._get_info_from_string(filename, 'frame')
        }

        return result

    def get_test_subjects(self):
        return self.subjects

    def get_test_view(self):
        return self.view

    def _get_info_from_string(filename, info, split_symbol='_'):
        return int(filename[filename.find(info):].split(split_symbol)[1])


    @staticmethod
    def get_label_from_filename(filepath) -> int:
        """Given the filepath, return the correspondent movement label (range [0, 32])

        Args:
            filepath (str): frame absolute filepath

        Returns:
            Frame label

        Examples:
            >>> DHP19Core.get_label_from_filename("S1_session_2_mov_1_frame_249_cam_2.npy")
            8

        """

        label = 0
        info = DHP19Core.get_frame_info(filepath)

        for i in range(1, info['session']):
            label += DHP19Core.MOVEMENTS_PER_SESSION[i]

        return label + info['mov'] - 1  # label in range [0, max_label)

    def _retrieve_2hm_files(self, labels_dir, suffix):
        labels_hm = [
            os.path.join(labels_dir,
                         os.path.basename(x).split('.')[0] + f'_2dhm.{suffix}')
            for x in self.file_paths
        ]
        return labels_hm


class NTUCore(BaseCore):
    DEFAULT_TEST_SUBJECTS = [18, 19, 20]

    def __init__(self, hparams_dataset):
        super(NTUCore, self).__init__(hparams_dataset)
        self.file_paths = NTUCore._get_file_paths(hparams_dataset.data_dir)

        if hparams_dataset.test_subjects is None:
            self.subjects = NTUCore.DEFAULT_TEST_SUBJECTS
        else:
            self.subjects = hparams_dataset.test_subjects

    @staticmethod
    def load_frame(path):
        x = np.load(path) / 255.
        if len(x.shape) == 2:
            x = np.expand_dims(x, -1)
        return x

    def get_frame_from_id(self, idx):
        img_name = self.file_paths[idx]
        x = self.load_frame(img_name)
        return x

    @staticmethod
    def get_frame_info(path):

        dir_name = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(path))))
        info = {'subject': int(dir_name[-2:])}
        return info

    def get_test_subjects(self):
        return self.subjects

    @staticmethod
    def _get_file_paths(data_dir):
        file_paths = []
        for root, dirs, files in os.walk(data_dir):
            if 'part_' in root:
                for f in files:
                    file_path = os.path.join(root, f)
                    file_paths.append(file_path)
        return file_paths

    
def _split_set(data_indexes, split_at=0.8):
    np.random.shuffle(data_indexes)
    n_data_for_training = len(data_indexes)
    train_split = int(split_at * n_data_for_training)
    train_indexes = data_indexes[:train_split]
    val_indexes = data_indexes[train_split:]

    return train_indexes, val_indexes
