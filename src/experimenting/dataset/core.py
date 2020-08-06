import os
from abc import ABC, abstractmethod

import numpy as np
from scipy import io

from ..utils import get_file_paths, load_heatmap


class BaseCore(ABC):
    def __init__(self, hparams_dataset):
        self.hparams_dataset = hparams_dataset
        self._set()
        self._set_train_test_split(self.hparams_dataset.split_at)

        if hparams_dataset.save_split:
            self._save_params(hparams_dataset.preload_dir)

    @abstractmethod
    def _set(self):
        pass

    @abstractmethod
    def get_frame_info(x):
        pass

    @abstractmethod
    def get_partition_function(self):
        pass

    def load_frame_from_id(self, idx):
        raise NotImplementedError()

    def get_label_from_id(self, idx):
        raise NotImplementedError()

    def get_joint_from_id(self, idx):
        raise NotImplementedError()

    def get_heatmap_from_id(self, idx):
        raise NotImplementedError()

    def _set_train_test_split(self, split_at):
        data_indexes = np.arange(len(self.file_paths))
        cond = self.get_partition_function()
        test_subject_indexes_mask = [cond(x) for x in self.file_paths]

        self.test_indexes = data_indexes[test_subject_indexes_mask]
        data_index = data_indexes[~np.in1d(data_indexes, self.test_indexes)]
        self.train_indexes, self.val_indexes = _split_set(data_index,
                                                          split_at=split_at)


class DHP19Core(BaseCore):
    MOVEMENTS_PER_SESSION = {1: 8, 2: 6, 3: 6, 4: 6, 5: 7}
    max_w = 346
    max_h = 260

    n_joints = 13

    def __init__(self, hparams_dataset):
        super(DHP19Core, self).__init__(hparams_dataset)

    @staticmethod
    def load_frame(path):
        ext = os.path.splitext(path)[1]
        if ext == '.mat':
            info = DHP19Core.get_frame_info(path)
            x = np.swapaxes(io.loadmat(path)[f'V{info["cam"] + 1}n'], 0, 1)
        elif ext == '.npy':
            x = np.load(path) / 255.
            if len(x.shape) == 2:
                x = np.expand_dims(x, -1)
        return x

    def load_frame_from_id(self, idx):
        return DHP19Core.load_frame(self.file_paths[idx])

    def _set(self):
        self.file_paths = DHP19Core._get_file_paths_with_cam(
            self.hparams_dataset.data_dir, self.hparams_dataset.cams)

        self.labels_dir = self.hparams_dataset.labels_dir
        self.classification_labels = [
            DHP19Core.get_label_from_filename(x_path)
            for x_path in self.file_paths
        ]
        self.joints = self._retrieve_2hm_files(self.hparams_dataset.joints_dir)
        self.heatmaps = self._retrieve_2hm_files(self.hparams_dataset.hm_dir)

        if self.hparams_dataset.test_subjects is None:
            self.subjects = [1, 2, 3, 4, 5]
        else:
            self.subjects = self.hparams_dataset.test_subjects

        if self.hparams_dataset.movements is None or self.hparams_dataset.movements == 'all':
            self.movements = range(0, 34)
        else:
            self.movements = self.hparams_dataset.movements

    def get_label_from_id(self, idx):
        return self.classification_labels[idx]

    def get_joint_from_id(self, idx):
        joints_file = np.load(self.joints[idx])
        return joints_file

    def get_heatmap_from_id(self, idx):
        hm_path = self.heatmaps[idx]
        return load_heatmap(hm_path, self.n_joints)

    def get_partition_function(self):
        return lambda x: DHP19Core.get_frame_info(x)[
            'subject'] in self.subjects and DHP19Core.get_label_from_filename(
                x) in self.movements

    def _get_file_paths_with_cam(data_dir, cams=None):

        if cams is None:
            cams = [3]

        file_paths = np.array(
            get_file_paths(data_dir, extensions=['.npy', '.mat']))

        cam_mask = [
            DHP19Core.get_frame_info(x)['cam'] in cams for x in file_paths
        ]

        file_paths = file_paths[cam_mask]

        return file_paths

    def get_frame_info(filename):
        filename = os.path.splitext(os.path.basename(filename))[0]

        result = {
            'subject':
            int(filename[filename.find('S') + 1:filename.find('S') +
                         4].split('_')[0]),
            'session':
            DHP19Core._get_info_from_string(filename, 'session'),
            'mov':
            DHP19Core._get_info_from_string(filename, 'mov'),
            'cam':
            DHP19Core._get_info_from_string(filename, 'cam'),
            'frame':
            DHP19Core._get_info_from_string(filename, 'frame')
        }

        return result

    def _get_info_from_string(filename, info, split_symbol='_'):
        return int(filename[filename.find(info):].split(split_symbol)[1])

    def get_label_from_filename(filepath):
        """Given the filepath, return the correspondent label E.g. n
        S1_session_2_mov_1_frame_249_cam_2.npy
        """

        label = 0
        info = DHP19Core.get_frame_info(filepath)

        for i in range(1, info['session']):
            label += DHP19Core.MOVEMENTS_PER_SESSION[i]

        return label + info['mov'] - 1

    def _retrieve_2hm_files(self, labels_dir):
        labels_hm = [
            os.path.join(labels_dir,
                         os.path.basename(x).split('.')[0] + '_2dhm.npz')
            for x in self.file_paths
        ]
        return labels_hm


class NTUCore(BaseCore):
    def __init__(self, hparams_dataset):
        super(NTUCore, self).__init__(hparams_dataset)

    def _set(self):
        self.file_paths = NTUCore._get_file_paths(
            self.hparams_dataset.data_dir)

        if self.hparams_dataset.test_subjects is None:
            self.subjects = [18, 19, 20]
        else:
            self.subjects = self.hparams_dataset.test_subjects

    @staticmethod
    def load_frame(path):
        x = np.load(path) / 255.
        if len(x.shape) == 2:
            x = np.expand_dims(x, -1)
        return x

    def load_frame_from_id(self, idx):
        img_name = self.file_paths[idx]
        x = self.load_frame(img_name)
        return x

    @staticmethod
    def get_frame_info(path):

        dir_name = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(path))))
        info = {}
        info['subject'] = int(dir_name[-2:])
        return info

    def _get_file_paths(data_dir):
        file_paths = []
        for root, dirs, files in os.walk(data_dir):
            if 'part_' in root:
                for f in files:
                    file_path = os.path.join(root, f)
                    file_paths.append(file_path)
        return file_paths

    def get_partition_function(self):
        return lambda x: NTUCore.get_frame_info(x)['subject'] in self.subjects


def _split_set(data_indexes, split_at=0.8):
    np.random.shuffle(data_indexes)
    n_data_for_training = len(data_indexes)
    train_split = int(split_at * n_data_for_training)
    train_indexes = data_indexes[:train_split]
    val_indexes = data_indexes[train_split:]

    return train_indexes, val_indexes
