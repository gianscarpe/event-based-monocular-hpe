import os
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
from scipy import io

from experimenting.utils import Skeleton


class BaseCore(ABC):
    """
    Base class for dataset cores. Each core should implement get_frame_info and
    load_frame_from_id for base functionalities. Labels, heatmaps, and joints
    loading may be implemented as well to use the relative task implementations
    """

    def __init__(self, name, partition):
        self._set_partition_function(partition)
        self.name = name

    def _set_partition_function(self, partition_param):
        if partition_param is None:
            partition_param = "cross-subject"

        if partition_param == "cross-subject":
            self.partition_function = self.get_cross_subject_partition_function()
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
        return (
            lambda x: type(self).get_frame_info(x)["subject"]
            in self.get_test_subjects()
        )

    def get_cross_view_partition_function(self):
        """
        Get partition function for cross-view evaluation method

        Note:
          Core class must implement get_test_view
          get_frame_info must provide frame's cam
        """

        return lambda x: type(self).get_frame_info(x)["cam"] in self.get_test_view()

    def get_test_subjects(self):
        raise NotImplementedError()

    def get_test_view(self):
        raise NotImplementedError()

    def get_frame_from_id(self, idx):
        raise NotImplementedError()

    def get_label_from_id(self, idx):
        raise NotImplementedError()

    def get_joint_from_id(self, idx) -> Tuple[Skeleton, torch.Tensor, torch.Tensor]:
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


def _split_set(data_indexes, split_at=0.8):
    np.random.shuffle(data_indexes)
    n_data_for_training = len(data_indexes)
    train_split = int(split_at * n_data_for_training)
    train_indexes = data_indexes[:train_split]
    val_indexes = data_indexes[train_split:]

    return train_indexes, val_indexes
