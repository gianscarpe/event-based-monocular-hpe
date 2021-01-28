"""
Factory module provide a set of Constructor for datasets using factory design
pattern.  It encapsulates core dataset implementation and implement
functionalities to get train, val, and test dataset
"""

from abc import ABC
from typing import Tuple

from torch.utils.data import Dataset

from ..utils import get_augmentation
from . import core
from .dataset import (
    AutoEncoderDataset,
    BaseDataset,
    ClassificationDataset,
    HeatmapDataset,
    Joints3DDataset,
    JointsDataset,
)

__all__ = [
    "BaseConstructor",
    "ClassificationConstructor",
    "AutoEncoderConstructor",
    "Joints3DConstructor",
    "JointsConstructor",
    "HeatmapConstructor",
]


class BaseConstructor(ABC):
    def __init__(self, dataset_task):
        self.dataset_task = dataset_task

    def get_datasets(
        self, core_dataset, augmentation_train, augmentation_test, **kwargs
    ) -> Tuple[Dataset, Dataset, Dataset]:
        train_indexes, val_indexes, test_indexes = core_dataset.get_train_test_split()
        preprocess_train = get_augmentation(augmentation_train)
        preprocess_val = get_augmentation(augmentation_test)

        return (
            self.dataset_task(
                dataset=core_dataset,
                indexes=train_indexes,
                transform=preprocess_train,
                **kwargs
            ),
            self.dataset_task(
                dataset=core_dataset,
                indexes=val_indexes,
                transform=preprocess_val,
                **kwargs
            ),
            self.dataset_task(
                dataset=core_dataset,
                indexes=test_indexes,
                transform=preprocess_val,
                **kwargs
            ),
        )


class ClassificationConstructor(BaseConstructor):
    def __init__(self, hparams):
        super(ClassificationConstructor, self).__init__(hparams, ClassificationDataset)


class JointsConstructor(BaseConstructor):
    def __init__(self, hparams):
        super(JointsConstructor, self).__init__(hparams, JointsDataset)


class Joints3DConstructor(BaseConstructor):
    def __init__(self):
        super(Joints3DConstructor, self).__init__(dataset_task=Joints3DDataset,)


class HeatmapConstructor(BaseConstructor):
    def __init__(self, hparams):
        super(HeatmapConstructor, self).__init__(hparams, HeatmapDataset)


class AutoEncoderConstructor(BaseConstructor):
    def __init__(self):
        super(AutoEncoderConstructor, self).__init__(AutoEncoderDataset)
