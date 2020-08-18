"""
Factory module provide a set of Constructor for datasets using factory design
pattern.  It encapsulates core dataset implementation and implement
functionalities to get train, val, and test dataset
"""

from abc import ABC

from ..utils import get_augmentation
from . import core
from .dataset import (
    AutoEncoderDataset,
    ClassificationDataset,
    HeatmapDataset,
    Joints3DDataset,
    JointsDataset,
)

__all__ = [
    'ClassificationConstructor', 'AutoEncoderConstructor',
    'Joints3DConstructor', 'JointsConstructor', 'HeatmapConstructor'
]


class BaseConstructor(ABC):
    def __init__(self, hparams, dataset_task):

        self.core = BaseConstructor._get_core(hparams)
        self.dataset_task = dataset_task
        self.train_params = {}
        self.val_params = {}
        self.test_params = {}

        preprocess_train, self.train_aug_info = get_augmentation(
            hparams.augmentation_train)
        preprocess_val, self.val_aug_info = get_augmentation(
            hparams.augmentation_test)

        train_indexes, val_indexes, test_indexes = self.core.get_train_test_split(
        )
        
        self._set_for_all('dataset', self.core)
        self._set_for_train('indexes', train_indexes)
        self._set_for_val('indexes', val_indexes)
        self._set_for_test('indexes', test_indexes)

        self._set_for_train('transform', preprocess_train)
        self._set_for_val('transform', preprocess_val)
        self._set_for_test('transform', preprocess_val)

    def _get_core(hparams):
        if hparams.dataset.core_class is None:
            dataset = "DHP19Core"
        else:
            dataset = hparams.dataset.core_class
        return getattr(core, dataset)(hparams.dataset)

    def _set_for_all(self, key, value):
        self._set_for_train(key, value)
        self._set_for_val(key, value)
        self._set_for_test(key, value)

    def _set_for_train(self, key, value):
        self.train_params[key] = value

    def _set_for_val(self, key, value):
        self.val_params[key] = value

    def _set_for_test(self, key, value):
        self.test_params[key] = value

    def get_datasets(self):
        return self.dataset_task(**self.train_params), self.dataset_task(
            **self.val_params), self.dataset_task(**self.test_params)


class ClassificationConstructor(BaseConstructor):
    def __init__(self, hparams):
        super(ClassificationConstructor,
              self).__init__(hparams, ClassificationDataset)


class JointsConstructor(BaseConstructor):
    def __init__(self, hparams):
        super(JointsConstructor, self).__init__(hparams, JointsDataset)


class Joints3DConstructor(BaseConstructor):
    def __init__(self, hparams):
        super(Joints3DConstructor, self).__init__(hparams, Joints3DDataset)
        self._set_for_all('in_shape', self.train_aug_info.in_shape)


class HeatmapConstructor(BaseConstructor):
    def __init__(self, hparams):
        super(HeatmapConstructor, self).__init__(hparams, HeatmapDataset)


class AutoEncoderConstructor(BaseConstructor):
    def __init__(self, hparams):
        super(AutoEncoderConstructor, self).__init__(hparams,
                                                     AutoEncoderDataset)
