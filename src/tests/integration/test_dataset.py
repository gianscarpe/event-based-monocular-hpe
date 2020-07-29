import unittest

from albumentations import CenterCrop, Compose
from albumentations.pytorch import ToTensor
from omegaconf import DictConfig

from ..dataset import (
    AutoEncoderConstructor,
    ClassificationConstructor,
    HeatmapConstructor,
    Joints3DConstructor,
    JointsConstructor,
)


class TestDHP19(unittest.TestCase):
    def setUp(self):
        aug = Compose([CenterCrop(256, 256), ToTensor()])
        data_dir = 'tests/data/dhp19/frames'
        labels_dir = 'tests/data/dhp19/labels'
        self.hparams = DictConfig({
            'dataset': {
                'data_dir': data_dir,
                'save_split': False,
                'labels_dir': labels_dir,
                'test_subjects': [1, 2, 3, 4, 5],
                'split_at': 0.8,
                'cams': [3],
                'params_class': 'DHP19Params'
            },
            'augmentation_train': {
                'info': {},
                'apply': {}
            },
            'augmentation_test': {
                'info': {},
                'apply': {}
            }
        })

    def test_classification(self):
        dc = ClassificationConstructor(self.hparams)
        self.assertIsNotNone(dc)


if __name__ == '__main__':
    unittest.main()
