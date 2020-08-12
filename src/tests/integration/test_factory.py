import unittest

from experimenting.dataset.factory import (
    AutoEncoderConstructor,
    ClassificationConstructor,
    HeatmapConstructor,
    Joints3DConstructor,
)
from omegaconf import DictConfig


class TestFactoryDHP19(unittest.TestCase):
    def setUp(self):
        data_dir = 'tests/data/dhp19/frames'
        labels_dir = 'tests/data/dhp19/labels'
        joints_dir = 'tests/data/dhp19/joints'
        self.hparams = DictConfig({
            'dataset': {
                'data_dir': data_dir,
                'joints_dir': joints_dir,
                'save_split': False,
                'labels_dir': labels_dir,
                'hm_dir': labels_dir,
                'test_subjects': [1, 2, 3, 4, 5],
                'in_shape': [256, 256],
                'split_at': 0.8,
                'cams': [3],
                'params_class': 'DHP19Core'
            },
            'augmentation_train': {
                'info': {
                    'in_shape': [256, 256],
                },
                'apply': {}
            },
            'augmentation_test': {
                'info': {
                    'in_shape': [256, 256]
                },
               'apply': {}
            }        
        })

    def test_ae(self):
        data_constructor = AutoEncoderConstructor(self.hparams)
        self.assertIsNotNone(data_constructor)
        train, val, test = data_constructor.get_datasets()

    def test_classification(self):
        data_constructor = ClassificationConstructor(self.hparams)
        self.assertIsNotNone(data_constructor)
        train, val, test = data_constructor.get_datasets()
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)

    def test_3d_joints(self):
        data_constructor = Joints3DConstructor(self.hparams)
        self.assertIsNotNone(data_constructor)
        train, val, test = data_constructor.get_datasets()
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)

    def test_hm(self):
        data_constructor = HeatmapConstructor(self.hparams)
        self.assertIsNotNone(data_constructor)
        train, val, test = data_constructor.get_datasets()
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)


class TestFactoryAutoencoderNTU(unittest.TestCase):
    def setUp(self):
        data_dir = 'tests/data/ntu/frames'
        labels_dir = 'tests/data/ntu/labels'

        self.hparams = DictConfig({
            'dataset': {
                'data_dir': data_dir,
                'save_split': False,
                'labels_dir': labels_dir,
                'test_subjects': [18],
                'split_at': 0.8,
                'params_class': 'NTUCore',
                'in_shape': [256, 256]
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
        self.data_constructor = AutoEncoderConstructor(self.hparams)

    def test_init(self):
        self.assertIsNotNone(self.data_constructor)


if __name__ == '__main__':
    unittest.main()
