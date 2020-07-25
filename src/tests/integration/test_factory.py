from albumentations import Compose, CenterCrop
from albumentations.pytorch import ToTensor
from omegaconf import DictConfig
import unittest
from experimenting.dataset.factory import AutoEncoderConstructor, ClassificationConstructor


class TestFactoryDHP19(unittest.TestCase):
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

    def test_ae(self):
        data_constructor = AutoEncoderConstructor(self.hparams)
        self.assertIsNotNone(data_constructor)

    def test_classification(self):
        data_constructor = ClassificationConstructor(self.hparams)
        self.assertIsNotNone(data_constructor)


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
                'params_class': 'NTUParams'
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
