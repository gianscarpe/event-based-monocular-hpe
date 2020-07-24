from albumentations import Compose, CenterCrop
from albumentations.pytorch import ToTensor
from omegaconf import DictConfig
import unittest
from experimenting.dataset.factory import AutoEncoderConstructor


class TestFactory(unittest.TestCase):
    def setUp(self):
        aug = Compose([CenterCrop(256, 256), ToTensor()])
        data_dir = '/home/gscarpellini/dev/event-camera/src/tests/data/frames'
        labels_dir = '/home/gscarpellini/dev/event-camera/src/tests/data/labels'
        self.hparams = DictConfig({
            'dataset': {
                'data_dir': data_dir,
                'save_split': False,
                'labels_dir': labels_dir,
                'test_subjects': [1, 2, 3, 4, 5],
                'split_at': 0.8,
                'cams': [3]
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
