from albumentations import Compose, CenterCrop
from albumentations.pytorch import ToTensor
from omegaconf import DictConfig
import unittest
from experimenting.dataset.params_utils import DHP19Params


class TestDHP19Params(unittest.TestCase):
    def setUp(self):
        data_dir = '/home/gscarpellini/dev/event-camera/src/tests/data/frames'
        labels_dir = '/home/gscarpellini/dev/event-camera/src/tests/data/labels'
        self.hparams = DictConfig({
            'data_dir': data_dir,
            'save_split': False,
            'labels_dir': labels_dir,
            'test_subjects': [1, 2, 3, 4, 5],
            'split_at': 0.8,
            'cams': [3]
        })
        self.params = DHP19Params(self.hparams)

    def test_init(self):
        self.assertIsNotNone(self.params)

    def test_paths_params(self):

        self.assertIsNotNone(self.params.file_paths)
        self.assertEqual(len(self.params.file_paths), 2)

    def test_train_test_split(self):

        self.assertIsNotNone(self.params.train_indexes)
        self.assertGreater(len(self.params.test_indexes), 0)
        self.assertGreater(
            len(self.params.val_indexes) + len(self.params.train_indexes), 0)



if __name__ == '__main__':
    unittest.main()
