import unittest

from experimenting.dataset.params_utils import DHP19Params, NTUParams
from omegaconf import DictConfig


class TestDHP19Params(unittest.TestCase):
    def setUp(self):
        data_dir = 'tests/data/dhp19/frames'
        labels_dir = 'tests/data/dhp19/labels'
        self.hparams = DictConfig({
            'data_dir': data_dir,
            'save_split': False,
            'labels_dir': labels_dir,
            'joints_dir': labels_dir,
            'hm_dir': labels_dir,
            'test_subjects': [1, 2, 3, 4, 5],
            'split_at': 0.8,
            'cams': [3]
        })
        self.params = DHP19Params(self.hparams)

    def test_init(self):
        self.assertIsNotNone(self.params)

    def test_paths_params(self):

        self.assertIsNotNone(self.params.file_paths)
        self.assertGreater(len(self.params.file_paths), 0)

    def test_train_test_split(self):

        self.assertIsNotNone(self.params.train_indexes)
        self.assertGreater(len(self.params.test_indexes), 0)
        self.assertGreater(
            len(self.params.val_indexes) + len(self.params.train_indexes), 0)


class TestNTUParams(unittest.TestCase):
    def setUp(self):
        data_dir = 'tests/data/ntu/frames'
        labels_dir = 'tests/data/ntu/labels'
        self.hparams = DictConfig({
            'data_dir': data_dir,
            'save_split': False,
            'labels_dir': labels_dir,
            'test_subjects': [19],
            'split_at': 0.8,
        })
        self.params = NTUParams(self.hparams)

    def test_init(self):
        self.assertIsNotNone(self.params)

    def test_paths_params(self):

        self.assertIsNotNone(self.params.file_paths)
        self.assertEqual(len(self.params.file_paths), 8)

    def test_train_test_split(self):

        self.assertIsNotNone(self.params.train_indexes)
        self.assertGreater(len(self.params.test_indexes), 0)
        self.assertGreater(
            len(self.params.val_indexes) + len(self.params.train_indexes), 0)


if __name__ == '__main__':
    unittest.main()
