import unittest

from omegaconf import DictConfig

from experimenting.dataset.core import DHP19Core, NTUCore


class TestCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if cls is TestCore:
            raise unittest.SkipTest("Skip TestCore tests")
        super(TestCore, cls).setUpClass()

    def test_init(self):
        self.assertIsNotNone(self.core)

    def test_paths_params(self):

        self.assertIsNotNone(self.core.file_paths)
        self.assertGreater(len(self.core.file_paths), 0)

    def test_train_test_split(self):
        train, val, test = self.core.get_train_test_split()
        self.assertIsNotNone(train)

        self.assertGreater(len(test), 0)
        self.assertGreater(len(val) + len(train), 0)


class TestDHP19ParamsDefaultPartition(TestCore):
    def setUp(self):
        data_dir = 'tests/data/dhp19/frames'
        labels_dir = 'tests/data/dhp19/labels'
        self.hparams = DictConfig(
            {
                'data_dir': data_dir,
                'save_split': False,
                'labels_dir': labels_dir,
                'joints_dir': labels_dir,
                'hm_dir': labels_dir,
                'test_subjects': [1, 2, 3, 4, 5],
                'split_at': 0.8,
                'cams': [3],
                'name': "test",
                'base_path': None,
                'movements': None,
                'preload_dir': None,
                'n_classes': 10,
                'n_joints': 10,
                'partition': None,
                'n_channels': 3,
            }
        )
        self.core = DHP19Core(**self.hparams)


class TestDHP19ParamsCrossSubject(TestCore):
    def setUp(self):
        data_dir = 'tests/data/dhp19/frames'
        labels_dir = 'tests/data/dhp19/labels'
        self.hparams = DictConfig(
            {
                'data_dir': data_dir,
                'save_split': False,
                'labels_dir': labels_dir,
                'joints_dir': labels_dir,
                'hm_dir': labels_dir,
                'test_subjects': [1, 2, 3, 4, 5],
                'split_at': 0.8,
                'cams': [3],
                'partition': 'cross-subject',
                'name': "test",
                'base_path': None,
                'movements': None,
                'preload_dir': None,
                'n_classes': 10,
                'n_joints': 10,
                'n_channels': 3,
            }
        )
        self.core = DHP19Core(**self.hparams)


class TestDHP19ParamsCrossView(TestCore):
    def setUp(self):
        data_dir = 'tests/data/dhp19/frames'
        labels_dir = 'tests/data/dhp19/labels'
        self.hparams = DictConfig(
            {
                'data_dir': data_dir,
                'save_split': False,
                'labels_dir': labels_dir,
                'joints_dir': labels_dir,
                'hm_dir': labels_dir,
                'test_subjects': [1, 2, 3, 4, 5],
                'split_at': 0.8,
                'cams': [0, 1, 2, 3],
                'partition': 'cross-view',
                'name': "test",
                'base_path': None,
                'movements': None,
                'preload_dir': None,
                'n_classes': 10,
                'n_joints': 10,
                'n_channels': 3,
            }
        )
        self.core = DHP19Core(**self.hparams)


class TestNTUParams(TestCore):
    def setUp(self):
        data_dir = 'tests/data/ntu/frames'
        labels_dir = 'tests/data/ntu/labels'
        self.hparams = DictConfig(
            {
                'name': 'test',
                'data_dir': data_dir,
                'labels_dir': labels_dir,
                'test_subjects': [19],
                'partition': 'cross-subject',
            }
        )
        self.core = NTUCore(**self.hparams)


if __name__ == '__main__':
    unittest.main()
