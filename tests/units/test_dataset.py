import unittest
from unittest import mock

import numpy as np
import torch
from experimenting.dataset.dataset import (
    AutoEncoderDataset,
    ClassificationDataset,
    Joints3DDataset,
)

TEST_IMAGE_SHAPE = (224, 224)


class TestBaseDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if cls is TestBaseDataset:
            raise unittest.SkipTest("Skip TestCore tests")
        super(TestBaseDataset, cls).setUpClass()

    def setUp(self):
        self.mocked_dataset = mock.MagicMock(N_JOINTS=13)
        self.mocked_dataset.get_frame_from_id.return_value = "image"
        self.mocked_dataset.get_label_from_id.return_value = "label"
        self.mocked_joint = {
            'xyz_cam': np.random.random((3, 13)),
            'xyz': np.random.random((3, 13)),
            'joints': np.random.randint(0, 255, (13, 2)),
            'camera': np.random.random((3, 4)),
            'M': np.random.random((3, 4))
        }

        self.mocked_dataset.get_joint_from_id.return_value.__getitem__.side_effect = self.mocked_joint.__getitem__

        self.mocked_indexes = mock.MagicMock()
        self.mocked_indexes.__len__.return_value = 10
        self.mocked_indexes.__getitem__.side_effect = range(10, 19)
        self.mocked_transform = mock.MagicMock()
        self.mocked_transform.return_value = {'image': 'aug_image'}


class TestClassificationDataset(TestBaseDataset):
    def test_getitem_frame(self):
        dataset_config = {
            'dataset': self.mocked_dataset,
            'indexes': self.mocked_indexes
        }
        task_dataset = ClassificationDataset(**dataset_config)
        idx = 0

        x, y = task_dataset[idx]

        self.mocked_indexes.__getitem__.assert_called_once_with(idx)
        self.mocked_dataset.get_frame_from_id.assert_called_once()
        self.mocked_dataset.get_label_from_id.assert_called_once()
        self.assertEqual(x, "image")
        self.assertEqual(y, "label")

    def test_getitem_frame_augmented(self):
        dataset_config = {
            'dataset': self.mocked_dataset,
            'indexes': self.mocked_indexes,
            'transform': self.mocked_transform
        }
        task_dataset = ClassificationDataset(**dataset_config)
        idx = 0

        x, y = task_dataset[idx]

        self.mocked_indexes.__getitem__.assert_called_once_with(idx)
        self.mocked_dataset.get_frame_from_id.assert_called_once()
        self.mocked_dataset.get_label_from_id.assert_called_once()
        self.mocked_transform.assert_called_once_with(image="image")
        self.assertEqual(x, "aug_image")
        self.assertEqual(y, "label")


class TestAutoencoderDataset(TestBaseDataset):
    def test_getitem(self):
        dataset_config = {
            'dataset': self.mocked_dataset,
            'indexes': self.mocked_indexes
        }
        task_dataset = AutoEncoderDataset(**dataset_config)
        idx = 0

        x = task_dataset[idx]

        self.mocked_indexes.__getitem__.assert_called_once_with(idx)
        self.mocked_dataset.get_frame_from_id.assert_called_once()
        self.mocked_dataset.get_frame_from_id.assert_called_once()
        self.assertEqual(x, "image")

    def test_getitem_augmented(self):
        dataset_config = {
            'dataset': self.mocked_dataset,
            'indexes': self.mocked_indexes,
            'transform': self.mocked_transform
        }
        task_dataset = AutoEncoderDataset(**dataset_config)
        idx = 0

        x = task_dataset[idx]

        self.mocked_indexes.__getitem__.assert_called_once_with(idx)
        self.mocked_dataset.get_frame_from_id.assert_called_once()
        self.mocked_dataset.get_frame_from_id.assert_called_once()
        self.mocked_transform.assert_called_once_with(image="image")
        self.assertEqual(x, "aug_image")


class TestJoints3DDataset(TestBaseDataset):
    @mock.patch('experimenting.utils.pose3d_utils.skeleton_normaliser')
    def test_getitem(self, mocked_normalizer):
        dataset_config = {
            'dataset': self.mocked_dataset,
            'indexes': self.mocked_indexes,
            'in_shape': (224, 224)
        }
        task_dataset = Joints3DDataset(**dataset_config)
        idx = 0
        expected_camera = torch.DoubleTensor(self.mocked_joint['camera'])
        expected_M = torch.DoubleTensor(self.mocked_joint['M'])
        expected_xyz = torch.DoubleTensor(self.mocked_joint['xyz'].swapaxes(
            1, 0))
        expected_normalized_skeleton = torch.randn((13, 3))
        mocked_normalizer.SkeletonNormaliser.return_value.normalise_skeleton.return_value = expected_normalized_skeleton

        x, y = task_dataset[idx]

        self.mocked_indexes.__getitem__.assert_called_once_with(idx)
        self.mocked_dataset.get_frame_from_id.assert_called_once()
        self.mocked_dataset.get_joint_from_id.assert_called_once()
        self.assertEqual(x, "image")
        self.assertTrue(torch.equal(y['camera'], expected_camera))
        self.assertTrue(torch.equal(y['M'], expected_M))
        self.assertTrue(torch.equal(y['xyz'], expected_xyz))
        self.assertTrue(
            torch.equal(y['normalized_skeleton'],
                        expected_normalized_skeleton))
        mocked_normalizer.SkeletonNormaliser.return_value.normalise_skeleton.assert_called_once(
        )
