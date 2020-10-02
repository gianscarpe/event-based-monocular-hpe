import unittest
from unittest import expectedFailure, mock

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
        self.mocked_dataset = mock.Mock()
        self.mocked_dataset.get_frame_from_id.return_value = "image"
        self.mocked_dataset.get_label_from_id.return_value = "label"
        self.mocked_dataset.get_joint_from_id.return_value = mock.MagicMock()

        self.mocked_indexes = mock.MagicMock()
        self.mocked_indexes.__len__.return_value = 10
        self.mocked_indexes.__getitem__.side_effect = range(10, 19)
        self.mocked_transform = mock.MagicMock()
        self.mocked_transform.return_value = {'image': 'aug_image'}

        
class TestClassificationDataset(TestBaseDataset):
    def test_getitem_frame(self):
        dataset_config = {'dataset': self.mocked_dataset, 'indexes': self.mocked_indexes}        
        task_dataset = ClassificationDataset(**dataset_config)
        idx = 0
        
        x, y = task_dataset[idx]

        self.mocked_indexes.__getitem__.assert_called_once_with(idx)
        self.mocked_dataset.get_frame_from_id.assert_called_once()
        self.mocked_dataset.get_label_from_id.assert_called_once()
        self.assertEqual(x, "image")
        self.assertEqual(y, "label")        

    def test_getitem_frame_augmented(self):
        dataset_config = {'dataset': self.mocked_dataset, 'indexes':
                          self.mocked_indexes, 'transform': self.mocked_transform}        
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
            dataset_config = {'dataset': self.mocked_dataset, 'indexes': self.mocked_indexes}        
            task_dataset = AutoEncoderDataset(**dataset_config)
            idx = 0

            x = task_dataset[idx]

            self.mocked_indexes.__getitem__.assert_called_once_with(idx)
            self.mocked_dataset.get_frame_from_id.assert_called_once()
            self.mocked_dataset.get_frame_from_id.assert_called_once()
            self.assertEqual(x, "image")

        def test_getitem_augmented(self):
            dataset_config = {'dataset': self.mocked_dataset, 'indexes':
                              self.mocked_indexes, 'transform': self.mocked_transform}        
            task_dataset = AutoEncoderDataset(**dataset_config)
            idx = 0

            x = task_dataset[idx]

            self.mocked_indexes.__getitem__.assert_called_once_with(idx)
            self.mocked_dataset.get_frame_from_id.assert_called_once()
            self.mocked_dataset.get_frame_from_id.assert_called_once()
            self.mocked_transform.assert_called_once_with(image="image")
            self.assertEqual(x, "aug_image")

class TestJoints3DDataset(TestBaseDataset):
    @expectedFailure
    def test_getitem(self):
        dataset_config = {'dataset': self.mocked_dataset, 'indexes':
                          self.mocked_indexes, 'in_shape':(224, 224)}        
        task_dataset = Joints3DDataset(**dataset_config)
        idx = 0

        x, y = task_dataset[idx]

        self.mocked_indexes.__getitem__.assert_called_once_with(idx)
        self.mocked_dataset.get_frame_from_id.assert_called_once()
        self.mocked_dataset.get_frame_from_id.assert_called_once()
        self.assertEqual(x, "image")
        self.assertEqual(y, "image")        
