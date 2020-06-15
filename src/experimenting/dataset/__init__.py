from enum import Enum

from torch.utils.data import DataLoader

from ..utils import get_augmentation
from .dataset import (
    DHP3DJointsDataset,
    DHP19ClassificationDataset,
    DHPHeatmapDataset,
    DHPJointsDataset,
)
from .params_utils import get_dataset_params


class DatasetType(Enum):
    CLASSIFICATION_DATASET = 0
    HM_DATASET = 1
    JOINTS_DATASET = 2
    JOINTS_3D_DATASET = 3


def get_data(hparams, dataset_type):
    batch_size = hparams.training['batch_size']
    num_workers = hparams.training.num_workers

    train_dataset, val_dataset, test_dataset = _get_datasets(
        hparams, dataset_type)

    train_loader = get_dataloader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

    val_loader = get_dataloader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)

    test_loader = get_dataloader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_dataloader(dataset, batch_size, num_workers, shuffle=True):
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=True)
    return loader


def _get_datasets(hparams, dataset_type):
    switch = {
        DatasetType.CLASSIFICATION_DATASET: _get_classification_datasets,
        DatasetType.HM_DATASET: _get_hm_datasets,
        DatasetType.JOINTS_DATASET: _get_joints_datasets,
        DatasetType.JOINTS_3D_DATASET: _get_3d_joints_datasets
    }

    return switch[dataset_type](hparams)


def _get_classification_datasets(hparams):
    preprocess_train = get_augmentation(hparams.augmentation_train)
    preprocess_val = get_augmentation(hparams.augmentation_test)

    params = get_dataset_params(hparams.dataset)
    file_paths = params['file_paths']
    train_index = params['train_indexes']
    val_index = params['val_indexes']
    test_index = params['test_indexes']

    train_params = {
        'file_paths': file_paths,
        'indexes': train_index,
        'transform': preprocess_train
    }
    val_params = {
        'file_paths': file_paths,
        'indexes': val_index,
        'transform': preprocess_val
    }

    test_params = {
        'file_paths': file_paths,
        'indexes': test_index,
        'transform': preprocess_val
    }

    return DHP19ClassificationDataset(
        **train_params), DHP19ClassificationDataset(
            **val_params), DHP19ClassificationDataset(**test_params)


def _get_joints_datasets(hparams):
    preprocess_train = get_augmentation(hparams.augmentation_train)
    preprocess_val = get_augmentation(hparams.augmentation_test)
    n_joints = hparams.dataset.n_joints
    params = get_dataset_params(hparams.dataset)

    file_paths = params['file_paths']
    train_index = params['train_indexes']
    val_index = params['val_indexes']
    test_index = params['test_indexes']

    train_params = {
        'file_paths': file_paths,
        'indexes': train_index,
        'transform': preprocess_train,
        'labels_dir': hparams.dataset.joints_dir,
        'n_joints': n_joints
    }

    val_params = {
        'file_paths': file_paths,
        'indexes': val_index,
        'transform': preprocess_val,
        'labels_dir': hparams.dataset.joints_dir
    }

    test_params = {
        'file_paths': file_paths,
        'indexes': test_index,
        'transform': preprocess_val,
        'labels_dir': hparams.dataset.joints_dir
    }

    return DHPJointsDataset(**train_params), DHPJointsDataset(
        **val_params), DHPJointsDataset(**test_params)


def _get_3d_joints_datasets(hparams):
    preprocess_train = get_augmentation(hparams.augmentation_train)
    preprocess_val = get_augmentation(hparams.augmentation_test)
    n_joints = hparams.dataset.n_joints
    params = get_dataset_params(hparams.dataset)

    file_paths = params['file_paths']
    train_index = params['train_indexes']
    val_index = params['val_indexes']
    test_index = params['test_indexes']
    train_params = {
        'file_paths': file_paths,
        'indexes': train_index,
        'transform': preprocess_train,
        'labels_dir': hparams.dataset.joints_dir,
        'n_joints': n_joints
    }

    val_params = {
        'file_paths': file_paths,
        'indexes': val_index,
        'transform': preprocess_val,
        'labels_dir': hparams.dataset.joints_dir
    }

    test_params = {
        'file_paths': file_paths,
        'indexes': test_index,
        'transform': preprocess_val,
        'labels_dir': hparams.dataset.joints_dir
    }

    return DHP3DJointsDataset(**train_params), DHP3DJointsDataset(
        **val_params), DHP3DJointsDataset(**test_params)


def _get_hm_datasets(hparams):
    preprocess_train = get_augmentation(hparams.augmentation_train)
    preprocess_val = get_augmentation(hparams.augmentation_test)
    n_joints = hparams.dataset.n_joints
    params = get_dataset_params(hparams.dataset)

    file_paths = params['file_paths']
    train_index = params['train_indexes']
    val_index = params['val_indexes']
    test_index = params['test_indexes']

    train_params = {
        'file_paths': file_paths,
        'indexes': train_index,
        'transform': preprocess_train,
        'labels_dir': hparams.dataset.hm_dir,
        'n_joints': n_joints
    }

    val_params = {
        'file_paths': file_paths,
        'indexes': val_index,
        'transform': preprocess_val,
        'labels_dir': hparams.dataset.hm_dir
    }

    test_params = {
        'file_paths': file_paths,
        'indexes': test_index,
        'transform': preprocess_val,
        'labels_dir': hparams.dataset.hm_dir
    }

    return DHPHeatmapDataset(**train_params), DHPHeatmapDataset(
        **val_params), DHPHeatmapDataset(**test_params)
