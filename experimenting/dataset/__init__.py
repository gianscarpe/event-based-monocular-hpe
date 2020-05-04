from .dataset import *
from .indexes import get_dataset_params
from torch.utils.data import DataLoader
import os
from enum import Enum
from ..utils import get_augmentation

class DatasetType(Enum):
        CLASSIFICATION_DATASET = 0
        RECONSTUCTION_DATASET = 1

def get_data(hparams, dataset_type):
    batch_size = hparams.training['batch_size']
    num_workers = hparams.training.num_workers

    train_dataset, val_dataset, test_dataset = _get_datasets(hparams,
                                                             dataset_type)

    train_loader = get_dataloader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)

    val_loader = get_dataloader(val_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers)

    test_loader = get_dataloader(test_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def get_dataloader(dataset, batch_size, num_workers, shuffle=True):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers)
    return loader

def _get_datasets(hparams, dataset_type):
    switch = { DatasetType.CLASSIFICATION_DATASET : _get_classification_datasets,
               DatasetType.RECONSTUCTION_DATASET : _get_reconstruction_datasets}

    return switch[dataset_type](hparams)

def _get_classification_datasets(hparams):
    preprocess_train = get_augmentation(hparams.augmentation_train)
    preprocess_val = get_augmentation(hparams.augmentation_test)
    
    params = get_dataset_params(hparams.dataset.path)
    file_paths, train_index, val_index, test_index, labels = params

    train_params = {'file_paths': file_paths, 'indexes':train_index,
                    'transform':preprocess_train, 'labels':labels}

    val_params = {'file_paths': file_paths, 'indexes':val_index,
                    'transform':preprocess_val, 'labels':labels}

    test_params = {'file_paths': file_paths, 'indexes':test_index,
                    'transform':preprocess_val, 'labels':labels}

    return DHP19ClassificationDataset(**train_params), DHP19ClassificationDataset(**val_params), DHP19ClassificationDataset(**test_params)

def _get_reconstruction_datasets(hparams):
    preprocess_train = get_augmentation(hparams.augmentation_train)
    preprocess_val = get_augmentation(hparams.augmentation_test)
    
    params = get_dataset_params(hparams.dataset.path)
    file_paths, train_index, val_index, test_index, _ = params

    train_params = {'file_paths': file_paths, 'indexes':train_index,
                    'transform':preprocess_train, 'labels_dir':hparams.dataset.labels_dir}

    val_params = {'file_paths': file_paths, 'indexes':val_index,
                  'transform':preprocess_val, 'labels_dir':hparams.dataset.labels_dir}

    test_params = {'file_paths': file_paths, 'indexes':test_index,
                    'transform':preprocess_val,
                    'labels_dir':hparams.dataset.labels_dir}
    
    return DHP3DDataset(**train_params), DHP3DDataset(**val_params), DHP3DDataset(**test_params)




