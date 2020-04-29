from .dataset import DHP19Dataset
from .indexes import get_dataset_params
from torch.utils.data import DataLoader
import os


def get_dataset(file_paths, index, preprocess, labels, augment_label):
    dataset = DHP19Dataset(file_paths=file_paths, indexes=index, labels=labels,
                           transform=preprocess, augment_label=augment_label)
    return dataset

def get_dataloader_from_dataset(dataset, batch_size, num_workers, shuffle=True):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers)
    return loader


def get_dataloader(file_paths, index, preprocess, augment_label, batch_size,
                   num_workers, labels=None, shuffle=True):
    loader = get_dataloader_from_dataset(dataset=get_dataset(file_paths, index,
                                                             preprocess=preprocess,
                                                             labels=labels,
                                                             augment_label=augment_label),
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=num_workers)

    return loader


def get_data(data_dir, preprocess_train, preprocess_val, batch_size,
             augment_label, num_workers=12):
        
    file_paths, train_index, val_index, test_index, labels = get_dataset_params(data_dir)
        
    train_loader = get_dataloader(file_paths, train_index, preprocess_train,
                                  augment_label, batch_size, num_workers, labels=labels,
                                  shuffle=True)

    val_loader = get_dataloader(file_paths, val_index, preprocess_val,
                                augment_label, batch_size, num_workers,
                                labels=labels,
                                shuffle=False)

    test_loader = get_dataloader(file_paths, test_index, preprocess_val,
                                 augment_label, batch_size, num_workers,
                                 labels=labels, shuffle=False)
    
    return train_loader, val_loader, test_loader
