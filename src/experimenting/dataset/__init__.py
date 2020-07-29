from torch.utils.data import DataLoader

from .factory import *
from .params_utils import DHP19Params


def get_data(hparams, dataset_constructor):
    batch_size = hparams.training['batch_size']
    num_workers = hparams.training.num_workers

    train_dataset, val_dataset, test_dataset = dataset_constructor.get_datasets()

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
