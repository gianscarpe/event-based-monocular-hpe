import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .core import BaseCore
from .factory import (
    AutoEncoderConstructor,
    BaseConstructor,
    ClassificationConstructor,
    HeatmapConstructor,
    Joints3DConstructor,
    JointsConstructor,
)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_factory: BaseConstructor,
        core: BaseCore,
        aug_train_config,
        aug_test_config,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.core = core
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_factory = dataset_factory
        self.aug_train_config = aug_train_config
        self.aug_test_config = aug_test_config

    def setup(self, stage=None):
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = self.dataset_factory.get_datasets(
            self.core, self.aug_train_config, self.aug_test_config
        )

    def prepare_data(self, *args, **kwargs):
        return super().prepare_data(*args, **kwargs)

    def transfer_batch_to_device(self, batch, device):
        return super().transfer_batch_to_device(batch, device)

    def train_dataloader(self):
        return get_dataloader(self.train_dataset, self.batch_size, self.num_workers)

    def val_dataloader(self):
        return get_dataloader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return get_dataloader(
            self.test_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def get_data(hparams, dataset_constructor):
    batch_size = hparams.training['batch_size']
    num_workers = hparams.training.num_workers

    train_dataset, val_dataset, test_dataset = dataset_constructor.get_datasets()

    train_loader = get_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = get_dataloader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    test_loader = get_dataloader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def get_dataloader(dataset, batch_size, num_workers, shuffle=True):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader
