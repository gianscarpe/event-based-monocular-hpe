import os

import hydra
import pytorch_lightning as pl
import torch

from experimenting.dataset import get_dataloader
from experimenting.utils import flatten, get_feature_extractor, unflatten

__all__ = [
]


class BaseModule(pl.LightningModule):
    def __init__(self, hparams, dataset_constructor):
        """
        Initialize Classifier model
        """

        super(BaseModule, self).__init__()

        self.hparams = flatten(hparams)
        self._hparams = unflatten(hparams)
        self.dataset_constructor = dataset_constructor

        self.optimizer_config = self._hparams.optimizer
        self.scheduler_config = None

        self.loss_func = hydra.utils.instantiate(self._hparams.loss)

        self.set_params()
        self.set_optional_params()

    def set_params(self):
        pass

    def set_optional_params(self):
        if self._hparams.optimizer.use_lr_scheduler:
            self.scheduler_config = self._hparams.lr_scheduler

    def prepare_data(self):
        datasets = self.dataset_constructor(self._hparams).get_datasets()
        self.train_dataset, self.val_dataset, self.test_dataset = datasets

    def _get_feature_extractor(self, model, n_channels, backbone_path,
                               pretrained):
        extractor_params = {'n_channels': n_channels, 'model': model}

        if backbone_path is not None and os.path.exists(backbone_path):
            extractor_params['custom_model_path'] = backbone_path
        else:
            if pretrained is not None:
                extractor_params['pretrained'] = pretrained
            else:
                extractor_params['pretrained'] = True

        feature_extractor = get_feature_extractor(extractor_params)

        return feature_extractor

    def forward(self, x):
        x = self.model(x)
        return x

    def train_dataloader(self):
        return get_dataloader(self.train_dataset,
                              self._hparams.training['batch_size'],
                              shuffle=True,
                              num_workers=self._hparams.training.num_workers)

    def val_dataloader(self):
        return get_dataloader(self.val_dataset,
                              self._hparams.training['batch_size'],
                              shuffle=False,
                              num_workers=self._hparams.training.num_workers)

    def test_dataloader(self):
        return get_dataloader(self.test_dataset,
                              self._hparams.training['batch_size'],
                              shuffle=False,
                              num_workers=self._hparams.training.num_workers)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_config['type'])(
            params=self.parameters(), **self.optimizer_config['params'])

        scheduler = None
        if self.scheduler_config:
            scheduler = getattr(torch.optim.lr_scheduler,
                                self.scheduler_config['type'])(
                                    optimizer,
                                    **self.scheduler_config['params'])
            return [optimizer], [scheduler]

        return optimizer

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'avg_train_loss': avg_loss, 'step': self.current_epoch}

        return {'train_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def _get_aggregated_results(self, outputs, prefix):
        results = {}
        for metric_key in self.metrics.keys():
            results[prefix + metric_key] = torch.stack(
                [x[metric_key] for x in outputs]).mean()

        return results
