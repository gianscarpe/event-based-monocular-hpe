import torch

from experimenting.agents.base import BaseModule
from experimenting.dataset import AutoEncoderConstructor
from experimenting.models.autoencoder import AutoEncoder


class AutoEncoderEstimator(BaseModule):
    def __init__(self, hparams):

        super(AutoEncoderEstimator, self).__init__(hparams,
                                                   AutoEncoderConstructor)

    def set_params(self):
        in_cnn = self._get_feature_extractor(
            self._hparams.training['model'],
            self._hparams.dataset['n_channels'], None,
            self._hparams.training['pretrained'])

        params = {
            'in_shape': (self.hparams.dataset['n_channels'],
                         *self.hparams.dataset['in_shape']),
            'in_cnn':
            in_cnn,
            'up_layers':
            self._hparams.training['up_layers'],
            'latent_size':
            self._hparams.training['latent_size'],
        }
        self.model = AutoEncoder(**params)

    def training_step(self, batch, batch_idx):
        b_x = batch

        output = self.forward(b_x)  # cnn output
        loss = self.loss_func(output, b_x)

        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):

        b_x = batch

        output = self.forward(b_x)  # cnn output
        loss = self.loss_func(output, b_x)  # cross entropy loss

        return {"batch_val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['batch_val_loss'] for x in outputs]).mean()

        logs = {'val_loss': avg_loss, 'step': self.current_epoch}

        return {'val_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_idx):
        b_x = batch
        output = self.forward(b_x)  # cnn output
        loss = self.loss_func(output, b_x)  # cross entropy loss

        return {"batch_test_loss": loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['batch_test_loss'] for x in outputs]).mean()

        logs = {'test_loss': avg_loss, 'step': self.current_epoch}

        return {**logs, 'log': logs, 'progress_bar': logs}
