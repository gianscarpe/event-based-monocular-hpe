from os.path import join

import torch
from kornia import geometry

from experimenting.agents.base import BaseModule
from experimenting.dataset import JointsConstructor
from experimenting.models.hourglass import HourglassModel
from experimenting.models.metrics import MPJPE
from experimenting.utils import average_loss


class HourglassEstimator(BaseModule):
    def __init__(self, hparams):

        super(HourglassEstimator, self).__init__(hparams, JointsConstructor)

    def set_params(self):
        self.n_channels = self._hparams.dataset.n_channels
        self.n_joints = self._hparams.dataset.n_joints

        params = {
            'n_channels':
            self._hparams.dataset['n_channels'],
            'n_joints':
            self._hparams.dataset['n_joints'],
            'backbone_path':
            join(self._hparams.model_zoo, self._hparams.training.backbone),
            'n_stages':
            self._hparams.training['stages']
        }

        self.model = HourglassModel(**params)

        self.metrics = {"MPJPE": MPJPE(reduction=average_loss)}

    def forward(self, x):
        x = self.model(x)
        return x

    def predict(self, output):
        pred_joints = geometry.denormalize_pixel_coordinates(
            geometry.spatial_expectation2d(output[-1]),
            self._hparams.dataset.max_h, self._hparams.dataset.max_w)
        return pred_joints

    def _calculate_loss(self, outs, b_y, b_masks):
        loss = 0
        for x in outs:
            loss += self.loss_func(x, b_y, b_masks)
        return loss

    def _eval(self, batch):
        b_x, b_y, b_masks = batch

        output = self.forward(b_x)  # cnn output

        loss = self._calculate_loss(output, b_y, b_masks)
        gt_joints = geometry.denormalize_pixel_coordinates(
            b_y, self._hparams.dataset.max_h, self._hparams.dataset.max_w)
        pred_joints = self.predict(output)

        results = {
            metric_name: metric_function(pred_joints, gt_joints, b_masks)
            for metric_name, metric_function in self.metrics.items()
        }
        return loss, results

    def training_step(self, batch, batch_idx):
        b_x, b_y, b_masks = batch

        outs = self.forward(b_x)  # cnn output

        loss = self._calculate_loss(outs, b_y, b_masks)

        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        loss, results = self._eval(batch)
        return {"batch_val_loss": loss, **results}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['batch_val_loss'] for x in outputs]).mean()
        results = self._get_aggregated_results(outputs, 'val_mean')
        logs = {'val_loss': avg_loss, **results, 'step': self.current_epoch}

        return {'val_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_idx):
        loss, results = self._eval(batch)
        return {"batch_test_loss": loss, **results}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['batch_test_loss'] for x in outputs]).mean()
        results = self._get_aggregated_results(outputs, 'test_mean')

        logs = {'test_loss': avg_loss, **results, 'step': self.current_epoch}

        return {**logs, 'log': logs, 'progress_bar': logs}
