from os.path import join

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score

import hydra
import pytorch_lightning as pl
from kornia import geometry

from ..dataset import DatasetType, get_data
from ..utils import average_loss, flatten, get_joints_from_heatmap, unflatten
from .cnns import get_cnn
from .hourglass import HourglassModel, MargiPoseModel
from .metrics import MPJPE

__all__ = [
    'Classifier', 'PoseEstimator', 'HourglassEstimator', 'MargiposeEstimator'
]


class BaseModule(pl.LightningModule):
    def __init__(self, hparams, dataset_type):
        """
        Initialize Classifier model
        """

        super(BaseModule, self).__init__()
        self.dataset_type = dataset_type

        self.hparams = flatten(hparams)
        self._hparams = unflatten(hparams)

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
        self.train_loader, self.val_loader, self.test_loader = get_data(
            unflatten(self._hparams), dataset_type=self.dataset_type)

    def forward(self, x):
        x = self.model(x)
        return x

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

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


class Classifier(BaseModule):
3    def __init__(self, hparams):
        """
        Initialize Classifier model
        """

        super(Classifier, self).__init__(hparams,
                                         DatasetType.CLASSIFICATION_DATASET)

    def set_params(self):
        params = {
            'n_channels': self._hparams.dataset['n_channels'],
            'n_classes': self._hparams.dataset['n_classes']
        }
        self.model = get_cnn(self._hparams.training.model, params)

    def training_step(self, batch, batch_idx):
        b_x, b_y = batch

        output = self.forward(b_x)  # cnn output
        loss = self.loss_func(output, b_y)  # cross entropy loss

        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        b_x, b_y = batch

        output = self.forward(b_x)  # cnn output
        loss = self.loss_func(output, b_y)  # cross entropy loss

        _, pred_y = torch.max(output.data, 1)
        return {"batch_val_loss": loss, "y_pred": pred_y, "y_true": b_y}

    def validation_epoch_end(self, outputs):
        y_true = torch.cat([x['y_true'] for x in outputs]).cpu()
        y_pred = torch.cat([x['y_pred'] for x in outputs]).cpu()

        avg_loss = torch.stack([x['batch_val_loss'] for x in outputs]).mean()
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')

        logs = {
            'val_loss': avg_loss,
            "val_acc": acc,
            'val_precision': precision,
            'val_recall': recall,
            'step': self.current_epoch
        }

        return {
            'val_loss': avg_loss,
            'val_acc': acc,
            'log': logs,
            'progress_bar': logs
        }

    def test_step(self, batch, batch_idx):
        b_x, b_y = batch
        output = self.forward(b_x)  # cnn output
        loss = self.loss_func(output, b_y)  # cross entropy loss

        _, pred_y = torch.max(output.data, 1)

        return {"batch_test_loss": loss, "y_pred": pred_y, "y_true": b_y}

    def test_epoch_end(self, outputs):
        y_true = torch.cat([x['y_true'] for x in outputs]).cpu()
        y_pred = torch.cat([x['y_pred'] for x in outputs]).cpu()

        avg_loss = torch.stack([x['batch_test_loss'] for x in outputs]).mean()
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')

        logs = {
            'test_loss': avg_loss,
            "test_acc": acc,
            'test_precision': precision,
            'test_recall': recall,
            'step': self.current_epoch
        }

        return {**logs, 'log': logs, 'progress_bar': logs}


class PoseEstimator(BaseModule):
    def __init__(self, hparams):
        super(PoseEstimator, self).__init__(hparams, DatasetType.HM_DATASET)

    def set_params(self):
        self.n_channels = self._hparams.dataset.n_channels
        self.n_joints = self._hparams.dataset.n_joints
        params = {
            'n_channels': self._hparams.dataset['n_channels'],
            'n_classes': self._hparams.dataset['n_joints'],
            'encoder_depth': self._hparams.training.encoder_depth
        }

        self.model = get_cnn(self._hparams.training.model, params)

        self.metrics = {"MPJPE": MPJPE(reduction=average_loss)}

    def forward(self, x):
        x = self.model(x)
        return x

    def predict(self, output):
        pred_joints, _ = get_joints_from_heatmap(output)
        return pred_joints

    def _eval(self, batch):
        b_x, b_y = batch

        output = self.forward(b_x)  # cnn output
        loss = self.loss_func(output, b_y)  # cross entropy loss

        gt_joints, _ = get_joints_from_heatmap(b_y)
        pred_joints = self.predict(output)

        results = {
            metric_name: metric_function(pred_joints, gt_joints)
            for metric_name, metric_function in self.metrics.items()
        }
        return loss, results

    def training_step(self, batch, batch_idx):
        b_x, b_y = batch

        output = self.forward(b_x)  # cnn output
        loss = self.loss_func(output, b_y)  # cross entropy loss
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


class HourglassEstimator(BaseModule):
    def __init__(self, hparams):

        super(HourglassEstimator, self).__init__(hparams,
                                                 DatasetType.JOINTS_DATASET)

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
        return loss / len(outs)

    def _eval(self, batch):
        b_x, b_y, b_masks = batch

        output = self.forward(b_x)  # cnn output

        loss = self._calculate_loss(output, b_y, b_masks)

        pred_joints = self.predict(output)
        gt_joints = geometry.denormalize_pixel_coordinates(
            b_y, self._hparams.dataset.max_h, self._hparams.dataset.max_w)

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


class MargiposeEstimator(BaseModule):
    def __init__(self, hparams):

        super(MargiposeEstimator, self).__init__(hparams,
                                                 DatasetType.JOINTS_3D_DATASET)

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

        self.max_x = self._hparams.dataset['max_x']
        self.max_y = self._hparams.dataset['max_y']
        self.max_z = self._hparams.dataset['max_z']
        self.model = MargiPoseModel(**params)

        self.metrics = {"MPJPE": MPJPE(reduction=average_loss)}

    def forward(self, x):
        x = self.model(x)
        return x

    def predict(self, output):

        xy_hm, zy_hm, xz_hm = output
        xy = geometry.spatial_expectation2d(xy_hm)
        zy = geometry.spatial_expectation2d(zy_hm)
        xz = geometry.spatial_expectation2d(xz_hm)
        x, y = xy.split(1, -1)
        z = 0.5 * (zy[:, :, 0:1] + xz[:, :, 1:2])

        return geometry.denormalize_pixel_coordinates3d(
            torch.cat([x, y, z], -1), self.max_z, self.max_y, self.max_x)

    def _calculate_loss(self, outs, b_y, b_masks):
        loss = 0

        xy_hm = outs[0]
        zy_hm = outs[1]
        xz_hm = outs[2]
        for out in zip(xy_hm, zy_hm, xz_hm):
            loss += self.loss_func(out, b_y, b_masks)
        return loss / len(outs)

    def _eval(self, batch):
        b_x, b_y, b_masks = batch

        outs = self.forward(b_x)  # cnn output
        xy_hm = outs[0]
        zy_hm = outs[1]
        xz_hm = outs[2]
        loss = self._calculate_loss(outs, b_y, b_masks)

        pred_joints = self.predict([xy_hm[-1], zy_hm[-1], xz_hm[-1]])
        gt_joints = geometry.denormalize_pixel_coordinates3d(
            b_y, self.max_z, self.max_y, self.max_x)

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
