import os
from os.path import join

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score

import hydra
import pytorch_lightning as pl
from kornia import geometry

from ..dataset import DatasetType, get_dataloader, get_datasets
from ..utils import (
    average_loss,
    denormalize_predict,
    flatten,
    get_cnn,
    get_feature_extractor,
    get_joints_from_heatmap,
    predict_xyz,
    unflatten,
)
from .autoencoder import AutoEncoder
from .hourglass import HourglassModel
from .margipose import get_margipose_model
from .metrics import MPJPE

__all__ = [
    'Classifier', 'PoseEstimator', 'HourglassEstimator', 'MargiposeEstimator',
    'AutoEncoderEstimator'
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
        self.train_dataset, self.val_dataset, self.test_dataset = get_datasets(
            self._hparams, dataset_type=self.dataset_type)

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

        feature_extractor, mid_dimension = get_feature_extractor(
            extractor_params)

        return feature_extractor, mid_dimension

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


class Classifier(BaseModule):
    def __init__(self, hparams):
        """
        Initialize Classifier model
        """

        super(Classifier, self).__init__(hparams,
                                         DatasetType.CLASSIFICATION_DATASET)

    def set_params(self):
        params = {
            'n_channels': self._hparams.dataset['n_channels'],
            'n_classes': self._hparams.dataset['n_classes'],
            'pretrained': self._hparams.training['pretrained']
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


class MargiposeEstimator(BaseModule):
    def __init__(self, hparams):

        super(MargiposeEstimator, self).__init__(hparams,
                                                 DatasetType.JOINTS_3D_DATASET)

    def set_params(self):

        in_cnn, mid_dimension = self._get_feature_extractor(
            self._hparams.training['model'],
            self._hparams.dataset['n_channels'],
            join(self._hparams.model_zoo, self._hparams.training.backbone),
            self._hparams.training['pretrained'])

        params = {
            'in_cnn': in_cnn,
            'mid_dimension': mid_dimension,
            'n_joints': self._hparams.dataset['n_joints'],
            'n_stages': self._hparams.training['stages'],
            'predict_3d': True
        }
        self.n_channels = self._hparams.dataset.n_channels
        self.n_joints = self._hparams.dataset.n_joints
        self.predict_3d = self._hparams.training.predict_3d

        self.max_x = self._hparams.dataset['max_x']
        self.max_y = self._hparams.dataset['max_y']
        self.max_z = self._hparams.dataset['max_z']
        self.model = get_margipose_model(params)

        self.metrics = {"MPJPE": MPJPE(reduction=average_loss)}

    def forward(self, x):
        x = self.model(x)
        return x

    def predict3d(self, outs):

        # Take last output
        xy_hm = outs[0][-1]
        zy_hm = outs[1][-1]
        xz_hm = outs[2][-1]
        return predict_xyz((xy_hm, zy_hm, xz_hm))

    def get_pred_gt_skeletons(self, outs, b_y):
        width, height = outs[-1][-1].shape[-2:]
        normalized_skeletons = self.predict3d(outs)

        gt_skeletons = b_y['skeleton']

        # TODO: denormalization is currently CPU only
        cameras = b_y['camera'].cpu()
        z_refs = b_y['z_ref'].cpu()
        normalized_skeletons = normalized_skeletons.cpu()
        pred_skeletons = []

        for i in range(len(normalized_skeletons)):
            camera = cameras[i]
            z_ref = z_refs[i]

            pred_skeleton = normalized_skeletons[i].narrow(-1, 0, 3)
            pred_skeleton = denormalize_predict(pred_skeleton, width, height,
                                                camera,
                                                z_ref).transpose(0, -1)
            pred_skeletons.append(pred_skeleton)

        pred_skeletons = torch.stack(pred_skeletons).to(gt_skeletons.device)

        return gt_skeletons, pred_skeletons

    def _calculate_loss3d(self, outs, b_y):
        loss = 0

        xy_hms = outs[0]
        zy_hms = outs[1]
        xz_hms = outs[2]
        normalized_skeletons = b_y['normalized_skeleton']
        b_masks = b_y['mask']

        for outs in zip(xy_hms, zy_hms, xz_hms):
            loss += self.loss_func(outs, normalized_skeletons, b_masks)

        return loss / len(outs)

    def _eval(self, batch, denormalize=False):
        b_x, b_y = batch

        outs = self.forward(b_x)  # cnn output
        loss = self._calculate_loss3d(outs, b_y)
        b_masks = b_y['mask']
        if not denormalize:  # Evaluate normalized and projected preds
            pred_joints = self.predict3d(outs)
            gt_joints = b_y['normalized_skeleton']
        else:  # Evaluate with actual skeletons
            pred_joints, gt_joints = self.get_pred_gt_skeletons(outs, b_y)

        results = {
            metric_name: metric_function(pred_joints, gt_joints, b_masks)
            for metric_name, metric_function in self.metrics.items()
        }

        return loss, results

    def training_step(self, batch, batch_idx):
        b_x, b_y = batch

        outs = self.forward(b_x)  # cnn output

        loss = self._calculate_loss3d(outs, b_y)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        loss, results = self._eval(batch,
                                   denormalize=False)  # Normalized results
        return {"batch_val_loss": loss, **results}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['batch_val_loss'] for x in outputs]).mean()
        results = self._get_aggregated_results(outputs, 'val_mean')
        logs = {'val_loss': avg_loss, **results, 'step': self.current_epoch}

        return {'val_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_idx):
        loss, results = self._eval(batch, denormalize=True)
        return {"batch_test_loss": loss, **results}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['batch_test_loss'] for x in outputs]).mean()
        results = self._get_aggregated_results(outputs, 'test_mean')

        logs = {'test_loss': avg_loss, **results, 'step': self.current_epoch}

        return {**logs, 'log': logs, 'progress_bar': logs}


class AutoEncoderEstimator(BaseModule):
    def __init__(self, hparams):

        super(AutoEncoderEstimator,
              self).__init__(hparams, DatasetType.AUTOENCODER_DATASET)

    def set_params(self):
        in_cnn, mid_dimension = self._get_feature_extractor(
            self._hparams.training['model'],
            self._hparams.dataset['n_channels'], None,
            self._hparams.training['pretrained'])

        params = {
            'in_channels': self._hparams.dataset['n_channels'],
            'in_cnn': in_cnn,
            'mid_dimension': mid_dimension,
            'up_layers': self._hparams.training['up_layers'],
            'latent_size': self._hparams.training['latent_size'],
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
