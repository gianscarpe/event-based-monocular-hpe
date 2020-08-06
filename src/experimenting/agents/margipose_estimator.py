from os.path import join

import torch

from experimenting.agents.base import BaseModule
from experimenting.dataset import Joints3DConstructor
from experimenting.models.margipose import get_margipose_model
from experimenting.models.metrics import MPJPE
from experimenting.utils import (
    average_loss,
    denormalize_predict,
    get_backbone_last_dimension,
    predict_xyz,
)


class MargiposeEstimator(BaseModule):
    def __init__(self, hparams):

        super(MargiposeEstimator, self).__init__(hparams, Joints3DConstructor)

    def set_params(self):

        in_cnn = self._get_feature_extractor(
            self._hparams.training['model'],
            self._hparams.dataset['n_channels'],
            join(self._hparams.model_zoo, self._hparams.training.backbone),
            self._hparams.training['pretrained'])
        if self._hparams.training['latent_size'] is None:
            self._hparams.training['latent_size'] = 128
        mid_dimension = get_backbone_last_dimension(
            in_cnn, self._hparams.training['model'])

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
