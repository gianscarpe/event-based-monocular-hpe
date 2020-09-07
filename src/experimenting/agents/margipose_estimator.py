from os.path import join

import torch

from ..agents.base import BaseModule
from ..dataset import Joints3DConstructor
from ..models.margipose import get_margipose_model
from ..models.metrics import AUC, MPJPE, PCK
from ..utils import average_loss, denormalize_predict, predict_xyz


class MargiposeEstimator(BaseModule):
    """
    Agents for training and testing multi-stage 3d joints estimator using
    marginal heatmaps (denoted as Margipose)
    """
    def __init__(self, hparams):

        super(MargiposeEstimator, self).__init__(hparams, Joints3DConstructor)

        in_cnn = MargiposeEstimator._get_feature_extractor(
            self._hparams.training['model'],
            self._hparams.dataset['n_channels'],
            join(self._hparams.model_zoo, self._hparams.training.backbone),
            self._hparams.training['pretrained'])

        params = {
            'in_shape': (self._hparams.dataset['n_channels'],
                         *self._hparams.dataset['in_shape']),
            'in_cnn':
            in_cnn,
            'n_joints':
            self._hparams.dataset['n_joints'],
            'n_stages':
            self._hparams.training['stages'],
            'predict_3d':
            True
        }
        self.n_channels = self._hparams.dataset.n_channels
        self.n_joints = self._hparams.dataset.n_joints
        self.predict_3d: bool = self._hparams.training.predict_3d

        #  Dataset parameters are used for 3d prediction
        self.max_x = self._hparams.dataset['max_x']
        self.max_y = self._hparams.dataset['max_y']
        self.max_z = self._hparams.dataset['max_z']

        self.model = get_margipose_model(params)

        self.metrics = {
            "MPJPE": MPJPE(reduction=average_loss),
            "AUC": AUC(reduction=average_loss, auc_reduction=None),
            "PCK": PCK(reduction=average_loss)
        }

    def forward(self, x):
        x = self.model(x)
        return x

    @staticmethod
    def predict3d(outs: (list, list, list)):
        """
        Predict normalized 3d skeleton joints

        Args:
            outs (list, list, list): output of the model

        Returns:
            torch tensor of normalized skeleton joints with shape (BATCH_SIZE, NUM_JOINTS, 3)
        Note:
            prediction used `dsnnt` toolbox
        """

        # Take last output (indexed -1)
        xy_hm = outs[0][-1]
        zy_hm = outs[1][-1]
        xz_hm = outs[2][-1]
        return predict_xyz((xy_hm, zy_hm, xz_hm))

    @staticmethod
    def get_denormalized_pred_gt_skeletons(outs, b_y):
        """
        It composes gt and pred denormalized skeletons

        Args:
            outs (list, list, list): output of the model
            b_y: batch y object (as returned by 3d joints dataset)

        Returns:
            Returns a tuple of torch tensor of shape (BATCH_SIZE, NUM_JOINTS, 3)

        Note:
            Prediction skeletons are normalized according to batch depth value
            `z_ref` and are kept in camera coordinate space.
            GT skeletons are provided in camera coordinate as well for comparison.

        Todo:
            [] de-normalization is currently CPU only
        """

        width, height = outs[-1][-1].shape[-2:]
        normalized_skeletons = MargiposeEstimator.predict3d(outs)

        gt_skeletons = b_y['skeleton']

        cameras = b_y['camera'].cpu()  # CPU only
        z_refs = b_y['z_ref'].cpu()  # CPU only
        normalized_skeletons = normalized_skeletons.cpu()  # CPU only
        pred_skeletons = []

        for i in range(len(normalized_skeletons)):
            camera = cameras[i]  # Internal camera parameter for input i
            z_ref = z_refs[i]  # Depth plane value for input i

            pred_skeleton = normalized_skeletons[i].narrow(-1, 0, 3)

            # Apply de-normalization using intrinsics, depth plane, and image plane pixel dimension
            pred_skeleton = denormalize_predict(pred_skeleton, width, height,
                                                camera, z_ref)
            pred_skeletons.append(pred_skeleton.transpose(
                0, -1))  # transpose to have shape (NUM_JOINTS, 3)

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
        """
        Note:
            De-normalization is time-consuming, therefore it can be specified to
            either compare normalized or de-normalized skeletons
        """
        b_x, b_y = batch
        outs = self.forward(b_x)  # cnn output
        loss = self._calculate_loss3d(outs, b_y)
        b_masks = b_y['mask']
        if not denormalize:  # Evaluate normalized and projected preds
            pred_joints = self.predict3d(outs)
            gt_joints = b_y['normalized_skeleton']
        else:  # Evaluate with actual skeletons
            pred_joints, gt_joints = MargiposeEstimator.get_denormalized_pred_gt_skeletons(
                outs, b_y)

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

        self.results = results
        logs = {'val_loss': avg_loss, 'step': self.current_epoch}

        return {'val_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_idx):
        loss, results = self._eval(
            batch, denormalize=True
        )  # Compare denormalized skeletons for test evaluation
        return {"batch_test_loss": loss, **results}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['batch_test_loss'] for x in outputs]).mean()
        results = self._get_aggregated_results(outputs, 'test_mean')
        self.results = results
        logs = {'test_loss': avg_loss, 'step': self.current_epoch}

        return {**logs, 'log': logs, 'progress_bar': logs}
