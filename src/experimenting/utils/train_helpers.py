import os

import pytorch_lightning as pl
import torch
from hyperopt import hp
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .. import models

__all__ = ['get_training_params', 'load_model']


def get_training_params(cfg: DictConfig):
    """

    Parameters
    ----------
    cfg: DictConfig : hydra configuration (examples in conf/train)
        

    Returns
    -------

    """
    print(cfg.pretty())

    exp_path = os.getcwd()
    logger = TensorBoardLogger(os.path.join(exp_path, "tb_logs"))

    debug = cfg.debug

    checkpoint_dir = os.path.join(exp_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "{epoch:02d}-{val_loss:.2f}"))

    profiler = pl.profiler.SimpleProfiler()
    gpus = cfg.gpus
    if type(gpus) == ListConfig:
        gpus = [int(x) for x in gpus]

    if debug:
        torch.autograd.set_detect_anomaly(True)

    trainer_configuration = {
        'gpus': gpus,
        'benchmark': True,
        'max_epochs': cfg.training.epochs,
        'fast_dev_run': debug,
        'checkpoint_callback': ckpt_cb,
        'track_grad_norm': 2,
        'weights_summary': 'top',
        'logger': logger,
        'profiler': profiler
    }

    if cfg.training.early_stopping > 0:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.01,
            patience=cfg.training.early_stopping,
            verbose=False,
            mode='min')
        trainer_configuration['early_stop_callback'] = early_stop_callback

    if cfg.resume:
        trainer_configuration['resume_from_checkpoint'] = cfg.load_path

    return trainer_configuration


def load_model(cfg):
    print('Loading training')
    model = getattr(models,
                    cfg.training.module).load_from_checkpoint(cfg.load_path)
    return model
