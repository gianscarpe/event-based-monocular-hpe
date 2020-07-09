import logging
import os
import shutil
from pathlib import Path

import numpy as np
import torch

import pytorch_lightning as pl
from hyperopt import hp
from omegaconf import ListConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .. import models

logging.basicConfig(level=logging.INFO)

__all__ = ['get_training_params', 'load_model', 'fit', 'get_hypersearch_cfg']


def get_training_params(cfg: dict):
    """

    Parameters
    ----------
    cfg: DictConfig :
        hydra configuration (examples in conf/train)

    Returns
    -------

    """

    exp_path = os.getcwd()
    logger = TensorBoardLogger(os.path.join(exp_path, "tb_logs"))

    debug = cfg['debug']

    checkpoint_dir = os.path.join(exp_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "{epoch:02d}-{val_loss:.2f}"))

    profiler = pl.profiler.SimpleProfiler()
    gpus = cfg['gpus']
    if type(gpus) == list or type(gpus) == ListConfig:
        gpus = [int(x) for x in gpus]

    trainer_configuration = {
        'gpus': gpus,
        'benchmark': True,
        'max_epochs': cfg['training']['epochs'],
        'checkpoint_callback': ckpt_cb,
        'track_grad_norm': 2,
        'weights_summary': 'top',
        'logger': logger,
        'profiler': profiler,
    }

    torch.autograd.set_detect_anomaly(debug)
    if debug:
        trainer_configuration['overfit_pct'] = 0.01
        trainer_configuration['']

    if cfg.training.early_stopping > 0:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.01,
            patience=cfg['training']['early_stopping'],
            verbose=False,
            mode='min')
        trainer_configuration['early_stop_callback'] = early_stop_callback

    if cfg.resume:
        trainer_configuration['resume_from_checkpoint'] = cfg['load_path']

    return trainer_configuration


def load_model(cfg: dict):
    print('Loading training')
    model = getattr(models, cfg['training']['module']).load_from_checkpoint(
        cfg['load_path'])
    return model


def _safe_train_end(trainer_configuration):
    exp_path = Path(os.getcwd())
    exp_name = exp_path.name
    error_path = os.path.join(exp_path.parent, 'with_errors', exp_name)
    shutil.move(exp_path, error_path)
    logging.log(logging.ERROR, 'exp moved to trash')


def _get_hyperopt_optimizer():
    return hp.choice('hp_optimizer', [{
        'type': 'SGD',
        'use_lr_scheduler': True,
        'params': {
            'lr': hp.loguniform('sgd_optimizer_lr', np.log(0.001),
                                np.log(0.2)),
            'nesterov': True,
            'momentum': 0.9
        }
    }, {
        'type': 'Adam',
        'params': {
            'lr': hp.loguniform('adam_optimizer_lr', np.log(0.001),
                                np.log(0.2))
        }
    }])


def _get_hyperopt_training(training_config: dict):
    """

    Parameters
    ----------
    training_config: dict :
        Hydra training config, loaded from confs/train/training

    Returns
    -------
    Config with hypersearch parameters

    """

    training_config['backbone'] = hp.choice(
        'backbone',
        ['classification/resnet34.pt', 'classification/resnet50.pt'])
    training_config['stages'] = hp.uniformint('n_stages', 0, 5)
    return training_config


def get_hypersearch_cfg(cfg):
    cfg = OmegaConf.to_container(cfg)
    cfg['optimizer'] = _get_hyperopt_optimizer()
    cfg['training'] = _get_hyperopt_training(cfg['training'])
    return cfg


def fit(cfg) -> pl.Trainer:
    """
    Main function for executing training of models
    Parameters
    ----------
    cfg :
        configuration for train; configurations can be found at /confs/train

    Returns
    -------
    Trainer object
    """
    trainer_configuration = get_training_params(cfg)
    if cfg.load_path:
        print('Loading training')
        checkpoint_dir = cfg.load_path
        checkpoints = sorted(os.listdir(checkpoint_dir))
        checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
        print(f'Loading {checkpoint_path} ...')
        model = getattr(models, cfg.training.module).load_from_checkpoint(
            checkpoint_path)
    else:
        model = getattr(models, cfg.training.module)(cfg)

    try:
        trainer = pl.Trainer(**trainer_configuration)
        trainer.fit(model)
        return trainer
    except Exception as ex:
        _safe_train_end(trainer_configuration)
        raise (ex)
