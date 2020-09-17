"""
Toolbox for helping training and evaluation of agents

"""

import glob
import logging
import os
import shutil
from pathlib import Path

import torch

import pytorch_lightning as pl
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .. import agents

logging.basicConfig(level=logging.INFO)


def fit(cfg) -> pl.Trainer:
    """
    Launch training for a given config. Confs file can be found at /src/confs

    Args:
       cfg (omegaconf.DictConfig): Config dictionary

    """
    trainer_configuration = get_training_params(cfg)
    if cfg.load_path:
        print('Loading training')
        model = load_model(cfg)
    else:
        model = getattr(agents, cfg.training.module)(cfg)

    try:
        trainer = pl.Trainer(**trainer_configuration)
        trainer.fit(model)
        return trainer
    except Exception as ex:
        _safe_train_end(trainer_configuration)
        raise ex


def get_training_params(cfg: DictConfig):
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
    log_profiler = os.path.join(exp_path, 'profile.txt')
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "{epoch:02d}-{val_loss:.2f}"))

    profiler = pl.profiler.AdvancedProfiler(log_profiler)
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


def load_model(cfg: dict, **kwargs):
    print('Loading training')
    load_path = get_checkpoint_path(cfg.load_path)
    print("Loading from ... ", load_path)

    if os.path.exists(load_path):
        model = getattr(agents,
                        cfg['training']['module']).load_from_checkpoint(
                            load_path,
                            estimate_depth=cfg.training.estimate_depth,
                            **kwargs)
    else:
        raise FileNotFoundError()

    return model


def get_checkpoint_path(checkpoint_dir):
    # CHECKPOINT file are
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.ckpt")))
    load_path = os.path.join(checkpoint_dir, checkpoints[0])
    return load_path


def get_result_path(load_path):
    return os.path.join(load_path, 'result.json')


def _safe_train_end(trainer_configuration):
    exp_path = Path(os.getcwd())
    exp_name = exp_path.name
    error_path = os.path.join(exp_path.parent, 'with_errors', exp_name)
    shutil.move(exp_path, error_path)
    logging.log(logging.ERROR, 'exp moved to trash')
