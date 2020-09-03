"""
Toolbox for helping training and evaluation of agents

"""
import collections
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

__all__ = [
    'get_training_params', 'load_model', 'fit', 'dhp19_evaluate_procedure'
]


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


def load_model(cfg: dict):
    print('Loading training')
    model = getattr(agents, cfg['training']['module']).load_from_checkpoint(
        cfg['load_path'])
    return model


def _safe_train_end(trainer_configuration):
    exp_path = Path(os.getcwd())
    exp_name = exp_path.name
    error_path = os.path.join(exp_path.parent, 'with_errors', exp_name)
    shutil.move(exp_path, error_path)
    logging.log(logging.ERROR, 'exp moved to trash')


def fit(cfg) -> pl.Trainer:
    """
    Launch training for a given config. Confs file can be found at /src/confs

    Args:
       cfg (omegaconf.DictConfig): Config dictionary

    """
    trainer_configuration = get_training_params(cfg)
    if cfg.load_path:
        print('Loading training')
        checkpoint_dir = cfg.load_path
        checkpoints = sorted(os.listdir(checkpoint_dir))
        checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
        print(f'Loading {checkpoint_path} ...')
        model = getattr(
            agents, cfg.training.module).load_from_checkpoint(checkpoint_path)
    else:
        model = getattr(agents, cfg.training.module)(cfg)

    try:
        trainer = pl.Trainer(**trainer_configuration)
        trainer.fit(model)
        return trainer
    except Exception as ex:
        _safe_train_end(trainer_configuration)
        raise ex


def dhp19_evaluate_procedure(cfg, metrics=None):
    """
    Retrieve trained agent using cfg and apply its evaluation protocol to
    extract results

    Args: cfg (omegaconf.DictConfig): Config dictionary (need to specify a
          load_path and a training task)


    Returns:
        Results obtained applying the dataset evaluation protocol, per metric
    """

    if metrics is None:
        metrics = ['test_meanMPJPE', 'test_meanAUC', 'test_meanPCK']

    checkpoint_dir = cfg.load_path
    checkpoints = sorted(os.listdir(checkpoint_dir))
    load_path = os.path.join(checkpoint_dir, checkpoints[0])

    print("Loading from ... ", load_path)
    if os.path.exists(load_path):
        model = getattr(agents,
                        cfg.training.module).load_from_checkpoint(load_path)
    else:
        raise FileNotFoundError()

    final_results = collections.defaultdict(dict)

    for movement in range(0, 33):
        model._hparams.dataset.movements = [movement]

        trainer = pl.Trainer(gpus=cfg['gpus'],
                             benchmark=True,
                             limit_val_batches=0.10,
                             weights_summary='top')
        trainer.test(model)
        results = model.results

        print(f"Movement {movement}")
        print(results)
        for metric in metrics:
            tensor_result = results[metric]
            if len(tensor_result.shape) > 0:  # list of values cannot be
                # converted to python numeric!
                tensor_result = tensor_result.tolist()
            else:
                tensor_result = float(tensor_result)

            final_results[metric][f'movement_{movement}'] = tensor_result

    return final_results
