"""
Toolbox for helping training and evaluation of agents
"""

import glob
import logging
import os
import shutil
from pathlib import Path

import comet_ml
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import (
    CometLogger,
    LightningLoggerBase,
    TensorBoardLogger,
    WandbLogger,
)

import experimenting

logging.basicConfig(level=logging.INFO)


def get_training_params(cfg: DictConfig):
    """

    Parameters

    cfg: DictConfig :
        hydra configuration (examples in conf/train)

    -------

    """

    exp_path = os.getcwd()
    logger = [TensorBoardLogger(os.path.join(exp_path, "tb_logs"))]

    debug = cfg["debug"]

    checkpoint_dir = os.path.join(exp_path, "checkpoints")
    log_profiler = os.path.join(exp_path, "profile.txt")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "{epoch:02d}-{val_loss:.2f}")
    )

    profiler = pl.profiler.AdvancedProfiler(log_profiler)
    gpus = cfg["gpus"]
    if type(gpus) == list or type(gpus) == ListConfig:
        gpus = [int(x) for x in gpus]

    trainer_configuration = {
        "gpus": gpus,
        "benchmark": True,
        "max_epochs": cfg["epochs"],
        "callbacks": [ckpt_cb],
        "track_grad_norm": 2,
        "weights_summary": "top",
        "logger": logger,
        "profiler": profiler,
    }

    if ((isinstance(gpus, list) and len(gpus) > 1)) or (
        (isinstance(gpus, int) and gpus > 1)
    ):
        if "accelerator" in cfg:
            trainer_configuration['accelerator'] = cfg["accelerator"]

    if debug:
        torch.autograd.set_detect_anomaly(debug)
        trainer_configuration["overfit_batches"] = 0.0005
        trainer_configuration["log_gpu_memory"] = True

    if cfg['early_stopping'] > 0:
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.01,
            patience=cfg["early_stopping"],
            verbose=False,
            mode="min",
        )
        trainer_configuration["callbacks"].append(early_stop_callback)

    if cfg.resume:
        trainer_configuration["resume_from_checkpoint"] = cfg["load_path"]

    return trainer_configuration


def load_model(load_path: str, module: str, **kwargs):
    """
    Main function to load a checkpoint.
    Args:
        load_path: path to the checkpoint directory
        module: python module (e.g., experimenting.agents.Base)
        kwargs: arguments to override while loading checkpoint

    Returns
        Lightning module loaded from checkpoint, if exists
    """
    print("Loading training")
    load_path = get_checkpoint_path(load_path)
    print("Loading from ... ", load_path)

    if os.path.exists(load_path):

        model = getattr(experimenting.agents, module).load_from_checkpoint(
            load_path, **kwargs
        )
    else:
        raise FileNotFoundError()

    return model


def get_checkpoint_path(checkpoint_dir):
    # CHECKPOINT file are
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.ckpt")))
    load_path = os.path.join(checkpoint_dir, checkpoints[0])
    return load_path


def get_result_path(load_path):
    return os.path.join(load_path, "result.json")


def _safe_train_end(trainer_configuration):
    exp_path = Path(os.getcwd())
    exp_name = exp_path.name
    error_path = os.path.join(exp_path.parent, "with_errors", exp_name)
    shutil.move(exp_path, error_path)
    logging.log(logging.ERROR, "exp moved to trash")


def _get_comet_logger(exp_name: str, project_name: str) -> LightningLoggerBase:
    # arguments made to CometLogger are passed on to the comet_ml.Experiment class
    comet_logger = CometLogger(
        api_key=os.environ.get('COMET_API_KEY'),
        experiment_name=exp_name,
        project_name=project_name,
    )
    return comet_logger


def _get_wandb_logger(exp_name: str, project_name: str) -> LightningLoggerBase:
    logger = WandbLogger(name=exp_name, project=project_name,)
    return logger


def _instantiate_new_model(
    cfg: DictConfig, core: experimenting.dataset.BaseCore
) -> pl.LightningModule:
    """
    Instantiate new module from scratch using provided `hydra` configuration
    """
    model = getattr(experimenting.agents, cfg.training.module)(
        loss=cfg.loss,
        optimizer=cfg.optimizer,
        lr_scheduler=cfg.lr_scheduler,
        model_zoo=cfg.model_zoo,
        core=core,
        **cfg.training
    )
    return model


def fit(cfg) -> pl.Trainer:
    """
    Launch training for a given config. Confs file can be found at /src/confs

    Args:
       cfg (omegaconf.DictConfig): Config dictionary

    """
    trainer_configuration = get_training_params(cfg)

    core = hydra.utils.instantiate(cfg.dataset)

    if cfg.load_path:
        print("Loading training")
        model = load_model(
            cfg.load_path,
            cfg.training.module,
            model_zoo=cfg.model_zoo,
            core=core,
            loss=cfg.loss,
            optimizer=cfg.optimizer,
            lr_scheduler=cfg.lr_scheduler,
            # TODO should remove the following, as they're loaded from the checkpoint
            backbone=cfg.training.backbone,
            model=cfg.training.model,
        )
    else:
        model = _instantiate_new_model(cfg, core)

    data_module = experimenting.dataset.DataModule(
        core=core,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        dataset_factory=model.get_data_factory(),
        aug_train_config=cfg.augmentation_train,
        aug_test_config=cfg.augmentation_test,
    )

    try:
        trainer = pl.Trainer(**trainer_configuration)
        trainer.fit(model, datamodule=data_module)
        return trainer
    except Exception as ex:
        _safe_train_end(trainer_configuration)
        print(ex)
        raise ex
