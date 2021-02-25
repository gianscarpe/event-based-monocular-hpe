"""
Toolbox for DHP19 evaluation procedure
"""
import collections
import json
import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

import experimenting

from ..dataset import Joints3DConstructor
from ..dataset.datamodule import get_dataloader
from .utilities import get_checkpoint_path, load_model


def _get_test_loaders_iterator(cfg):
    for movement in range(0, 33):
        cfg.dataset.movements = [movement]
        factory = Joints3DConstructor(cfg)
        _, _, test = factory.get_datasets()

        loader = get_dataloader(
            dataset=test, batch_size=32, shuffle=False, num_workers=12
        )
        yield loader


def evaluate_per_movement(cfg):
    """
    Retrieve trained agent using cfg and apply its evaluation protocol to
    extract results

    Args: cfg (omegaconf.DictConfig): Config dictionary (need to specify a
          load_path and a training task)


    Returns:
        Results obtained applying the dataset evaluation protocol, per metric
    """

    test_metrics = cfg.training.metrics
    metrics = ["test_mean" + k for k in test_metrics]

    model = load_model(cfg, test_metrics=test_metrics)

    final_results = collections.defaultdict(dict)
    test_loaders = _get_test_loaders_iterator(cfg)
    trainer = pl.Trainer(gpus=cfg.gpus)

    for loader_id, loader in enumerate(test_loaders):
        results = trainer.test(model, test_dataloaders=loader)[0]

        print(f"Step {loader_id}")
        print(results)
        for metric in metrics:
            tensor_result = results[metric]
            final_results[metric][f'movement_{loader_id}'] = tensor_result

    return final_results


def evaluate(cfg: DictConfig):
    """
    Launch training for a given config. Confs file can be found at /src/confs

    Args:
       cfg (DictConfig): Config dictionary

    """

    core = hydra.utils.instantiate(cfg.dataset)

    if os.path.exists(cfg.load_path):
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
        raise Exception("Checkpoint dir not provided")

    data_module = experimenting.dataset.DataModule(
        core=core,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        dataset_factory=model.get_data_factory(),
        aug_train_config=cfg.augmentation_train,
        aug_test_config=cfg.augmentation_test,
    )

    trainer = pl.Trainer()
    trainer.test(model, datamodule=data_module)
    return trainer
