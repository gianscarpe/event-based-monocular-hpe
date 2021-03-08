"""
Toolbox for DHP19 evaluation procedure
"""
import collections
import json
import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, ListConfig

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
            final_results[metric][f"movement_{loader_id}"] = tensor_result

    return final_results
