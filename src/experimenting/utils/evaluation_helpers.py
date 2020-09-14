"""
Toolbox for DHP19 evaluation procedure
"""
import collections

import pytorch_lightning as pl

from ..dataset import Joints3DConstructor, get_dataloader
from .train_helpers import _get_checkpoint_path, load_model


def _get_test_loaders_iterator(cfg):
    for movement in range(0, 33):
        cfg.dataset.movements = [movement]
        factory = Joints3DConstructor(cfg)
        _, _, test = factory.get_datasets()

        loader = get_dataloader(dataset=test,
                                batch_size=1,
                                shuffle=False,
                                num_workers=6)
        yield loader


def evaluate_per_movement(cfg, metrics=None):
    """
    Retrieve trained agent using cfg and apply its evaluation protocol to
    extract results

    Args: cfg (omegaconf.DictConfig): Config dictionary (need to specify a
          load_path and a training task)


    Returns:
        Results obtained applying the dataset evaluation protocol, per metric
    """

    if metrics is None:
        metrics = ['test_meanMPJPE', 'test_meanPCK', 'test_meanAUC']

    model = load_model(cfg)
    load_path = _get_checkpoint_path(cfg)
    final_results = collections.defaultdict(dict)
    test_loaders = _get_test_loaders_iterator(cfg)
    trainer = pl.Trainer(gpus=cfg.gpus, resume_from_checkpoint=load_path)

    for loader_id, loader in enumerate(test_loaders):
        results = trainer.test(model, test_dataloaders=loader)[0]

        print(f"Step {loader_id}")
        print(results)
        for metric in metrics:
            tensor_result = results[metric]
            final_results[metric][f'movement_{loader_id}'] = tensor_result

    return final_results
