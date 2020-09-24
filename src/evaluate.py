import json
import logging
import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from experimenting import agents
from experimenting.utils import get_checkpoint_path

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path='./confs/train/config.yaml')
def main(cfg: DictConfig) -> None:

    print(cfg.pretty())

    load_path = cfg.load_path
    result_path = os.path.join(cfg.load_path, 'classification.json')
    print("Loading from ... ", load_path)
    if (os.path.exists(load_path)):
        checkpoint_path = get_checkpoint_path(load_path)
        model = getattr(agents, cfg.training.module).load_from_checkpoint(
            checkpoint_path)
        breakpoint()
        trainer = pl.Trainer(gpus=cfg.gpus, resume_from_checkpoint=checkpoint_path)
        results = trainer.test(model)
        breakpoint()
        with open(result_path, 'w') as json_file:
            json.dump(results, json_file)

        print("Model loaded")

    else:
        print(f"Error loading, {load_path} does not exist!")


if __name__ == '__main__':
    main()
