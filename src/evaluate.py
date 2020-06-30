import logging
import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from experimenting import MargiposeEstimator as Model

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path='./confs/train/config.yaml')
def main(cfg: DictConfig) -> None:

    print(cfg.pretty())

    load_path = cfg.load_path
    print("Loading from ... ", load_path)
    if (os.path.exists(load_path)):
        model = Model
        .load_from_checkpoint(load_path)
        trainer = pl.Trainer(gpus=1, benchmark=True, weights_summary='top')
        trainer.test(model)
        print("Model loaded")

    else:
        print(f"Error loading, {load_path} does not exist!")


if __name__ == '__main__':
    main()
