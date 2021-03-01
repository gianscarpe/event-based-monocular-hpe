import json
import logging
import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from experimenting import agents
from experimenting.utils.evaluation_helpers import evaluate

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path='../confs/train/eval.yaml')
def main(cfg: DictConfig) -> None:

    print(cfg.pretty())

    evaluate(cfg, save_results=True)


if __name__ == '__main__':
    main()
