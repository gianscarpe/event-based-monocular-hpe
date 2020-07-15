import logging
import os

import hydra
import pytorch_lightning as pl
from experimenting import models
from experimenting.dataset import DatasetType, get_datasets
from experimenting.utils import dhp19_evaluate_procedure
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path='./confs/train/config.yaml')
def main(cfg: DictConfig) -> None:

    dhp19_evaluate_procedure(cfg)
    print(cfg.pretty())
    


if __name__ == '__main__':
    main()
