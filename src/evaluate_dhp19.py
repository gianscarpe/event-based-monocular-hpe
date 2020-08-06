import logging

import hydra
from omegaconf import DictConfig

from experimenting.utils import dhp19_evaluate_procedure

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path='./confs/train/config.yaml')
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())
    dhp19_evaluate_procedure(cfg)


if __name__ == '__main__':
    main()
