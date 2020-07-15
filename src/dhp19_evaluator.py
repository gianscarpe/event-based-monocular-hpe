import logging

import hydra
from experimenting.utils import dhp19_evaluate_procedure
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path='./confs/train/config.yaml')
def main(cfg: DictConfig) -> None:

    dhp19_evaluate_procedure(cfg)
    print(cfg.pretty())


if __name__ == '__main__':
    main()
