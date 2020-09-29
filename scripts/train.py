import logging

import hydra
from omegaconf import DictConfig

from experimenting.utils import fit

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path='./confs/train/config.yaml')
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())
    trainer = fit(cfg)
    trainer.test()


if __name__ == '__main__':
    main()
