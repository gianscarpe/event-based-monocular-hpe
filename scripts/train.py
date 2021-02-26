import logging

import comet_ml
import hydra
from omegaconf import DictConfig

from experimenting.utils.train_helpers import fit

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path='../confs/train', config_name='config.yaml')
def main(cfg: DictConfig) -> None:
    trainer = fit(cfg)
    trainer.test()


if __name__ == '__main__':
    main()
