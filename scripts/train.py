import logging

import comet_ml
import hydra
from omegaconf import DictConfig

from experimenting.utils.train_helpers import fit

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path='../confs/train', config_name='config.yaml')
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())
    trainer = fit(cfg)
    results = trainer.test()
    logging.info("RESULTS: ", results)


if __name__ == '__main__':
    main()
