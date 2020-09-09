import json
import logging
import os

import hydra
from experimenting.utils import dhp19_evaluate_procedure
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path='./confs/train/config.yaml')
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())
    results = dhp19_evaluate_procedure(cfg)

    with open(os.path.join(cfg.load_path, 'auc.json'), 'w') as json_file:
        json.dump(results, json_file)


if __name__ == '__main__':
    main()
