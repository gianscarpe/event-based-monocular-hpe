import json
import logging
import os

import hydra
from omegaconf import DictConfig

from experimenting.utils.evaluation_helpers import evaluate_per_movement

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path='./confs/train/config.yaml')
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())

    result_path = os.path.join(cfg.load_path, cfg.result_file)
    results = evaluate_per_movement(cfg)
    with open(result_path, 'w') as json_file:
        json.dump(results, json_file)


if __name__ == '__main__':
    main()
