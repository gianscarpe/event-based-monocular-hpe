import json
import logging
import os

import hydra
from omegaconf import DictConfig

from experimenting import agents
from experimenting.utils.skeleton_helpers import Skeleton
from experimenting.utils.trainer import HydraTrainer

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path='../confs/train/eval.yaml')
def main(cfg: DictConfig) -> None:
    trainer = HydraTrainer(cfg)
    normalized_predictions = trainer.get_raw_test_outputs()
    for idx, pred in enumerate(normalized_predictions):
        sk = Skeleton(pred)
        test_index = trainer.core.test_indexes[idx]
        frame_info = 
        sk.denormalize


if __name__ == '__main__':
    main()
