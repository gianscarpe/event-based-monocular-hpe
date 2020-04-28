import os
import hydra
from omegaconf import DictConfig
from experimenting import Model

import pytorch_lightning as pl

import logging
logging.basicConfig(level=logging.INFO)

@hydra.main(config_path='./confs/train/config.yaml')
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())

    load_path = cfg.training.load_path
    print("Loading from ... ", load_path)
    if (os.path.exists(load_path)):
        model = Model.load_from_checkpoint(load_path)
        print("Model loaded")

    else:
        print(f"Error loading, {load_path} does not exist!")

    
if __name__ == '__main__':
    main()
    
