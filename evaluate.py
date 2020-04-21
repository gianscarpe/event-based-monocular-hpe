import os
import hydra
from omegaconf import DictConfig
from module import Model

import pytorch_lightning as pl

import logging
logging.basicConfig(level=logging.INFO)

@hydra.main(config_path='conf/config.yaml')
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())
    breakpoint()
    load_path = cfg.load_path
    print(load_path)
    model = Model.load_from_checkpoint(load_path, loading=True)
    print("Model loaded")

    
if __name__ == '__main__':
    main()
    
