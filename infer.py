from functools import partial
import pytorch_lightning as pl
from timeit import default_timer as timer
import timeit
import logging
import argparse
from experimenting.dataset import Joints3DConstructor, get_dataloader, AutoEncoderConstructor
from experimenting.utils.evaluation_helpers import load_model
from omegaconf import DictConfig
import hydra
logging.basicConfig(level=logging.INFO)

def get_loader(cfg):
    factory = AutoEncoderConstructor(cfg)
    _, _, test = factory.get_datasets()

    loader = get_dataloader(dataset=test,
                            batch_size=32,
                            shuffle=False,
                            num_workers=12)
    return loader




@hydra.main(config_path='confs/train/', config_name='infer.yaml')
def main(cfg: DictConfig) -> None:
    model = load_model(cfg, **cfg)
    model.eval()
    test_loader = get_loader(cfg)
    test_iter = iter(test_loader)

    test_data = next(test_iter)
    start = timer()
    model(test_data)

    end = timer()    

    logging.info(f"Inference time {end - start}")
    
if __name__ == '__main__':
    main()
