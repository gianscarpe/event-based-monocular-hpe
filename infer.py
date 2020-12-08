import logging
import argparse
from experimenting.utils.evaluation_helpers import load_model
from omegaconf import DictConfig
import hydra
logging.basicConfig(level=logging.INFO)

@hydra.main(config_path='confs/train/', config_name='infer.yaml')
def main(cfg: DictConfig) -> None:
    __import__("pdb").set_trace()
    print("OK")
    model = load_model(cfg, test_metrics=[], **cfg)



if __name__ == '__main__':
    main()
