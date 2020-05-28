import logging

import hydra
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, tpe
from omegaconf import DictConfig

from experimenting.utils import fit, get_hypersearch_cfg

logging.basicConfig(level=logging.INFO)


def objective(cfg: dict):
    try:
        trainer = fit(cfg)
        breakpoint()
        results = trainer
        trial_path = trainer.checkpoint_callback.filename
        return {
            'loss': results[-1],
            'status': STATUS_OK,
            'model_path': trial_path
        }
    except Exception as ex:
        print(ex)
        return {'loss': 0, 'status': STATUS_FAIL, 'model_path': ''}


@hydra.main(config_path='./confs/train/config.yaml')
def main(cfg: DictConfig) -> None:
    space = get_hypersearch_cfg(cfg)
    best = fmin(objective, space, algo=tpe.suggest, max_evals=100)
    print(best)


if __name__ == '__main__':
    main()
