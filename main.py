import os
import hydra
from omegaconf import DictConfig
from module import Model

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from callbacks import AvgLossCallback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import Profiler

import logging
logging.basicConfig(level=logging.INFO)

@hydra.main(config_path='conf/config.yaml')
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())
    exp_path = os.getcwd()

    logger = TensorBoardLogger(exp_path)
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.01,
        patience=cfg.training.early_stopping,
        verbose=False,
        mode='min'
    )
    
    checkpoint_dir = os.path.join(exp_path,"checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(filepath=os.path.join(checkpoint_dir, "{epoch:02d}-{val_loss:.2f}"))
    
    profiler = Profiler()
    model = Model(cfg)

    trainer = pl.Trainer(gpus=1, benchmark=True, max_epochs=cfg.training.epochs,
                         early_stop_callback=early_stop_callback, checkpoint_callback=ckpt_cb,
                         track_grad_norm=2, weights_summary='full', logger=logger, profiler=profiler,
                         callbacks=[AvgLossCallback()])
    trainer.fit(model)

    

    
if __name__ == '__main__':
    main()
    
