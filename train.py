import os
import hydra
from omegaconf import DictConfig
import experimenting
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import logging

logging.basicConfig(level=logging.INFO)

@hydra.main(config_path='./confs/train/config.yaml')
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())

    exp_path = os.getcwd()
    log_path, cur_dir = os.path.split(exp_path)
    logger = TensorBoardLogger(os.path.join(cur_dir, "tb_logs"),
                               version=cur_dir)

    
    debug = cfg.debug
    
    checkpoint_dir = os.path.join(exp_path,"checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(filepath=os.path.join(checkpoint_dir, "{epoch:02d}-{val_loss:.2f}"))

    profiler = pl.profiler.SimpleProfiler()

    if debug:
        torch.autograd.set_detect_anomaly(True)
    trainer_configuration = {
        'gpus':[1], 'benchmark':True, 'max_epochs':cfg.training.epochs,
        'fast_dev_run':debug,
        'checkpoint_callback':ckpt_cb, 'track_grad_norm':2,
        'weights_summary': 'top', 'logger':logger,
        'gradient_clip_val':1.5,
        'profiler':profiler}

    if cfg.training.early_stopping > 0:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.01,
            patience=cfg.training.early_stopping,
            verbose=False,
            mode='min'
        )
        trainer_configuration['early_stop_callback'] = early_stop_callback

    model = getattr(experimenting, cfg.training.module)(cfg)
    
    if cfg.load_path:
        print('Loading training')
        model = getattr(experimenting,
                        cfg.training.module).load_from_checkpoint(cfg.load_path)
    if cfg.resume:
        trainer_configuration['resume_from_checkpoint'] = cfg.load_path
        
    trainer = pl.Trainer(**trainer_configuration)
    trainer.fit(model)
    trainer.test(model)

    

    
if __name__ == '__main__':
    main()
    
