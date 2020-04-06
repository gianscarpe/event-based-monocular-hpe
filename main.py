import os
import torch
from torch import nn
import numpy as np
import torchvision
import time
import argparse
import json
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from callbacks import AvgLossCallback
from pytorch_lightning.loggers import TensorBoardLogger
from dataset import get_dataset, get_dataloader
import models
from torch.utils.tensorboard import SummaryWriter
import torchvision
from pytorch_lightning.profiler import Profiler
import logging
import pandas as pd
logging.basicConfig(level=logging.INFO)
import collections
from albumentations import *
from albumentations.pytorch import ToTensor
import torch.nn.functional as F
 

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
            return dict(items)
                                                


class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()
        
        self.num_workers = hparams.num_workers
        self.preload = hparams.preload
        
        config_path = hparams.config_path
        with open(config_path) as json_file:
            config = json.load(json_file)

        
        self.train_config = config['train']
        self.data_config = config['data']
        self.loss_func = getattr(nn, config['train']['loss'])()


        self.model = getattr(models,
                           self.train_config['model'])(self.data_config['n_channels'],
                                                       self.data_config['n_classes'])

    def forward(self, x):
        x = self.model(x)
        return x

    def prepare_data(self):
        root_dir = self.data_config['root_dir']
        batch_size = self.data_config['batch_size']
        n_channels = self.data_config['n_channels']

        data_dir = os.path.join(root_dir,'movements_per_frame')
        preload_dir = os.path.join(data_dir, 'preload')

        if os.path.exists(preload_dir):
            from utils.generate_indexes import load_npy_indexes_and_map
            file_paths, train_index, val_index, test_index = load_npy_indexes_and_map(data_dir)
        else:
            from utils.generate_indexes import save_npy_indexes_and_map
            file_paths, train_index, val_index, test_index = save_npy_indexes_and_map(data_dir, split_at=0.8)

        transform_train = Compose([
            ToTensor(),
            
        ])
        transform_val = Compose([
            ToTensor()
        ])

        
        self.train_dataset = get_dataset(file_paths, train_index, False, n_channels, preprocess=transform_val)
        self.val_dataset = get_dataset(file_paths, val_index, False, n_channels, preprocess=transform_val)
        self.test_dataset = get_dataset(file_paths, test_index, False, n_channels, preprocess=transform_val)

    def train_dataloader(self):

        train_loader = get_dataloader(self.train_dataset,
                                      self.data_config['batch_size'], self.num_workers)
        return train_loader

    def val_dataloader(self):
        val_loader = get_dataloader(self.val_dataset,
                                      self.data_config['batch_size'], self.num_workers)
        return val_loader
        

    def configure_optimizers(self):
        optimizer = getattr(torch.optim,
                    self.train_config['optimizer']['type'])(params=self.parameters(),
                                                            **self.train_config['optimizer']['params'])

        return optimizer

    def training_step(self, batch, batch_idx):
        b_x, b_y = batch
        sample_imgs = b_x[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('Batch images', grid, 0)

        
        output = self.forward(b_x)               # cnn output
        loss = self.loss_func(output, b_y)   # cross entropy loss

        logs = {"loss":loss}
        return {"loss":loss, "log":logs}


    def validation_step(self, batch, batch_idx): 
        b_x, b_y = batch
        output = self.forward(b_x)               # cnn output
        loss = self.loss_func(output, b_y)   # cross entropy loss
        _, pred_y = torch.max(output.data, 1)
        correct_predictions = (pred_y == b_y).sum().item()

        
        return {"batch_val_loss":loss, "correct_predictions":correct_predictions}

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['batch_val_loss'] for x in outputs]).mean()
        acc = sum([x['correct_predictions'] for x in outputs]) / len(self.val_dataset) * 100
        
        logs = {'val_loss': avg_loss, "val_acc":acc, 'step': self.current_epoch}
        
        return {'val_loss': avg_loss, 'val_acc':acc,'log': logs, 'progress_bar': logs}


if __name__ == '__main__':
    # TRAIN ARGUMENTS -- refer to config.json for train and data configuration
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config-path', type=str, help='Path to confg json')
    parser.add_argument('--preload', default=False, action='store_true')
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=100)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    root_path = "/home/gianscarpe/dev/exps/balanced_exps"
    exp_name = os.path.basename(args.config_path).split(".")[0]
    exp_path = os.path.join(root_path, exp_name)
    os.makedirs(exp_path, exist_ok=True)

    logger = TensorBoardLogger(root_path, name=exp_name)
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=10,
        verbose=False,
        mode='min'
    )
    
    checkpoint_dir = os.path.join(logger.log_dir,"checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    ckpt_cb = ModelCheckpoint(filepath=os.path.join(checkpoint_dir, exp_name + "_{epoch:02d}-{val_loss:.2f}"))

    profiler = Profiler()
    model = Model(hparams=args)

    trainer = pl.Trainer(gpus=1, benchmark=True, max_epochs=args.epochs,
                         track_grad_norm=2, weights_summary='full', logger=logger, profiler=profiler,
                         callbacks=[AvgLossCallback()])
    trainer.fit(model)
