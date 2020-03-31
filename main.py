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

from pytorch_lightning.loggers import TensorBoardLogger
from dataset import get_dataset, get_dataloader
import models

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

        self.cnn = getattr(models,
                           self.train_config['model'])(self.data_config['n_channels'],
                                                       self.data_config['n_classes'])

    def forward(self, x):
        x = self.cnn(x)
        return x
    
    def prepare_data(self):
        root_dir = self.data_config['root_dir']
        batch_size = self.data_config['batch_size']
        n_channels = self.data_config['n_channels']

        data_dir = os.path.join(root_dir,'movements_per_frame')
        preload_dir = os.path.join(root_dir, 'preload')

        if os.path.exists(preload_dir):
            from utils.generate_indexes import load_npy_indexes_and_map
            file_paths, train_index, val_index, test_index = load_npy_indexes_and_map(data_dir)
        else:
            from utils.generate_indexes import save_npy_indexes_and_map
            file_paths, train_index, val_index, test_index = save_npy_indexes_and_map(data_dir, split_at=0.8)

        self.train_dataset = get_dataset(file_paths, train_index, self.preload, n_channels)
        self.val_dataset = get_dataset(file_paths, val_index, self.preload, n_channels)
        self.test_dataset = get_dataset(file_paths, test_index, self.preload, n_channels)

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
        return [optimizer], [ReduceLROnPlateau(optimizer)]

    def training_step(self, batch, batch_idx):
        b_x, b_y = batch
        b_x = b_x.float()

        output = self.forward(b_x)               # cnn output
        loss = self.loss_func(output, b_y)   # cross entropy loss

        logs = {"loss":loss}
        return {"loss":loss, "log":logs}

    def validation_step(self, batch, batch_idx): 
        b_x, b_y = batch
        b_x = b_x.float()

        output = self.forward(b_x)               # cnn output
        loss = self.loss_func(output, b_y)   # cross entropy loss
        _, pred_y = torch.max(output.data, 1)
        correct_predictions = (pred_y == b_y).sum().item()
        num_predictions = pred_y.size()

        return {"val_loss":loss, "correct_predictions":correct_predictions, "n_predictions":num_predictions}

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        acc = torch.Tensor([x['correct_predictions'] for x in outputs]).sum() / torch.Tensor([x['n_predictions'] for x in outputs]).sum() *100
        
        tensorboard_logs = {'val_loss': avg_loss, "val_acc":acc}
        return {'avg_val_loss': avg_loss, 'val_acc':acc,'log': tensorboard_logs}

       
            
                

if __name__ == '__main__':
    # TRAIN ARGUMENTS -- refer to config.json for train and data configuration
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config-path', type=str, help='Path to confg json')
    parser.add_argument('--preload', default=False, action='store_true')
    parser.add_argument('--multi-gpu', type=bool, default=False)
    parser.add_argument('--num-workers', type=int, default=6)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    exp_path = "/home/gianscarpe/dev/exps"
    exp_name = os.path.basename(args.config_path).split(".")[0]
    logger = TensorBoardLogger(exp_path, name=exp_name)

    model = Model(hparams=args)
    trainer = pl.Trainer(gpus=1, benchmark=True, early_stop_callback=True, logger=logger, profiler=True)
    trainer.fit(model)
