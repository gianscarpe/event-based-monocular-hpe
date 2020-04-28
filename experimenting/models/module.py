import torch
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..dataset  import get_dataset, get_dataloader, save_npy_indexes_and_map, load_npy_indexes_and_map
from .cnns import get_cnn
import torchvision

import torch.nn.functional as F
import hydra
import pytorch_lightning as pl
import collections
import os
from ..utils import flatten, unflatten, get_augmentation
from sklearn.metrics import precision_score, recall_score, accuracy_score

        
class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()
        
        hparams = unflatten(hparams)
        self.hparams = flatten(hparams)
        
        
        self.train_config = hparams.training
        self.data_config = hparams.dataset
        self.optimizer_config = hparams.optimizer
        self.scheduler_config = None
        
        self.augmentation_train = get_augmentation(hparams.augmentation_train)
        self.augmentation_test = get_augmentation(hparams.augmentation_test)
        
        self.num_workers = self.train_config.num_workers

        self.loss_func = hydra.utils.instantiate(hparams.loss)

        params = {'n_channels': self.data_config['n_channels'],
                  'n_classes': self.data_config['n_classes']}
        print(params)
        self.model = get_cnn(hparams.model, params)
        
    def set_optional_params(self, hparams):
        if hparams.training.use_lr_scheduler:
            self.scheduler_config = hparams.lr_scheduler
        
        
    def forward(self, x):
        x = self.model(x)
        return x

    def prepare_data(self):
        data_dir = self.data_config['path']
        batch_size = self.train_config['batch_size']
        n_channels = self.data_config['n_channels']

        preload_dir = os.path.join(data_dir, 'preload')

        if os.path.exists(os.path.join(preload_dir, 'train_indexes.npy')):

            file_paths, train_index, val_index, test_index = load_npy_indexes_and_map(data_dir)
        else:

            file_paths, train_index, val_index, test_index = save_npy_indexes_and_map(data_dir,
                                                                                     split_at=.8,
                                                                                      balanced=self.data_config.balanced)
        
        self.train_dataset = get_dataset(file_paths, train_index, False,
                                         n_channels, preprocess=self.augmentation_train)
        self.val_dataset = get_dataset(file_paths, val_index, False, n_channels,
                                       preprocess=self.augmentation_test)
        
        self.test_dataset = get_dataset(file_paths, test_index, False,
                                        n_channels, preprocess=self.augmentation_test)

    def train_dataloader(self):
        train_loader = get_dataloader(self.train_dataset,
                                      self.train_config['batch_size'], self.num_workers)
        return train_loader

    def val_dataloader(self):
        val_loader = get_dataloader(self.val_dataset,
                                    self.train_config['batch_size'],
                                    self.num_workers, shuffle=False)
        
        return val_loader

    def test_dataloader(self):
        test_loader = get_dataloader(self.test_dataset,
                                    self.train_config['batch_size'],
                                    self.num_workers, shuffle=False)
                                    
        return test_loader

    

    def configure_optimizers(self):
        optimizer = getattr(torch.optim,
                            self.optimizer_config['type'])(params=self.parameters(),
                                                           **self.optimizer_config['params'])
        scheduler = None
        if self.scheduler_config:
            scheduler = getattr(torch.optim.lr_scheduler,
                                self.scheduler_config['type'])(optimizer,
                                                               **self.scheduler_config['params'])
            return [optimizer], [scheduler]

        return optimizer

    def training_step(self, batch, batch_idx):
        b_x, b_y = batch
        sample_imgs = b_x[:6]

        output = self.forward(b_x)               # cnn output
        loss = self.loss_func(output, b_y)   # cross entropy loss

        logs = {"loss":loss}
        return {"loss":loss, "log":logs}

    
    def training_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        logs = {'avg_train_loss': avg_loss,  'step': self.current_epoch}
        
        return {'train_loss': avg_loss, 'log': logs, 'progress_bar':    logs}
     

    def validation_step(self, batch, batch_idx): 
        b_x, b_y = batch
        output = self.forward(b_x)               # cnn output
        loss = self.loss_func(output, b_y)   # cross entropy loss

        _, pred_y = torch.max(output.data, 1)
        correct_predictions = (pred_y == b_y).sum().item()

        return {"batch_val_loss":loss, "y_pred":pred_y, "y_true":b_y}

    
    def validation_epoch_end(self, outputs):
        y_true = torch.cat([x['y_true'] for x in outputs]).cpu()
        y_pred = torch.cat([x['y_pred'] for x in outputs]).cpu()

        avg_loss = torch.stack([x['batch_val_loss'] for x in outputs]).mean()
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        
        logs = {'val_loss': avg_loss, "val_acc":acc, 'val_precision':precision,
                'val_recall':recall, 'step': self.current_epoch}
        
        return {'val_loss': avg_loss, 'val_acc':acc,'log': logs, 'progress_bar':    logs}
   
    
    def test_step(self, batch, batch_idx): 
        b_x, b_y = batch
        output = self.forward(b_x)               # cnn output
        loss = self.loss_func(output, b_y)   # cross entropy loss

        _, pred_y = torch.max(output.data, 1)
        correct_predictions = (pred_y == b_y).sum().item()

        return {"batch_test_loss":loss, "y_pred":pred_y, "y_true":b_y}

    
    def test_epoch_end(self, outputs):
        y_true = torch.cat([x['y_true'] for x in outputs]).cpu()
        y_pred = torch.cat([x['y_pred'] for x in outputs]).cpu()

        avg_loss = torch.stack([x['batch_test_loss'] for x in outputs]).mean()
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        
        logs = {'test_loss': avg_loss, "test_acc":acc,
                'test_precision':precision, 'test_recall':recall, 'step':
                self.current_epoch}
        
        return {**logs, 'log': logs, 'progress_bar':    logs}
