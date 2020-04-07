import torch
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import get_dataset, get_dataloader
import models
import albumentations
import torchvision

from omegaconf import DictConfig, ListConfig
import torch.nn.functional as F
import hydra
import pytorch_lightning as pl
import collections
import os

from sklearn.metrics import precision_score, recall_score, accuracy_score

def flatten(d):
    out = {}
    for key, val in d.items():
        if isinstance(val, dict) or isinstance(val, DictConfig):
            val = [val]
        if isinstance(val, list) or isinstance(val, ListConfig):
            for subdict in val:
                deeper = flatten(subdict).items()
                out.update({key + '__' + key2: val2 for key2, val2 in deeper})
        else:
            out[key] = val
    return out

def unflatten(dictionary):
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split("__")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict

def get_augmentation(augmentation_specifics):
    augmentations = []
    for augmentation_class in augmentation_specifics.apply:
        aug = hydra.utils.instantiate(augmentation_class)
        augmentations.append(aug)

    return albumentations.Compose(augmentations)

        
class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()
        print(hparams)
        if hparams.loading:
            hparams = unflatten(hparams)
        self.hparams = flatten(hparams)
        self.train_config = hparams.training
        self.data_config = hparams.dataset
        self.optimizer_config = hparams.optimizer
        
        self.augmentation_train = get_augmentation(hparams.augmentation_train)
        self.augmentation_test = get_augmentation(hparams.augmentation_test)
        
        self.num_workers = self.train_config.num_workers

        self.loss_func = hydra.utils.instantiate(hparams.loss)

        self.model = getattr(models, hparams.model)(self.data_config['n_channels'],
                                                self.data_config['n_classes'])
        
    
    def forward(self, x):
        x = self.model(x)
        return x

    def prepare_data(self):
        data_dir = self.data_config['path']
        batch_size = self.train_config['batch_size']
        n_channels = self.data_config['n_channels']

        preload_dir = os.path.join(data_dir, 'preload')

        if os.path.exists(preload_dir):
            from utils.generate_indexes import load_npy_indexes_and_map
            file_paths, train_index, val_index, test_index = load_npy_indexes_and_map(data_dir)
        else:
            from utils.generate_indexes import save_npy_indexes_and_map
            file_paths, train_index, val_index, test_index = save_npy_indexes_and_map(data_dir, split_at=0.8, balanced=self.data_config.balanced)
        
        self.train_dataset = get_dataset(file_paths, train_index, False, n_channels, preprocess=self.augmentation_train)
        self.val_dataset = get_dataset(file_paths, val_index, False, n_channels, preprocess=self.augmentation_test)
        self.test_dataset = get_dataset(file_paths, test_index, False, n_channels, preprocess=self.augmentation_test)

    def train_dataloader(self):

        train_loader = get_dataloader(self.train_dataset,
                                      self.train_config['batch_size'], self.num_workers)
        return train_loader

    def val_dataloader(self):
        val_loader = get_dataloader(self.val_dataset,
                                    self.train_config['batch_size'], self.num_workers)
        return val_loader
    

    def configure_optimizers(self):
        optimizer = getattr(torch.optim,
                            self.optimizer_config['type'])(params=self.parameters(),
                                                                    **self.optimizer_config['params'])

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

        return {"batch_val_loss":loss, "y_pred":pred_y, "y_true":b_y}

    def validation_epoch_end(self, outputs):
        
        y_true = torch.cat([x['y_true'] for x in outputs]).cpu()
        y_pred = torch.cat([x['y_pred'] for x in outputs]).cpu()

        
        avg_loss = torch.stack([x['batch_val_loss'] for x in outputs]).mean()
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        
        logs = {'val_loss': avg_loss, "val_acc":acc, 'val_precision':precision, 'val_recall':recall, 'step': self.current_epoch}
        
        return {'val_loss': avg_loss, 'val_acc':acc,'log': logs, 'progress_bar': logs}
