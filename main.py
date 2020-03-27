import os
import torch
from torch import nn
import numpy as np
import torchvision
import time
import argparse
import json
import pytorch_lightning as pl

from dataset import get_dataset, get_dataloader
import models

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

class Model(pl.LightningModule):
    def __init__(self, config, preload=False, num_workers=6):
        super(Model, self).__init__()
        self.num_workers = num_workers
        self.preload = preload

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
            from utils.generate_indexes import get_npy_indexes_and_map
            file_paths, train_index, val_index, test_index = save_npy_indexes_and_map(data_dir)

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
        return optimizer

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
        correct_predictions = (pred_y == b_y.to(device)).sum().item()
        num_predictions = pred_y.size()

        return {"val_loss":loss, "correct_predictions":correct_predictions, "n_predictions":num_predictions}

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = torch.Tensor([x['correct_predictions'] for x in outputs]).sum() / torch.Tensor([x['n_predictions'] for x in outputs]).sum() *100
        
        tensorboard_logs = {'val_loss': avg_loss, "val_acc":acc}
        return {'avg_val_loss': avg_loss, 'val_acc':acc,'log': tensorboard_logs}

       
            
                
def train_val_model(train_loader, val_loader, cnn, train_config):
    # optimize all cnn parameters
    # the target label is not one-hotted


    print("--- Start training ---")
    STAMP_EVERY = int(len(train_loader) * .1)
    for epoch in range(train_config['epochs']):
        start_time = time.time()
        accumulated_loss = 0
        mean_loss = 0
        cnn.train()
        # gives batch data, normalize x when iterate train_loader
        for step, (b_x, b_y) in enumerate(train_loader):        
            b_x = b_x.float()
            optimizer.zero_grad()           # clear gradients for this training step
            output = cnn(b_x.to(device))               # cnn output
            loss = loss_func(output, b_y.to(device))   # cross entropy loss
            
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # appnly gradients
            accumulated_loss += loss.item()
            mean_loss = accumulated_loss / (step+1)
            if (step % STAMP_EVERY == STAMP_EVERY-1):
                print(
                    f"Epoch: {epoch+1} - [{step/len(train_loader)*100:.2f}%] | train loss: {mean_loss:.4f}")
                
        print(f"Epoch: {epoch+1} | Evaluating")
        correct_predictions = 0
        with torch.no_grad():
            for step, (b_x, b_y) in enumerate(val_loader):
                b_x = b_x.float()
                output = cnn(b_x.to(device))
                _, pred_y = torch.max(output.data, 1)
                correct_predictions += (pred_y == b_y.to(device)).sum().item()
                
        print(f"Epoch: {epoch} | Duration: {(time.time() - start_time):.4f}s | val accuracy: [{correct_predictions}/{n_val_data}] | {correct_predictions/n_val_data * 100:.4f}%")
                
                
    torch.save(cnn, './model.h5')
        

if __name__ == '__main__':
    # TRAIN ARGUMENTS -- refer to config.json for train and data configuration
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config-path', type=str, help='Path to confg json')
    parser.add_argument('--preload', default=False, action='store_true')
    parser.add_argument('--multi-gpu', type=bool, default=False)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda:0')
        
    args = parser.parse_args()
    config_path = args.config_path
    with open(config_path) as json_file:
        config = json.load(json_file)

    data_config = config['data']
    train_config = config['train']
    
    MULTI_GPU = args.multi_gpu
    NUM_WORKERS = args.num_workers
    DEV_TYPE = args.device
    PRELOAD = args.preload
    device = torch.device(DEV_TYPE)

    # Get MODEL
    
    
    #print(cnn)  # net architecture

    if torch.cuda.device_count() > 1 and MULTI_GPU:
        print("MULTI GPU!")
        cnn = nn.DataParallel(cnn, device_ids = [0, 1])
        
    #cnn = cnn.to(device)

    # Get DATA
    #train_loader, val_loader, test_loader = get_data(data_config)

    # TRAIN and EVAL
    #train_val_model(train_loader, val_loader, cnn, train_config)

    model = Model(config, preload=PRELOAD)

    trainer = pl.Trainer(gpus=1)
    trainer.fit(model)
