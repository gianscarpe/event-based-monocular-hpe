import os
import torch
from torch import nn
import numpy as np
import torchvision
import time
import argparse
import json
import pytorch_lightning as pl

from dataset import get_data
import models

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def train_val_model(train_loader, val_loader, cnn, train_config):
    # optimize all cnn parameters
    optimizer = getattr(torch.optim,
                        train_config['optimizer']['type'])(params=cnn.parameters(),
                                                           **train_config['optimizer']['params'])
    # the target label is not one-hotted
    loss_func = getattr(nn, train_config['loss'])()

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
    parser.add_argument('--preload', type=bool, default=False)
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

    cnn = getattr(models, train_config['model'])(data_config['n_channels'], data_config['n_classes'])
    print(cnn)  # net architecture

    if torch.cuda.device_count() > 1 and MULTI_GPU:
        print("MULTI GPU!")
        cnn = nn.DataParallel(cnn, device_ids = [0, 1])
        
    cnn = cnn.to(device)

    # Get DATA
    train_loader, val_loader, test_loader = get_data(data_config)

    # TRAIN and EVAL
    train_val_model(train_loader, val_loader, cnn, train_config)
