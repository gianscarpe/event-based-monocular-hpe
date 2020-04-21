from PIL import Image
import cv2
import os
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import scipy.io
from utils.config import MOVEMENTS_PER_SESSION
import numpy as np
import torch

def get_label_from_filename(filepath):
    """Given the filepath of .h5 data, return the correspondent label

    E.g.
n    S1_session_2_mov_1_frame_249.npy
    """
    
    label = 0
    filename = os.path.basename(filepath)
    session = int(filename[filename.find('session_') + len('session_')])
    mov = int(filename[filename.find('mov_') + len('mov_')])

    for i in range(1, session):
        label += MOVEMENTS_PER_SESSION[i]

    return label + mov - 1


class DHP19Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_paths, indexes=None, transform=None, n_channels=1, preload=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sampletot += n_frames.
        """

        self.x_paths = file_paths
        self.x_indexes = indexes if indexes is not None else np.arange(
            len(self.x_paths))
        self.labels = [get_label_from_filename(
            x_path) for x_path in self.x_paths]

        self.n_channels = n_channels
        self.transform = transform
        self.images = None
        
    def __len__(self):
        return len(self.x_indexes)

    def _get_x(self, idx):        
        img_name = self.x_paths[idx]
        x = DHP19Dataset._load(img_name)                                
        return x

    def _load(path):
        ext = os.path.splitext(path)[1]
        if ext == '.mat':
            x = np.swapaxes(scipy.io.loadmat(path)['V3n'], 0, 1)
        elif ext == '.npy' :
            x = np.load(path) / 255.
            if len(x.shape) == 2:
                x = np.expand_dims(x, -1)
        return x

    def __getitem__(self, idx):
        idx = self.x_indexes[idx]
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self._get_x(idx)
        y = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=x)
            x = augmented['image']


        return x, y


def get_loader(file_paths, index, preprocess, preload, n_channels):
    batch_size = data_config['batch_size']
    n_channels = data_config['n_channels']

    data_dir = os.path.join(root_dir,'movements_per_frame')
    preload_dir = os.path.join(root_dir, 'preload')

    loader = DataLoader(DHP19Dataset(file_paths, train_index,
                                     transform=preprocess,preload=preload, n_channels=n_channels),
                        batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return loader

    
def get_dataset(file_paths, index,  preload, n_channels, preprocess):
    dataset = DHP19Dataset(file_paths, indexes=index, transform=preprocess, preload=preload, n_channels=n_channels)
    return dataset

def get_dataloader(dataset, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader

def get_dataloader_from_row(file_paths, index,  preload, n_channels, batch_size, num_workers):
    loader = DataLoader(get_dataset(file_paths, index,  preload, n_channels), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader


def get_data(data_config, num_workers=12, preload=False):
    root_dir = data_config['root_dir']
    batch_size = data_config['batch_size']
    n_channels = data_config['n_channels']

    data_dir = os.path.join(root_dir,'movements_per_frame')
    preload_dir = os.path.join(root_dir, 'preload')

    if os.path.exists(preload_dir):
        from utils.generate_indexes import load_npy_indexes_and_map
        file_paths, train_index, val_index, test_index = load_npy_indexes_and_map(data_dir)
    else:
        from utils.generate_indexes import get_npy_indexes_and_map
        file_paths, train_index, val_index, test_index = save_npy_indexes_and_map(data_dir)
        
    train_loader = get_dataloader(file_paths, train_index, preload, n_channels, batch_size, num_workers)
    val_loader = get_dataloader(file_paths, val_index, preload, n_channels, batch_size, num_workers)
    test_loader = get_dataloader(file_paths, test_index, preload, n_channels, batch_size, num_workers)


    return train_loader, val_loader, test_loader


    
