from PIL import Image
import cv2
import os
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import MOVEMENTS_PER_SESSION
import numpy as np
import torch

def get_label_from_filename(filepath):
    """Given the filepath of .h5 data, return the correspondent label

    E.g.
    S1_session_2_mov_1_frame_249.npy
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

    def __init__(self, root_dir, file_paths, indexes=None, transform=None, n_channels=1, preload=False):
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
        
        if (preload):
            print("Start loading")
            self.images = np.zeros((len(self.x_indexes), 224, 224, N_CHANNELS))
            for i, idx in enumerate(self.x_indexes):
                path = self.x_paths[idx]
                x = DHP19Dataset._load_npy_and_resize(path)
                self.images[i] = x
            print("Finish loading")

        
    def __len__(self):
        return len(self.x_indexes)

    def _get_x(self, idx):
        
        if self.images is not None:
            x = self.images[idx]
        else:
            img_name = self.x_paths[self.x_indexes[idx]]
            x = DHP19Dataset._load_npy_and_resize(img_name)
                                        
        return x

    def _load_npy_and_resize(path):
        x = np.load(path).astype("float32")
        x = cv2.resize(x, (224, 224))
        return x

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self._get_x(idx)
        if self.n_channels == 3:
            x = np.repeat(x[:, :,  np.newaxis], 3, axis=-1)
            x = Image.fromarray(x, 'RGB')
        elif self.n_channels == 1:
            x = Image.fromarray(x, 'L')
        
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)

        return x, y


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

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.016, 0.025, 0.015,0.013), (0.188, 0.253, 0.186, 0.173))
    ])
    
    train_loader = DataLoader(DHP19Dataset(
        data_dir, file_paths, train_index, transform=preprocess,preload=preload, n_channels=n_channels), batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(DHP19Dataset(
        data_dir, file_paths,val_index, transform=preprocess, preload=preload,n_channels=n_channels), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_loader = DataLoader(DHP19Dataset(
        data_dir, file_paths,test_index, transform=preprocess, preload=preload,n_channels=n_channels), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


    
