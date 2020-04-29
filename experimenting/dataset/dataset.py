import cv2
import os
import glob
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.io
import numpy as np
import torch
from ..utils import get_label_from_filename

class DHP19Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_paths, labels=None, indexes=None, transform=None, augment_label=False):
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
        self.labels = labels if labels is not None else [get_label_from_filename( x_path)
                                             for x_path in self.x_paths]

        self.transform = transform
        self.augment_label = augment_label

        
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
            if self.augment_label:
                augmented = self.transform(image=x, mask=y)
                y = augmented['mask']
            else:
                augmented = self.transform(image=x)
            x = augmented['image']

        return x, y



    
