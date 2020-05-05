import cv2
import os
from os.path import join, basename
import glob
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.io
import numpy as np
import torch
from ..utils import get_label_from_filename, decay_heatmap
from .config import N_JOINTS

__all__ = ['DHP19ClassificationDataset', 'DHP3DDataset']

class DHP19BaseDataset(Dataset):
    def __init__(self, file_paths, labels=None, indexes=None, transform=None, augment_label=False):
    
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sampletot += n_frames.
        """

        self.x_paths = file_paths
        self.x_indexes = indexes
        self.labels = labels 
        self.transform = transform
        self.augment_label = augment_label
            
    def __len__(self):
        return len(self.x_indexes)

    def _get_x(self, idx):        
        img_name = self.x_paths[idx]
        x = DHP19BaseDataset._load(img_name)                                
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
        y = self._get_y(idx)
        
        if self.transform:
            if self.augment_label:
                augmented = self.transform(image=x, mask=y)
                y = augmented['mask']
                y = torch.squeeze(y.transpose(0, -1))
            else:
                augmented = self.transform(image=x)
            x = augmented['image']

        return x, y
    
class DHP19ClassificationDataset(DHP19BaseDataset):
    def __init__(self, file_paths, labels=None, indexes=None, transform=None):
    
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sampletot += n_frames.
        """


        x_indexes = indexes if indexes is not None else np.arange(
            len(self.x_paths))
        labels = labels if labels is not None else [get_label_from_filename(x_path)
                                                                   for x_path in self.x_paths]
        
        super(DHP19ClassificationDataset, self).__init__(file_paths, labels,
                                                         x_indexes, transform,
                                                         False)

    def _get_y(self, idx):
        return self.labels[idx]


class DHP3DDataset(DHP19BaseDataset):
        def __init__(self, file_paths, labels_dir, indexes=None, transform=None,
                     n_joints=N_JOINTS):

            labels = DHP3DDataset._retrieve_2hm_files(file_paths=file_paths,
                                                      labels_dir=labels_dir)

            super(DHP3DDataset, self).__init__(file_paths, labels, indexes,
                                               transform, True)

            self.n_joints = n_joints
            self.augment_label = True

        def _retrieve_2hm_files(labels_dir, file_paths):
            labels_hm = [join(labels_dir, basename(x).split('.')[0] + '_2dhm.npy') for x in file_paths]
            return labels_hm

        def _get_y(self, idx):
            joints_file = self.labels[idx]
            joints = np.load(joints_file)
            h, w = joints.shape
            y = np.zeros((h, w, self.n_joints))
            
            for joint_id in range(1, self.n_joints+1):
                heatmap = (joints == joint_id).astype('float')
                if heatmap.sum() > 0:
                    y[:, :, joint_id-1] = decay_heatmap(heatmap)
                
            return y

            
