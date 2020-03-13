import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from config import MOVEMENTS_PER_SESSION
import numpy as np


def get_label_from_filename(filepath):
    """Given the filepath of .h5 data, return the correspondent label

    E.g.
    S13_session2_mov2_7500events.h5 -> Session 2, movement 2 -> label 10
    """

    label = 0
    filename = os.path.basename(filepath)
    session = int(filename[filename.find('session') + len('session')])
    mov = int(filename[filename.find('mov') + len('mov')])

    for i in range(session):
        label += MOVEMENTS_PER_SESSION[i]


    return label + mov


class DHP19Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sampletot += n_frames.
        """

        self.x_paths = sorted(glob.glob(os.path.join(root_dir, "*events.h5")))
        self.x_indexes = np.arange(len(self.x_paths))
        self.labels = [get_label_from_filename(x_path) for x_path in self.x_paths]

        self.transform = transform

    def __len__(self):
        return len(self.x_indexes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.x_paths[idx]
        x = np.load(img_name)
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


if __name__ == '__main__':
    root_dir = '/home/gianscarpe/dev/dhp19/data/h5_dataset_7500_events/movements_per_frame'
    dataloader = DataLoader(DHP19Dataset(root_dir), batch_size=4, shuffle=True, num_workers=4)
