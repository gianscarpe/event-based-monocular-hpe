import numpy as np
import argparse
import os
import pickle
import re
from ..utils import get_file_paths, get_preload_dir, get_label_from_filename

def save_npy_indexes_and_map(data_dir, labels_dir, split_at=0.8, balanced=False,
                             cams=None):
    print("Creating split ...")
    if cams is None:
        cams = ["cam_2", "cam_3"]

    file_paths = get_file_paths(data_dir, extensions=['.npy','.mat'])
    file_paths = [x for x in file_paths if sum([cam in x for cam in cams]) > 0]
    out_dir = get_preload_dir(labels_dir)
    os.makedirs(out_dir, exist_ok=True)

    if balanced:
        indexes = np.load(os.path.join(out_dir, "balanced_indexes.npy"))
    else:
        indexes = np.arange(len(file_paths))

    np.random.shuffle(indexes)

    n_files = len(indexes)
    
    test_split = int(split_at * n_files)
    data_indexes = indexes[:test_split]
    test_indexes = indexes[test_split:]

    n_data_for_training = len(data_indexes)
    train_split = int(split_at * n_data_for_training)
    train_indexes = data_indexes[:train_split]
    val_indexes = data_indexes[train_split:]

    labels = [get_label_from_filename(x) for x in file_paths]    

                                                 
    with open(os.path.join(out_dir, "file_paths.pkl"), "wb") as f:
        pickle.dump(file_paths, f)

    np.save(os.path.join(out_dir, "train_indexes.npy"), train_indexes)
    np.save(os.path.join(out_dir, "val_indexes.npy"), val_indexes)
    np.save(os.path.join(out_dir, "test_indexes.npy"), test_indexes)
    np.save(os.path.join(out_dir, "labels.npy"), labels)

    return file_paths, train_indexes, val_indexes, test_indexes, labels
    
def load_npy_indexes_and_map(path):
                        
    train_indexes = np.load(os.path.join(path, "train_indexes.npy"))
    val_indexes = np.load(os.path.join(path, "val_indexes.npy"))
    test_indexes = np.load(os.path.join(path, "test_indexes.npy"))
    labels = np.load(os.path.join(path, "labels.npy"))
    
    with open(os.path.join(path, "file_paths.pkl"), "rb") as f:
        file_paths = pickle.load(f)

    print(f"LOADED INDEXES! train: {len(train_indexes)} \t val: " +
          f"{len(val_indexes)} \t test: {len(test_indexes)}")
    return file_paths, train_indexes, val_indexes, test_indexes, labels


def get_dataset_params(hparams_dataset):
    preload_dir = get_preload_dir(hparams_dataset.labels_dir)

    if os.path.exists(preload_dir):
        return load_npy_indexes_and_map(preload_dir)
    else:
        return save_npy_indexes_and_map(hparams_dataset.path, hparams_dataset.labels_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate train, test and val indexes')
    parser.add_argument('--dataset_path', type=str, help ='path to root dataset directory')
    parser.add_argument('--split', type=float, default=.8, help='Split at %')
    args = parser.parse_args()

    PATH = args.dataset_path
    SPLIT_AT = args.split

    save_npy_indexes_and_map(PATH, SPLIT_AT)
    print("DONE!")

    
