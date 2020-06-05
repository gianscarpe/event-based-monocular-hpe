import argparse
import os

import numpy as np
from omegaconf import DictConfig

from ..utils import get_file_paths, get_frame_info


def get_dataset_params(hparams_dataset):
    split_at = hparams_dataset.split_at
    save_split = hparams_dataset.save_split

    if hparams_dataset.load_split:
        params = load_npy_indexes_and_map(hparams_dataset.preload_dir)
    else:
        file_paths = _get_file_paths_with_cam(hparams_dataset.data_dir,
                                              hparams_dataset.cams)

        data_index, test_index = _get_train_test_split(
            file_paths, hparams_dataset.test_subjects)
        train_index, val_index = _split_set(data_index, split_at=split_at)

        params = {
            'file_paths': file_paths,
            'train_indexes': train_index,
            'val_indexes': val_index,
            'test_indexes': test_index
        }

        if save_split:
            _save_params(hparams_dataset.preload_dir, **params)

    return params


def _get_train_test_split(file_paths, subjects=None):
    if subjects is None:
        subjects = [1, 8, 14, 17]

    data_indexes = np.arange(len(file_paths))
    test_subject_indexes_mask = [
        get_frame_info(x)['subject'] in subjects for x in file_paths
    ]
    test_index = data_indexes[test_subject_indexes_mask]
    train_index = data_indexes[~np.in1d(data_indexes, test_index)]

    return train_index, test_index


def _split_set(data_indexes, split_at=0.8):
    np.random.shuffle(data_indexes)
    n_data_for_training = len(data_indexes)
    train_split = int(split_at * n_data_for_training)
    train_indexes = data_indexes[:train_split]
    val_indexes = data_indexes[train_split:]

    return train_indexes, val_indexes


def _save_params(out_dir, file_paths, train_indexes, val_indexes,
                 test_indexes):
    print("Saving split ...")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "file_paths.npy"), file_paths)
    np.save(os.path.join(out_dir, "train_indexes.npy"), train_indexes)
    np.save(os.path.join(out_dir, "val_indexes.npy"), val_indexes)
    np.save(os.path.join(out_dir, "test_indexes.npy"), test_indexes)


def load_npy_indexes_and_map(path):

    train_index = np.load(os.path.join(path, "train_indexes.npy"))
    val_index = np.load(os.path.join(path, "val_indexes.npy"))
    test_index = np.load(os.path.join(path, "test_indexes.npy"))
    file_paths = np.load(os.path.join(path, "file_paths.npy"))

    print(f"LOADED INDEXES! train: {len(train_index)} \t val: " +
          f"{len(val_index)} \t test: {len(test_index)}")

    params = {
        'file_paths': file_paths,
        'train_indexes': train_index,
        'val_indexes': val_index,
        'test_indexes': test_index
    }

    return params


def _get_file_paths_with_cam(data_dir, cams=None):
    if cams is None:
        cams = [3]

    file_paths = np.array(get_file_paths(data_dir, extensions=['.npy',
                                                               '.mat']))
    cam_mask = [get_frame_info(x)['cam'] in cams for x in file_paths]

    file_paths = file_paths[cam_mask]

    return file_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate train, test and val indexes')
    parser.add_argument('--data_dir',
                        type=str,
                        help='path to root dataset directory')
    parser.add_argument('--labels_dir',
                        type=str,
                        help='path to root dataset directory')
    parser.add_argument('--split', type=float, default=.8, help='Split at %')
    parser.add_argument('--preload_dir',
                        type=str,
                        help='path to root dataset directory')
    parser.add_argument('--cams',
                        type=int,
                        default=[1, 2, 3, 4],
                        nargs='+',
                        help='Cams')
    parser.add_argument('--test_subjects',
                        type=int,
                        default=[1, 2, 3, 4, 5],
                        nargs='+',
                        help='Test subjects')

    args = parser.parse_args()

    hparams = DictConfig({
        'data_dir': args.data_dir,
        'save_split': True,
        'labels_dir': args.labels_dir,
        'preload_dir': args.preload_dir,
        'test_subjects': args.test_subjects,
        'cams': args.cams
    })

    PATH = args.dataset_path
    SPLIT_AT = args.split

    print("DONE!")
