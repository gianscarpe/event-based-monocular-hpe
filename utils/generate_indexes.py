import glob
import numpy as np
import argparse
import os
import pickle

PRELOAD_DIR = "preload"

def save_npy_indexes_and_map(path, split_at):
    print("Creating split ...")
    file_paths = sorted(glob.glob(os.path.join(path, "*.npy")))
    out_dir = os.path.join(path, PRELOAD_DIR)

    os.makedirs(out_dir, exist_ok=True)
    
    indexes = np.load(os.path.join(out_dir, "balanced_indexes.npy"))
    np.random.shuffle(indexes)

    n_files = len(indexes)
    
    test_split = int(split_at * n_files)
    data_indexes = indexes[:test_split]
    test_indexes = indexes[test_split:]

    n_data_for_training = len(data_indexes)
    train_split = int(split_at * n_data_for_training)
    train_indexes = data_indexes[:train_split]
    val_indexes = data_indexes[train_split:]

    with open(os.path.join(out_dir, "file_paths.pkl"), "wb") as f:
        pickle.dump(file_paths, f)

    np.save(os.path.join(out_dir, "train_indexes.npy"), train_indexes)
    np.save(os.path.join(out_dir, "val_indexes.npy"), val_indexes)
    np.save(os.path.join(out_dir, "test_indexes.npy"), test_indexes)

    return file_paths, train_indexes, val_indexes, test_indexes
    
def load_npy_indexes_and_map(path):
    path = os.path.join(path, PRELOAD_DIR)
                        
    train_indexes = np.load(os.path.join(path, "train_indexes.npy"))
    val_indexes = np.load(os.path.join(path, "val_indexes.npy"))
    test_indexes = np.load(os.path.join(path, "test_indexes.npy"))
    with open(os.path.join(path, "file_paths.pkl"), "rb") as f:
        file_paths = pickle.load(f)

    print(f"LOADED INDEXES! train: {len(train_indexes)} \t val: " +
          f"{len(val_indexes)} \t test: {len(test_indexes)}")
    return file_paths, train_indexes, val_indexes, test_indexes
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate train, test and val indexes')
    parser.add_argument('--dataset_path', type=str, help ='path to root dataset directory')
    parser.add_argument('--split', type=float, default=.8, help='Split at %')
    args = parser.parse_args()

    PATH = args.dataset_path
    SPLIT_AT = args.split

    save_npy_indexes_and_map(PATH, SPLIT_AT)
    print("DONE!")

    
