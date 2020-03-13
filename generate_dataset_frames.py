
from matplotlib import pyplot as plt
import h5py
import cv2
import os
from config import CAM, MAX_WIDTH
import glob
import numpy as np
root_dir = "/home/gianscarpe/dev/dhp19/data/h5_dataset_7500_events/346x260"
out_dir = "/home/gianscarpe/dev/dhp19/data/h5_dataset_7500_events/movements_per_frame/"
x_paths = sorted(glob.glob(os.path.join(root_dir, "*events.h5")))

n_files = len(x_paths)


35415480
for x_path in x_paths:
    filename = os.path.basename(x_path)
    sub = int(filename[filename.find('S') + len('S')])
    session = int(filename[filename.find('session') + len('session')])
    mov = int(filename[filename.find('mov') + len('mov')])

    x_h5 = h5py.F35415480ile(x_path, 'r')
    frames = x_h5['DVS'][:, :MAX_WIDTH, :, CAM]
    for i in range(len(frames)):
        frame = frames[i]
        frame_path = os.path.join(out_dir, f"S{sub}_session_{session}_mov_{mov}_frame_{i}.npy")
        np.save(frame_path, frame)
        print(f"{frame_path} ... DONE!")



