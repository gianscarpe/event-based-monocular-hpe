from matplotlib import pyplot as plt
import h5py
import cv2
import os
from config import CAM, MAX_WIDTH
import glob
import numpy as np
root_dir = "/home/gianscarpe/dev/data/dataset/h5_dataset_7500_events/346x260/"
out_dir = "/home/gianscarpe/dev/data/dataset/h5_dataset_7500_events/movements_per_frame/"
#example S12_session3_mov6_7500events_voxel.h5
x_paths = sorted(glob.glob(os.path.join(root_dir, "*voxel.h5")))

n_files = len(x_paths)

for x_path in x_paths:
    filename = os.path.basename(x_path)
    sub = filename[filename.find('S') + 1 : filename.find('S') + 4].split('_')[0]
    session = int(filename[filename.find('session') + len('session')])
    mov = int(filename[filename.find('mov') + len('mov')])

    frame_path = os.path.join(out_dir,
                              "S{}_session_{}_mov_{}_frame_".format(sub, session, mov)  + "{}.npy")
 
    if not os.path.exists(frame_path.format(1)):
        x_h5 = h5py.File(x_path, 'r')
        frames = x_h5['DVS'][:, CAM, :MAX_WIDTH, :, :].swapaxes(2, 1)
        for ind in list(range(len(frames))):
            frame = frames[ind]
            print(frame_path.format(ind))
            out_path = os.path.join(out_dir, frame_path.format(ind))
            np.save(out_path, frame)
            



