import sys
sys.path.insert(0, "../")

import h5py
import os
from experimenting.dataset.config import CAM, MAX_WIDTH
import glob
import numpy as np
root_dir = "/home/gianscarpe/dev/data/dhp19/time_count_dataset/346x260/"
out_dir = "/home/gianscarpe/dev/data/dhp19/time_count_dataset/movements_per_frame/"
#example S12_session3_mov6_7500events_voxel.h5

x_paths = sorted(glob.glob(os.path.join(root_dir, "*events.h5")))

n_files = len(x_paths)

for x_path in x_paths:
        filename = os.path.basename(x_path)
        sub = filename[filename.find('S') + 1 : filename.find('S') + 4].split('_')[0]
        session = int(filename[filename.find('session') + len('session')])
        mov = int(filename[filename.find('mov') + len('mov')])

        if sub not in ['1', '8', '15']:
                continue
        frame_path = os.path.join(out_dir,
                                  "S{}_session_{}_mov_{}_frame_".format(sub, session, mov)  + "{}_cam_{}.npy")
        x_h5 = h5py.File(x_path, 'r')
        for cam in range(4):
                frames = x_h5['DVS'][..., cam]
                for ind in list(range(len(frames))):
                    frame = frames[ind, :, :]
                    print(frame_path.format(ind, cam))
                    out_path = os.path.join(out_dir, frame_path.format(ind, cam))
                    np.save(out_path, frame)


            



