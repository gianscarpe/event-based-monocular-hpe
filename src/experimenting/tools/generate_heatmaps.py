import glob
import os
from os.path import join

import h5py
import numpy as np
from tqdm import tqdm

from experimenting.utils import get_heatmap

if __name__ == '__main__':
    homedir = '/home/gianscarpe/dev/data/dhp19'
    dataset = 'time_count_dataset'
    input_dir = join(homedir, 'labels')
    events_dir = join(homedir, dataset, '346x260')
    p_mat_dir = join(homedir, 'P_matrices/')
    out_dir = join('/data/dhp19/', dataset, 'labels_joints')
    os.makedirs(out_dir, exist_ok=True)
    width = 344
    height = 260

    p_mat_cam1 = np.load(join(p_mat_dir, 'P1.npy'))
    p_mat_cam2 = np.load(join(p_mat_dir, 'P2.npy'))
    p_mat_cam3 = np.load(join(p_mat_dir, 'P3.npy'))
    p_mat_cam4 = np.load(join(p_mat_dir, 'P4.npy'))
    p_mats = [p_mat_cam4, p_mat_cam1, p_mat_cam3, p_mat_cam2]

    x_paths = sorted(glob.glob(join(input_dir, "*.h5")))

    n_files = len(x_paths)
    print(f"N of files: {n_files}")

    for x_path in tqdm(x_paths):
        filename = os.path.basename(x_path)
        sub = filename[filename.find('S') + 1:filename.find('S') +
                       4].split('_')[0]
        session = int(filename[filename.find('session') + len('session')])
        mov = int(filename[filename.find('mov') + len('mov')])

        frame_path = os.path.join(
            out_dir, "S{}_session_{}_mov_{}_frame_".format(sub, session, mov) +
            "{}_cam_{}_2dhm")
        x_h5 = h5py.File(x_path, 'r')

        frames = x_h5['XYZ']
        for cam in range(4):
            for ind in list(range(len(frames))):

                xyz, joints, mask, _ = get_heatmap(frames[ind, :], p_mats[cam],
                                                   width, height)
                out_filename = frame_path.format(ind, cam)
                out_path = os.path.join(out_dir, out_filename)

                np.savez(out_path, joints=joints, mask=mask)
