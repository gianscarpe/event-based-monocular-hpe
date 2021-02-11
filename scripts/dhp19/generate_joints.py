import argparse
import glob
import os
from os.path import join

import h5py
import numpy as np
from tqdm import tqdm

from experimenting import utils
from experimenting.dataset.core import DHP19Core

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate GT from DHP19')

    parser.add_argument('--homedir', type=str, help='Dataset base path')
    parser.add_argument('--outdir', type=str, help='Dataset base path')
    args = parser.parse_args()
    homedir = args.homedir
    out_dir = args.outdir

    input_dir = join(homedir, 'raw_labels')
    p_mat_dir = join(homedir, 'P_matrices/')

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
        info = DHP19Core.get_frame_info(filename)

        sub = info['subject']
        session = info['session']
        mov = info['mov']
        out_label_path = os.path.join(
            out_dir,
            "S{}_session_{}_mov_{}_frame_".format(sub, session, mov) + "{}_cam_{}_2dhm",
        )

        x_h5 = h5py.File(x_path, 'r')

        frames = x_h5['XYZ']  # JOINTS xyz
        for cam in range(4):  # iterating cams (0, 1, 2, 3)
            extrinsics_matrix, camera_matrix = utils.decompose_projection_matrix(
                p_mats[cam]
            )

            for ind in list(range(len(frames))):
                xyz = frames[ind, :]
                (
                    joints_3d_projected_onto_cam,
                    joints_2d,
                    mask,
                ) = utils.get_heatmaps_steps(xyz, p_mats[cam], width, height)

                out_filename = out_label_path.format(ind, cam)
                out_path = os.path.join(out_dir, out_filename)

                np.savez(
                    out_path,
                    joints=joints_2d,
                    mask=mask,
                    skeleton=joints_3d_projected_onto_cam,
                    xyz=xyz,
                    M=extrinsics_matrix,
                    camera=camera_matrix,
                )
