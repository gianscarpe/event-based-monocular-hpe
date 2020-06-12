"""
Utilities for DHP19 dataset

Gianluca Scarpellini - gianluca.scarpellini@iit.it
"""
import os

import numpy as np
import scipy

from .cv_helpers import decay_heatmap

__all__ = ['get_frame_info', 'load_frame', 'load_heatmap',
           'get_label_from_filename', 'MOVEMENTS_PER_SESSION', 'MAX_CAM_WIDTH',
           'MAX_CAM_HEIGHT', 'MAX_X', 'MAX_Y', 'MAX_Z', 'N_JOINTS']


def get_frame_info(filename):
    filename = os.path.splitext(os.path.basename(filename))[0]

    result = {'subject': int(filename[filename.find('S') + 1 :
                                      filename.find('S') + 4].split('_')[0]),
              'session': _get_info_from_string(filename,
                                               'session'), 'mov':
              _get_info_from_string(filename, 'mov'),
              'cam': _get_info_from_string(filename,
                                           'cam'), 'frame':
              _get_info_from_string(filename, 'frame') }
    
    return result


def _get_info_from_string(filename, info, split_symbol='_'):
    return int(filename[filename.find(info) :].split(split_symbol)[1])


def load_frame(path):
    ext = os.path.splitext(path)[1]
    if ext == '.mat':
        x = np.swapaxes(scipy.io.loadmat(path)['V3n'], 0, 1)
    elif ext == '.npy' :
        x = np.load(path) / 255.
        if len(x.shape) == 2:
            x = np.expand_dims(x, -1)
    return x


def get_label_from_filename(self, filepath, movements_per_session):
    """Given the filepath, return the
correspondent label
    
        E.g.  n S1_session_2_mov_1_frame_249_cam_2.npy 
    """

    label = 0
    info = get_frame_info(filepath)

    for i in range(1, info['session']):
        label += movements_per_session[i]

    return label + info['mov'] - 1


def load_heatmap(path, n_joints):
    joints = np.load(path)
    h, w = joints.shape
    y = np.zeros((h, w, n_joints))

    for joint_id in range(1, n_joints+1):
        heatmap = (joints == joint_id).astype('float')
        if heatmap.sum() > 0:
            y[:, :, joint_id-1] = decay_heatmap(heatmap)

    return y


MOVEMENTS_PER_SESSION = {
    1 : 8,
    2 : 6,
    3 : 6,
    4 : 6,
    5 : 7
}
MAX_CAM_WIDTH = 344
MAX_CAM_HEIGHT = 300

MAX_X = 867.40
MAX_Y = 959.81
MAX_Z = 2238.23


N_JOINTS = 13
