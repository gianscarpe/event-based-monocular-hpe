import os

import numpy as np
import scipy.io

# Load in annotations
frame_root = './data/penn-crop/frames/'
label_root = './data/penn-crop/labels/'


def list_dir(path, ext):
    list_file = [f for f in os.listdir(path) if f.endswith(ext)]
    list_file.sort()
    return list_file


# list_inds is an N x 2 array where the first/second coloum is seq/fr id
# warning: list_inds is 0-based
list_inds = np.zeros((0, 2), dtype='uint16')
list_anno = []
list_seq = list_dir(label_root, '.mat')
for ind, seq in enumerate(list_seq):
    name_seq = os.path.splitext(seq)[0]
    # get seq and fr ind
    list_fr = list_dir(frame_root + name_seq + '/', '.jpg')
    s_ind = ind * np.ones((len(list_fr), 1), dtype='uint16')
    f_ind = np.arange(len(list_fr))[:, np.newaxis]
    list_inds = np.vstack((list_inds, np.hstack((s_ind, f_ind))))
    # load annotation
    anno = scipy.io.loadmat(label_root + seq)
    list_anno.append(anno)

nimages = list_inds.shape[0]

# Part info
parts = [
    'head',
    'rsho',
    'lsho',
    'relb',
    'lelb',
    'rwri',
    'lwri',
    'rhip',
    'lhip',
    'rkne',
    'lkne',
    'rank',
    'lank',
]
nparts = len(parts)


def seqind(idx):
    return list_inds[idx, 0]


def frind(idx):
    return list_inds[idx, 1]


def imgpath(idx):
    # Path to image under frame_root
    # Need to convert from zero-based to one-based
    s_ind = list_inds[idx, 0]
    f_ind = list_inds[idx, 1]
    return '{:04d}/{:06d}.jpg'.format(s_ind + 1, f_ind + 1)


def istrain(idx):
    # Return true if image is in training set
    s_ind = list_inds[idx, 0]
    return list_anno[s_ind]['train'][0, 0] == 1


def partinfo(idx):
    # Part location and visibility
    s_ind = list_inds[idx, 0]
    f_ind = list_inds[idx, 1]
    coords = np.zeros((13, 2))
    vis = np.zeros(13)
    for p_ind in xrange(13):
        if list_anno[s_ind]['visibility'][f_ind, p_ind] == 1:
            coords[p_ind, 0] = list_anno[s_ind]['x'][f_ind, p_ind]
            coords[p_ind, 1] = list_anno[s_ind]['y'][f_ind, p_ind]
            assert coords[p_ind, 0] != 0
            assert coords[p_ind, 1] != 0
            vis[p_ind] = 1
    return coords, vis
