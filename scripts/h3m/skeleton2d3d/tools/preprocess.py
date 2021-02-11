#!/usr/bin/env python

import os
import sys

import h5py
import numpy as np
import penn


def read_file_lines(file):
    lines = [line.strip() for line in open(file)]
    return lines


def write_file_h5(file, dict, attrs):
    with h5py.File(file, 'w') as f:
        f.attrs['name'] = attrs
        for k in dict.keys():
            f[k] = np.array(dict[k])


# read valid ind (one-based)
vind_file = './data/penn-crop/valid_ind.txt'
valid_ind = read_file_lines(vind_file)
valid_ind = [int(x) for x in valid_ind]

# init variables
keys = ['ind2sub', 'part']
annot_tr = {k: [] for k in keys}
annot_vl = {k: [] for k in keys}
annot_ts = {k: [] for k in keys}

# set outputs
op_dir = './data/penn-crop/'
tr_h5 = op_dir + 'train.h5'
vl_h5 = op_dir + 'val.h5'
ts_h5 = op_dir + 'test.h5'

for idx in xrange(penn.nimages):
    # Part annotations and visibility
    coords, vis = penn.partinfo(idx)
    # skip if no visible joints
    # 1. all joints
    # if np.all(vis == False): continue
    # 2. difficult joints, i.e. idx 3 to 12
    if np.all(vis[3:] == False):
        continue

    # Check train/valid/test association
    if penn.istrain(idx):
        if penn.seqind(idx) + 1 not in valid_ind:
            annot = annot_tr
        else:
            annot = annot_vl
    else:
        annot = annot_ts

    # Get seq and fr id: use one-based index
    s_ind = penn.seqind(idx) + 1
    f_ind = penn.frind(idx) + 1

    # Add info to annotation list
    annot['ind2sub'] += [[s_ind, f_ind]]
    annot['part'] += [coords]

    # Show progress
    print "images\rprocessed", idx + 1,
    sys.stdout.flush()

print ""

# generate h5 files
if not os.path.isfile(tr_h5):
    write_file_h5(tr_h5, annot_tr, 'penn-crop')
if not os.path.isfile(vl_h5):
    write_file_h5(vl_h5, annot_vl, 'penn-crop')
if not os.path.isfile(ts_h5):
    write_file_h5(ts_h5, annot_ts, 'penn-crop')

print "done."
