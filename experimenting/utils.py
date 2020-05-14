import re
import os
import hydra
from omegaconf import DictConfig, ListConfig
import albumentations
from .dataset.config import MOVEMENTS_PER_SESSION
import numpy as np
import cv2
import torch

def get_file_paths(path, extensions):
    extension_regex = "|".join(extensions)
    print(extension_regex)
    res = [os.path.join(path, f) for f in os.listdir(path) if
           re.search(r'({})$'.format(extension_regex),f)]
    return sorted(res)

def flatten(d):
    out = {}
    for key, val in d.items():
        if isinstance(val, dict) or isinstance(val, DictConfig):
            val = [val]
        if isinstance(val, list) or isinstance(val, ListConfig):
            count_element = 0
            for subdict in val:
                if isinstance(subdict, dict) or isinstance(subdict, DictConfig):
                    deeper = flatten(subdict).items()
                    out.update({key + '__' + key2: val2 for key2, val2 in
                                deeper})
                else:
                    out.update({key + '__list__' + str(count_element): subdict})
                    count_element+=1
                        
    
        else:
            out[key] = val
    return out

def unflatten(dictionary):
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split("__")
        d = resultDict     
        i = 0
        while (i < len(parts)-1):
            part = parts[i]
            is_list = False
            if part == "list":
                part = parts[i-1]
                if not isinstance(previous[part], list):
                    previous[part] = list()
                d = previous[part]
                break
                                
            if part not in d:
                d[part] = dict()
                    
            previous = d
            d = d[part]
            i+=1

        if isinstance(d, list):
            d.append(value)
        else:
            d[parts[-1]] = value
    return DictConfig(resultDict)


def get_augmentation(augmentation_specifics):
    augmentations = []


    for _, aug_spec in augmentation_specifics.apply.items():
        aug = hydra.utils.instantiate(aug_spec)

        augmentations.append(aug)

    return albumentations.Compose(augmentations)

def get_label_from_filename(filepath):
    """Given the filepath of .h5 data, return the correspondent label

    E.g.
n    S1_session_2_mov_1_frame_249.npy
    """
    
    label = 0
    filename = os.path.basename(filepath)
    session = int(filename[filename.find('session_') + len('session_')])
    mov = int(filename[filename.find('mov_') + len('mov_')])

    for i in range(1, session):
        label += MOVEMENTS_PER_SESSION[i]

    return label + mov - 1

def get_preload_dir(data_dir):
    return os.path.join(data_dir, 'preload')


def decay_heatmap(heatmap, sigma2=4):
    heatmap = cv2.GaussianBlur(heatmap,(0,0),sigma2)
    heatmap /= np.max(heatmap) # keep the max to 1
    return heatmap

def get_heatmap(vicon_xyz, p_mat, width, heigth):
    num_joints = vicon_xyz.shape[-1]
    vicon_xyz_homog = np.concatenate([vicon_xyz, np.ones([1, num_joints])], axis=0)
    coord_pix_homog = np.matmul(p_mat, vicon_xyz_homog)
    coord_pix_homog_norm = coord_pix_homog / coord_pix_homog[-1]
    
    u = coord_pix_homog_norm[0]
    v = heigth - coord_pix_homog_norm[1] # flip v coordinate to match the image direction

    # mask is used to make sure that pixel positions are in frame range.
    mask = np.ones(u.shape).astype(np.float32)
    mask[np.isnan(u)] = 0; mask[np.isnan(v)] = 0
    mask[u>width] = 0; mask[u<=0] = 0; mask[v>heigth] = 0; mask[v<=0] = 0

    # pixel coordinates
    u = u.astype(np.int32)
    v = v.astype(np.int32)


    
    return vicon_xyz, np.stack((v,u), axis=-1), mask, label_heatmaps

def _get_heatmap(joints, mask, heigth, width, num_joints):
    u, v = joints
    # initialize, fill and smooth the heatmaps
    label_heatmaps = np.zeros((heigth, width, num_joints))
    for fmidx, zipd in enumerate(zip(v, u, mask)):
        if zipd[2]==1: # write joint position only when projection within frame boundaries
            label_heatmaps[zipd[0], zipd[1], fmidx] = 1
            label_heatmaps[:,:,fmidx] = decay_heatmap(label_heatmaps[:,:,fmidx])
    return label_heatmaps


def get_joints_from_heatmap(y_pr):
    batch_size = y_pr.shape[0]
    n_joints = y_pr.shape[1]
    device = y_pr.device
    confidence = torch.zeros((batch_size, n_joints), device=device)

    p_coords_max = torch.zeros((batch_size, n_joints, 2), dtype=torch.float32, device=device)
    for b in range(batch_size):
        for j in range(n_joints):
            pred_joint = y_pr[b, j]
            max_value = torch.max(pred_joint)
            p_coords_max[b, j] = (pred_joint == max_value).nonzero()[0]
            # Confidence of the joint
            confidence[b, j] = max_value
            
    return p_coords_max, confidence
