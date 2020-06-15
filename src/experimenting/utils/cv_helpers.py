import numpy as np
import scipy
import torch

import cv2

__all__ = ['decay_heatmap', 'get_heatmaps', 'get_joints_from_heatmap']


def decay_heatmap(heatmap, sigma2=4):
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma2)
    heatmap /= np.max(heatmap)  # keep the max to 1
    return heatmap


def get_heatmaps(vicon_xyz, p_mat, width, height):
    K, R, t = decompose_projection_matrix(p_mat)
    M = np.concatenate([R, np.expand_dims(t, 1)], axis=1)
    camera = np.concatenate([K, np.zeros((3, 1))], axis=1)

    num_joints = vicon_xyz.shape[-1]
    vicon_xyz_homog = np.concatenate(
        [vicon_xyz, np.ones([1, num_joints])], axis=0)
    coord_pix_homog = np.matmul(p_mat, vicon_xyz_homog)
    coord_pix_homog_norm = coord_pix_homog / coord_pix_homog[-1]

    u = coord_pix_homog_norm[0]
    v = height - coord_pix_homog_norm[
        1]  # flip v coordinate to match the image direction

    # mask is used to make sure that pixel positions are in frame range.
    mask = np.ones(u.shape).astype(np.float32)
    mask[np.isnan(u)] = 0
    mask[np.isnan(v)] = 0
    mask[u > width] = 0
    mask[u <= 0] = 0
    mask[v > height] = 0
    mask[v <= 0] = 0

    # pixel coordinates
    u = u.astype(np.int32)
    v = v.astype(np.int32)
    joints = np.stack((v, u), axis=-1)
    #hms = _get_heatmap((u, v), mask, height, width, num_joints)

    # Get xyz w.r.t. camera coord system
    xyz_cam = M.dot(vicon_xyz_homog)
    # Note: cam coord system is left-handed; Z is along the negative axis
    xyz_cam[2, :] = - xyz_cam[2, :]
    z_ref = xyz_cam[2, 5]
    return xyz_cam, joints, mask, camera, z_ref


def _get_heatmap(joints, mask, heigth, width, num_joints):
    u, v = joints
    # initialize, fill and smooth the heatmaps
    label_heatmaps = np.zeros((heigth, width, num_joints))
    for fmidx, zipd in enumerate(zip(v, u, mask)):
        if zipd[2] == 1:  # write joint position only when projection within frame boundaries
            label_heatmaps[zipd[0], zipd[1], fmidx] = 1
            label_heatmaps[:, :, fmidx] = decay_heatmap(label_heatmaps[:, :,
                                                                       fmidx])
    return label_heatmaps


def decompose_projection_matrix(P):
    Q = P[:3, :3]
    q = P[:, 3]
    U, S = scipy.linalg.qr(np.linalg.inv(Q))
    R = np.linalg.inv(U)
    K = np.linalg.inv(S)
    t = S.dot(q)
    K = K / K[2, 2]

    return K, R, t


def get_joints_from_heatmap(y_pr):
    batch_size = y_pr.shape[0]
    n_joints = y_pr.shape[1]
    device = y_pr.device
    confidence = torch.zeros((batch_size, n_joints), device=device)

    p_coords_max = torch.zeros((batch_size, n_joints, 2),
                               dtype=torch.float32,
                               device=device)
    for b in range(batch_size):
        for j in range(n_joints):
            pred_joint = y_pr[b, j]
            max_value = torch.max(pred_joint)
            p_coords_max[b, j] = (pred_joint == max_value).nonzero()[0]
            # Confidence of the joint
            confidence[b, j] = max_value

    return p_coords_max, confidence
