import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2
from pose3d_utils.camera import CameraIntrinsics
from pose3d_utils.skeleton_normaliser import SkeletonNormaliser

from .dsntnn import dsnt

__all__ = [
    'decay_heatmap', 'get_heatmaps_steps', 'get_joints_from_heatmap', 'predict_xyz',
    'plot_skeleton_2d', 'plot_skeleton_3d', 'plot_heatmap',
    'decompose_projection_matrix', 'denormalize_predict', 'reproject_skeleton', 'plot_3d'
]

_normalizer = SkeletonNormaliser()


def decay_heatmap(heatmap, sigma2=10):
    """

    Parameters
    ----------
    heatmap :
       WxH matrix to decay
    sigma2 :
         (Default value = 1)

    Returns
    -------
    Heatmap obtained by gaussian-blurring the input
    """
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma2)
    heatmap /= np.max(heatmap)  # keep the max to 1
    return heatmap


def _project_xyz_onto_image(xyz, p_mat, width, height):
    """

    Parameters
    ----------
    xyz :
        xyz in world coordinate system
    p_mat :
        projection matrix word2cam_plane
    width :
        width of resulting frame
    height :
        height of resulting frame

    Returns
    -------
    u, v coordinate of skeleton joints as well as joints mask
    """
    num_joints = xyz.shape[-1]
    xyz_homog = np.concatenate([xyz, np.ones([1, num_joints])], axis=0)
    coord_pix_homog = np.matmul(p_mat, xyz_homog)
    coord_pix_homog_norm = coord_pix_homog / coord_pix_homog[-1]

    u = coord_pix_homog_norm[0]
    # flip v coordinate to match the  image direction
    v = height - coord_pix_homog_norm[1]

    # pixel coordinates
    u = u.astype(np.int32)
    v = v.astype(np.int32)

    mask = np.ones(u.shape).astype(np.float32)
    mask[np.isnan(u)] = 0
    mask[np.isnan(v)] = 0
    mask[u > width] = 0
    mask[u <= 0] = 0
    mask[v > height] = 0
    mask[v <= 0] = 0

    return u, v, mask


def _project_xyz_onto_camera_coord(xyz, M):
    """

    Parameters
    ----------
    xyz :
        xyz coordinates as 3xNUM_JOINTS wrt world coord
    M :
        word2cam EXTRINSIC matrix

    Returns
    -------
    xyz coordinates projected onto cam coordinates system
    """
    num_joints = xyz.shape[-1]
    xyz_homog = np.concatenate([xyz, np.ones([1, num_joints])], axis=0)
    # Get xyz w.r.t. camera coord system
    xyz_cam = M.dot(xyz_homog)
    # Note: cam coord system is left-handed; Z is along the negative axis
    xyz_cam[2, :] = -xyz_cam[2, :]
    return xyz_cam


def get_heatmaps_steps(xyz, p_mat, width, height):
    """

    Parameters
    ----------
    xyz :
        xyz coordinates as 3XNUM_JOINTS wrt world coord system
    p_mat :
        projection matrix from world to image plane
    width :
        width of the resulting frame
    height :
        height of the resulting frame

    Returns
    -------
    xyz wrf image coord system, uv image points of skeleton's joints, uv mask,
    """
    M, K = decompose_projection_matrix(p_mat)

    u, v, mask = _project_xyz_onto_image(xyz, p_mat, height, width)
    joints = np.stack((v, u), axis=-1)
    num_joints = len(joints)
    #hms = get_heatmap((u, v), mask, height, width, num_joints)
    xyz_cam = _project_xyz_onto_camera_coord(xyz, M)

    return xyz_cam, joints, mask#, hms


def get_heatmap(joints, mask, heigth, width, num_joints=13):
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
    """
    QR decomposition of world2imageplane projection matrix
    Parameters
    ----------
    P :
        Projection matrix word 2 image plane

    Returns
    -------
    M matrix, camera matrix
    """
    Q = P[:3, :3]
    q = P[:, 3]
    U, S = scipy.linalg.qr(np.linalg.inv(Q))
    R = np.linalg.inv(U)
    K = np.linalg.inv(S)
    t = S.dot(q)
    K = K / K[2, 2]

    M = np.concatenate([R, np.expand_dims(t, 1)], axis=1)
    camera = np.concatenate([K, np.zeros((3, 1))], axis=1)

    return M, camera


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


def predict_xyz(out):
    """
    Derivable extraction of xyz normalized predicton from heatmaps

    Parameters
    ----------
    out :
        Heatmap as torch tensor BxNUM_JOINTSxWxH

    Returns
    -------
    normalized xyz (BxNUM_JOINTSx3)
    """
    xy_hm, zy_hm, xz_hm = out

    xy = dsnt(xy_hm)
    zy = dsnt(zy_hm)
    xz = dsnt(xz_hm)
    x, y = xy.split(1, -1)
    z = 0.5 * (zy[:, :, 0:1] + xz[:, :, 1:2])

    return torch.cat([x, y, z], -1)


def _skeleton_z_ref(skeleton):
    return skeleton[0, 2] - skeleton[-1, 2]


def denormalize_predict(pred, height, width, camera):
    """

    Parameters
    ----------
    pred :
        joints coordinates as 3xNUM_JOINTS
    height :
        height of frame
    width :
        width of frame
    camera :
        intrinsics parameters

    Returns
    -------
    Denormalized skeleton joints as NUM_JOINTSx3
    """

    # skeleton
    homog = torch.cat([pred, torch.ones((13, 1), device=pred.device, dtype=pred.dtype)], axis=-1)
    camera = CameraIntrinsics(camera)
    z_ref = _normalizer.infer_depth(homog, _skeleton_z_ref, camera, height,                                    width)
    pred_skeleton = _normalizer.denormalise_skeleton(homog, z_ref, camera,
                                                     height, width)
    pred_skeleton = pred_skeleton.narrow(-1, 0, 3).transpose(0, 1)
    return pred_skeleton


def reproject_skeleton(M, joints, inv=-1):
    """

    Parameters
    ----------
    M :
        World to camera projection matrix
    joints :
        Skeleton joints as 3xNUM_JOINTS
    inv :
         Inverse y direction

    Returns
    -------
    Skeleton joints reprojected in world coord system
    """
    j = joints.copy()
    j[2, :] *= inv

    gt = np.matmul(np.linalg.pinv(M), j)
    gt = gt / gt[3, :]
    gt = gt[:3, :].swapaxes(0, 1)
    return gt


def plot_heatmap(img):
    fig, ax = plt.subplots(ncols=img.shape[0], nrows=1, figsize=(20, 20))
    for i in range(img.shape[0]):
        ax[i].imshow(img[i])
        ax[i].axis('off')
        
    plt.show()


def plot_skeleton_2d(dvs_frame, sample_gt, sample_pred):
    """
    To plot image and 2D ground truth and prediction 
    Parameters
    ----------
    dvs_frame :
        
    sample_gt :
        
    sample_pred :
        

    Returns
    -------

    """
    
    plt.figure()
    plt.imshow(dvs_frame, cmap='gray')
    plt.plot(sample_gt[:, 1], sample_gt[:, 0], '.', c='red', label='gt')
    plt.plot(sample_pred[:, 1], sample_pred[:, 0], '.', c='blue', label='pred')
    plt.legend()


def plot_skeleton_3d(gt, pred, M):
    """

    Parameters
    ----------
    gt :
        
    pred :
        
    M :
        

    Returns
    -------

    """
    gt = reproject_skeleton(M, gt, -1)
    pred = reproject_skeleton(M, pred, -1)
    fs = 5
    fig = plt.figure(figsize=(fs, fs))

    ax = Axes3D(fig)
    plot_3d(ax, gt, c='red')
    plot_3d(ax, pred, c='blue')


def _get_skeleton_lines(x, y, z):
    # rename joints to identify name and axis
    x_head, x_shoulderR, x_shoulderL, x_elbowR = x[0], x[1], x[2], x[3]
    x_elbowL, x_hipR, x_hipL = x[4], x[5], x[6],
    x_handR, x_handL, x_kneeR = x[7], x[8], x[9],
    x_kneeL, x_footR, x_footL = x[10], x[11], x[12]

    y_head, y_shoulderR, y_shoulderL, y_elbowR = y[0], y[1], y[2], y[3]
    y_elbowL, y_hipR, y_hipL = y[4], y[5], y[6],
    y_handR, y_handL, y_kneeR = y[7], y[8], y[9],
    y_kneeL, y_footR, y_footL = y[10], y[11], y[12]

    z_head, z_shoulderR, z_shoulderL, z_elbowR = z[0], z[1], z[2], z[3]
    z_elbowL, z_hipR, z_hipL = z[4], z[5], z[6],
    z_handR, z_handL, z_kneeR = z[7], z[8], z[9],
    z_kneeL, z_footR, z_footL = z[10], z[11], z[12]

    # definition of the lines of the skeleton graph
    skeleton = np.zeros((14, 3, 2))
    skeleton[0, :, :] = [[x_head, x_shoulderR], [y_head, y_shoulderR],
                         [z_head, z_shoulderR]]
    skeleton[1, :, :] = [[x_head, x_shoulderL], [y_head, y_shoulderL],
                         [z_head, z_shoulderL]]
    skeleton[2, :, :] = [[x_elbowR, x_shoulderR], [y_elbowR, y_shoulderR],
                         [z_elbowR, z_shoulderR]]
    skeleton[3, :, :] = [[x_elbowL, x_shoulderL], [y_elbowL, y_shoulderL],
                         [z_elbowL, z_shoulderL]]
    skeleton[4, :, :] = [[x_elbowR, x_handR], [y_elbowR, y_handR],
                         [z_elbowR, z_handR]]
    skeleton[5, :, :] = [[x_elbowL, x_handL], [y_elbowL, y_handL],
                         [z_elbowL, z_handL]]
    skeleton[6, :, :] = [[x_hipR, x_shoulderR], [y_hipR, y_shoulderR],
                         [z_hipR, z_shoulderR]]
    skeleton[7, :, :] = [[x_hipL, x_shoulderL], [y_hipL, y_shoulderL],
                         [z_hipL, z_shoulderL]]
    skeleton[8, :, :] = [[x_hipR, x_kneeR], [y_hipR, y_kneeR],
                         [z_hipR, z_kneeR]]
    skeleton[9, :, :] = [[x_hipL, x_kneeL], [y_hipL, y_kneeL],
                         [z_hipL, z_kneeL]]
    skeleton[10, :, :] = [[x_footR, x_kneeR], [y_footR, y_kneeR],
                          [z_footR, z_kneeR]]
    skeleton[11, :, :] = [[x_footL, x_kneeL], [y_footL, y_kneeL],
                          [z_footL, z_kneeL]]
    skeleton[12, :, :] = [[x_shoulderR,
                           x_shoulderL], [y_shoulderR, y_shoulderL],
                          [z_shoulderR, z_shoulderL]]
    skeleton[13, :, :] = [[x_hipR, x_hipL], [y_hipR, y_hipL], [z_hipR, z_hipL]]
    return skeleton


def plot_3d(ax,
            y_true_pred,
            c='red',
            limits=[[-500, 500], [-500, 500], [0, 1500]],
            plot_lines=True):
    " 3D plot of single frame. Can be both label or prediction "
    x = y_true_pred[:, 0]
    y = y_true_pred[:, 1]
    z = y_true_pred[:, 2]
    ax.scatter(x, y, z, zdir='z', s=20, c=c, marker='o', depthshade=True)

    lines_skeleton = _get_skeleton_lines(x, y, z)
    if plot_lines:
        for line in range(len(lines_skeleton)):
            ax.plot(lines_skeleton[line, 0, :],
                    lines_skeleton[line, 1, :],
                    lines_skeleton[line, 2, :],
                    c,
                    label='gt')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    x_limits = limits[0]
    y_limits = limits[1]
    z_limits = limits[2]
    x_range = np.abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = np.abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = np.abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * np.max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
