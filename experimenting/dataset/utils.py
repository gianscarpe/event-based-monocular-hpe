import numpy as np

def get_heatmap(vicon_xyz, p_mat, heigth)
" From 3D label, get 2D label coordinates and heatmaps for selected camera "
    # use homogeneous coordinates representation to project 3d XYZ coordinates to 2d UV pixel coordinates.
    vicon_xyz_homog = np.concatenate([vicon_xyz, np.ones([1,13])], axis=0)
    coord_pix_homog = np.matmul(P_mat_cam, vicon_xyz_homog)
    coord_pix_homog_norm = coord_pix_homog / coord_pix_homog[-1]
    
    u = coord_pix_homog_norm[0]
    v = heigth - coord_pix_homog_norm[1] # flip v coordinate to match the image direction

    # mask is used to make sure that pixel positions are in frame range.
    mask = np.ones(u.shape).astype(np.float32)
    mask[np.isnan(u)] = 0; mask[np.isnan(v)] = 0
    mask[u>W] = 0; mask[u<=0] = 0; mask[v>H] = 0; mask[v<=0] = 0

    # pixel coordinates
    u = u.astype(np.int32)
    v = v.astype(np.int32)
    
    # initialize, fill and smooth the heatmaps
    label_heatmaps = np.zeros((H, W, num_joints))
    for fmidx, zipd in enumerate(zip(v, u, mask)):
        if zipd[2]==1: # write joint position only when projection within frame boundaries
            label_heatmaps[zipd[0], zipd[1], fmidx] = 1
            label_heatmaps[:,:,fmidx] = decay_heatmap(label_heatmaps[:,:,fmidx])

    return np.stack((v,u), axis=-1), mask, label_heatmaps

