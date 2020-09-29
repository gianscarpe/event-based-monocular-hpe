"""
Skeleton wrapper. It provides a toolbox for plotting, projection, normalization,
and denormalization of skeletons joints

"""

import numpy as np
import torch
from matplotlib import pyplot as plt

from pose3d_utils.camera import CameraIntrinsics
from pose3d_utils.coords import ensure_homogeneous
from pose3d_utils.skeleton_normaliser import SkeletonNormaliser

from .cv_helpers import _project_xyz_onto_image, reproject_xyz_onto_world_coord


class Skeleton:
    _SKELETON_D = 3

    def __init__(self, skeleton, label="skeleton"):
        skeleton = ensure_homogeneous(skeleton, d=Skeleton._SKELETON_D)
        self._skeleton = skeleton
        self.label = label
        self._normalizer = SkeletonNormaliser()
        self.head_point = skeleton[0]
        self.right_shoulder_point = skeleton[1]
        self.left_shoulder_point = skeleton[2]
        self.right_elbow_point = skeleton[3]
        self.left_elbow_point = skeleton[4]
        self.right_hip_point = skeleton[5]
        self.left_hip_point = skeleton[6]
        self.right_hand_point = skeleton[7]
        self.left_hand_point = skeleton[8]
        self.right_knee_point = skeleton[9]
        self.left_knee_point = skeleton[10]
        self.right_foot_point = skeleton[11]
        self.left_foot_point = skeleton[12]
        self.neck_point = skeleton.index_select(0,
                                                torch.LongTensor([1,
                                                                  2])).mean(0)

        self.pelvic_point = skeleton.index_select(0,
                                                  torch.LongTensor([5, 6
                                                                    ])).mean(0)

    def _get_tensor(self):
        return self._skeleton.narrow(-1, 0, 3)

    def get_left_arm_length(self):
        return (torch.norm(self.left_shoulder_point - self.left_elbow_point)
                ) + (torch.norm(self.left_elbow_point - self.left_hand_point))

    def get_right_arm_length(self):
        return (torch.norm(self.right_shoulder_point - self.right_elbow_point)
                ) + (torch.norm(self.right_elbow_point -
                                self.right_hand_point))

    def get_shoulders_distance(self):
        return (torch.norm(self.right_shoulder_point -
                           self.left_shoulder_point))

    def get_skeleton_longitudinal_lenght(self):
        return self.get_right_arm_length() + self.get_shoulders_distance(
        ) + self.get_left_arm_length()

    def get_skeleton_height(self):
        """
        """
        return self.get_skeleton_head_neck_length(
        ) + self.get_skeleton_torso_length() + self.get_skeleton_leg_length()

    def get_skeleton_femur_length(self):

        return torch.norm(self.left_hip_point - self.left_knee_point)

    def get_skeleton_torso_length(self):
        """
        alculates the length of skeleton torso in input m.u.
        """
        return (torch.norm(self.neck_point - self.pelvic_point))

    def get_skeleton_head_neck_length(self):
        """
        alculates the length of skeleton torso in input m.u.
        """
        return (torch.norm(self.head_point - self.neck_point))

    def get_skeleton_leg_length(self):
        """
        alculates the length of skeleton torso in input m.u.
        """

        return (torch.norm(self.left_hip_point - self.left_knee_point)) + (
            torch.norm(self.left_knee_point - self.left_foot_point))

    def proportion(self, torso_length):
        """

        """
        return (self.get_skeleton_torso_length() / torso_length)

    def reproject_onto_world(self, M):
        return Skeleton(reproject_xyz_onto_world_coord(self._get_tensor(), M))

    def denormalize(self,
                    height,
                    width,
                    camera,
                    torso_length=None,
                    z_ref=None):
        """

        Args:
            pred : joints coordinates as NUM_JOINTSx3
            height : height of frame
            width : width of frame
            camera : intrinsics parameters

        Returns
            Denormalized skeleton joints as NUM_JOINTSx3
        """

        # skeleton

        camera = CameraIntrinsics(camera)
        homog = ensure_homogeneous(self._get_tensor(), d=Skeleton._SKELETON_D)
        if z_ref is None:
            if torso_length is None:
                torso_length = 450

            z_ref = self._normalizer.infer_depth(
                homog,
                lambda x: Skeleton(x).proportion(torso_length),
                camera,
                height,
                width,
            )

        pred_skeleton = self._normalizer.denormalise_skeleton(
            homog, z_ref, camera, height, width)
        pred_skeleton = pred_skeleton.narrow(-1, 0, 3)
        return Skeleton(pred_skeleton)

    @staticmethod
    def _get_skeleton_lines(x, y, z):
        """
        From DHP19 toolbox
        """
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
        skeleton[12, :, :] = [[x_shoulderR, x_shoulderL],
                              [y_shoulderR, y_shoulderL],
                              [z_shoulderR, z_shoulderL]]
        skeleton[13, :, :] = [[x_hipR, x_hipL], [y_hipR, y_hipL],
                              [z_hipR, z_hipL]]
        return skeleton

    def plot_3d(self, ax, c='red', limits=None, plot_lines=True):
        """
        Plot the provided skeletons in 3D coordinate space
        Args:
          ax: axis for plot
          y_true_pred: joints to plot in 3D coordinate space
          c: color (Default value = 'red')
          limits: list of 3 ranges (x, y, and z limits)
          plot_lines:  (Default value = True)

        Note:
          Plot the provided skeletons. Visualization purpose only

        From DHP19 toolbox        
        """

        if limits is None:
            limits = [[-500, 500], [-500, 500], [0, 1500]]

        points = self._get_tensor()
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        ax.scatter(x, y, z, zdir='z', s=20, c=c, marker='o', depthshade=True)

        lines_skeleton = Skeleton._get_skeleton_lines(x, y, z)

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

    def get_2d_points(self, p_mat, height, width):
        points = self._get_tensor()[:, :3].transpose(1, 0)
        xj, yj, _ = _project_xyz_onto_image(points.numpy(), p_mat.numpy(),
                                            height, width)
        joints = np.array([xj, yj]).transpose(1, 0)
        return joints