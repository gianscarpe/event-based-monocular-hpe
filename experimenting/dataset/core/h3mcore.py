import os
import re
from typing import List

import numpy as np
import torch
from kornia import quaternion_to_rotation_matrix
from pose3d_utils.camera import CameraIntrinsics

from experimenting.utils import Skeleton

from ..utils import get_file_paths
from .base import BaseCore
from .h3m import h36m_cameras_extrinsic_params, h36m_cameras_intrinsic_params


class HumanCore(BaseCore):
    """
    Human3.6m core class. It provides implementation to load frames, 2djoints, 3d joints

    """

    CAMS_ID_MAP = {'54138969': 0, '55011271': 1, '58860488': 2, '60457274': 3}
    JOINTS = [15, 25, 17, 26, 18, 1, 6, 27, 19, 2, 7, 3, 8]
    LABELS_MAP = {
        'Directions': 0,
        'Discussion': 1,
        'Eating': 2,
        'Greeting': 3,
        'Phoning': 4,
        'Posing': 5,
        'Purchase': 6,
        'Sitting': 7,
        'SittingDown': 8,
        'Smoking': 9,
        'Photo': 10,
        'Waiting': 11,
        'Walking': 12,
        'WalkDog': 13,
        'WalkTogether': 14,
        'Purchases': 15,
        '_ALL': 15,
    }
    MAX_WIDTH = 260  # DVS resolution
    MAX_HEIGHT = 346  # DVS resolution
    N_JOINTS = 13
    N_CLASSES = 2
    TORSO_LENGTH = 453.5242317  # TODO
    DEFAULT_TEST_SUBJECTS: List[int] = [6, 7]
    DEFAULT_TEST_VIEW = [1, 3]

    def __init__(
        self,
        name,
        data_dir,
        joints_path,
        partition,
        n_channels,
        movs=None,
        test_subjects=None,
        test_cams=None,
        avg_torso_length=TORSO_LENGTH,
        *args,
        **kwargs,
    ):
        super(HumanCore, self).__init__(name, partition)

        self.file_paths = HumanCore._get_file_paths_with_movs(data_dir, movs)

        self.in_shape = (HumanCore.MAX_HEIGHT, HumanCore.MAX_WIDTH)
        self.n_channels = n_channels

        self.avg_torso_length = avg_torso_length

        self.classification_labels = [
            HumanCore.get_label_from_filename(x_path) for x_path in self.file_paths
        ]
        self.n_joints = HumanCore.N_JOINTS
        self.joints = HumanCore.get_pose_data(joints_path)
        self.frames_info = [HumanCore.get_frame_info(x) for x in self.file_paths]

        if test_subjects is None:
            self.subjects = HumanCore.DEFAULT_TEST_SUBJECTS
        else:
            self.subjects = test_subjects

        if test_cams is None:
            self.view = HumanCore.DEFAULT_TEST_VIEW
        else:
            self.view = test_cams

    def get_test_subjects(self):
        return self.subjects

    def get_test_view(self):
        return self.view

    @staticmethod
    def get_label_from_filename(filepath) -> int:
        """
        Given the filepath, return the correspondent movement label (range [0, 32])

        Args:
            filepath (str): frame absolute filepath

        Returns:
            Frame label

        Examples:
            >>> HumanCore.get_label_from_filename("S1_session_2_mov_1_frame_249_cam_2.npy")
        """

        info = HumanCore.get_frame_info(filepath)

        return HumanCore.LABELS_MAP[info['action'].split(" ")[0]]

    @staticmethod
    def get_frame_info(filepath):
        """
        >>> HumanCore.get_label_frame_info("tests/data/h3m/S1/Directions 1.54138969S1/frame0000001.npy")
        {'subject': 1, 'actions': 'Directions', cam': 0, 'frame': '0000007'}
        """
        base_subject_dir = filepath[re.search(r'(?<=S)\d+/', filepath).span()[1] :]
        infos = base_subject_dir.split('/')

        cam = re.search(r"(?<=\.)\d+", base_subject_dir)
        cam = HumanCore.CAMS_ID_MAP[cam.group(0)] if cam is not None else None
        result = {
            "subject": int(re.search(r'(?<=S)\d+', filepath).group(0)),
            "action": re.findall(r"(\w+\s?\d?)\.\d+", base_subject_dir)[0],
            "cam": cam,
            "frame": re.search(r"\d+", infos[-1]).group(0),
        }

        return result

    @staticmethod
    def _get_file_paths_with_movs(data_dir, movs):
        file_paths = np.array(get_file_paths(data_dir, ['npy']))
        if movs is not None:
            mov_mask = [
                HumanCore.get_label_from_filename(x) in movs for x in file_paths
            ]

            file_paths = file_paths[mov_mask]
        return file_paths

    @staticmethod
    def get_pose_data(path):
        # Load serialized dataset
        data = np.load(path, allow_pickle=True)['positions_3d'].item()

        result = {}
        for subject, actions in data.items():
            subject_n = int(re.search(r"\d+", subject).group(0))
            result[subject_n] = {}

            for action_name, positions in actions.items():
                result[subject_n][action_name] = {
                    'positions': positions,
                    'extrinsics': h36m_cameras_extrinsic_params[subject],
                }

        return result

    @staticmethod
    def _build_intrinsic(intrinsic_matrix_params: dict) -> torch.Tensor:
        # scale to DVS frame dimension
        w_ratio = HumanCore.MAX_WIDTH / intrinsic_matrix_params['res_w']
        h_ratio = HumanCore.MAX_HEIGHT / intrinsic_matrix_params['res_h']

        intr_linear_matrix = torch.tensor(
            [
                [
                    w_ratio * intrinsic_matrix_params['focal_length'][0],
                    0,
                    w_ratio * intrinsic_matrix_params['center'][0],
                    0,
                ],
                [
                    0,
                    h_ratio * intrinsic_matrix_params['focal_length'][1],
                    h_ratio * intrinsic_matrix_params['center'][1],
                    0,
                ],
                [0, 0, 1, 0],
            ]
        )
        return intr_linear_matrix

    @staticmethod
    def _build_extrinsic(extrinsic_matrix_params: dict) -> torch.Tensor:

        quaternion = torch.tensor(extrinsic_matrix_params['orientation'])[[1, 2, 3, 0]]
        quaternion[:3] *= -1

        R = quaternion_to_rotation_matrix(quaternion)
        t = torch.tensor(extrinsic_matrix_params['translation'])
        tr = -torch.matmul(R, t)

        return torch.cat([torch.cat([R, tr.unsqueeze(1)], dim=1)], dim=0,)

    def get_joint_from_id(self, idx):
        frame_info = self.frames_info[idx]
        frame_n = int(frame_info['frame'])
        joints_data = self.joints[frame_info['subject']][frame_info['action']][
            'positions'
        ][frame_n]

        joints_data = joints_data[HumanCore.JOINTS] * 1000  # Scale to cm

        intr_matrix = HumanCore._build_intrinsic(
            h36m_cameras_intrinsic_params[frame_info['cam']]
        )

        extr = self.joints[frame_info['subject']][frame_info['action']]['extrinsics'][
            frame_info['cam']
        ]

        extr_matrix = HumanCore._build_extrinsic(extr)
        return Skeleton(joints_data), intr_matrix, extr_matrix

    def get_frame_from_id(self, idx):
        path = self.file_paths[idx]
        x = np.load(path, allow_pickle=True) / 255.0
        if len(x.shape) == 2:
            x = np.expand_dims(x, -1)
        return x
