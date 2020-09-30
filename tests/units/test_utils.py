import unittest
from unittest import mock

import torch

from experimenting import utils


class TestSkeleton(unittest.TestCase):
    precision_error = 1e-5  # acceptable precision error in mm for algebra calculation

    def setUp(self):
        self.joints = torch.randn(13, 3)
        self.sk = utils.skeleton_helpers.Skeleton(self.joints)

    def test_init(self):
        self.assertIsInstance(self.sk, utils.skeleton_helpers.Skeleton)
        self.assertTrue(torch.equal(self.sk._get_tensor(), self.joints))

    def test_torso_length(self):
        neck_point = self.joints.index_select(0, torch.LongTensor([1,
                                                                   2])).mean(0)

        pelvic_point = self.joints.index_select(0,
                                                torch.LongTensor([5,
                                                                  6])).mean(0)
        distance = torch.norm(neck_point - pelvic_point)
        self.assertEqual(self.sk.get_skeleton_torso_length(), distance)

    def test_normalize_denormalize_z_ref(self):
        camera = torch.randn(3, 4)
        height = 100
        width = 100
        z_ref = self.joints[0, -1]

        normalized_skeleton = self.sk.normalize(height, width, camera, z_ref)
        denormalized_skeleton = normalized_skeleton.denormalize(height,
                                                                width,
                                                                camera,
                                                                z_ref=z_ref)

        self.assertIsInstance(normalized_skeleton,
                              utils.skeleton_helpers.Skeleton)
        self.assertIsInstance(denormalized_skeleton,
                              utils.skeleton_helpers.Skeleton)

        self.assertLess(
            torch.norm(self.sk._get_tensor() -
                       denormalized_skeleton._get_tensor()),
            TestSkeleton.precision_error)

    def test_normalize_denormalize_torso(self):
        camera = torch.randn(3, 4)
        height = 100
        width = 100
        z_ref = self.joints[0, -1]
        torso_length = self.sk.get_skeleton_torso_length()
        estimation_error = 20  # 20mm of estimation error

        normalized_skeleton = self.sk.normalize(height, width, camera, z_ref)
        denormalized_skeleton = normalized_skeleton.denormalize(
            height, width, camera, torso_length=torso_length)

        self.assertIsInstance(normalized_skeleton,
                              utils.skeleton_helpers.Skeleton)
        self.assertIsInstance(denormalized_skeleton,
                              utils.skeleton_helpers.Skeleton)
