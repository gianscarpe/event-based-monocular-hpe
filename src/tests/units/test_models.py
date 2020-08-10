import unittest

import torch

from experimenting.models.autoencoder import AutoEncoder
from experimenting.models.margipose import MargiPoseModel3D
from experimenting.utils import get_feature_extractor


class TestMargipose(unittest.TestCase):
    def setUp(self):
        self.hparams = {
            'n_joints':
            13,
            'in_cnn':
            get_feature_extractor({
                'model': 'resnet34',
                'n_channels': 1,
                'pretrained': True
            }),
            'mid_dimension': (128, 16, 16),
            'n_stages':
            3
        }
        self.model = MargiPoseModel3D(**self.hparams)

    def test_init(self):
        self.assertIsNotNone(self.model)

    def test_forward(self):
        batch_input = torch.randn(32, 1, 260, 346)
        out = self.model(batch_input)
        self.assertIsNotNone(out)


class TestAE(unittest.TestCase):
    def setUp(self):

        self.hparams = {
            'in_channels':
            4,
            'in_cnn':
            get_feature_extractor({
                'model': 'resnet34_cut_256',
                'n_channels': 4,
                'pretrained': True
            }),
            'mid_dimension': (256, 16, 16),
            'latent_size':
            256,
            'up_layers':
            4,
        }
        self.model = AutoEncoder(**self.hparams)

    def test_init(self):
        self.assertIsNotNone(self.model)

    def test_forward(self):
        batch_input = torch.randn(32, 4, 256, 256)
        out = self.model(batch_input)
        self.assertEquals(out.shape, batch_input.shape)
