import unittest

import torch

from experimenting.models.autoencoder import AutoEncoder
from experimenting.models.margipose import MargiPoseModel3D
from experimenting.utils import get_feature_extractor


class TestAE(unittest.TestCase):
    def setUp(self):

        self.hparams = {
            'in_shape':
            (4, 260, 346),
            'in_cnn':
            get_feature_extractor({
                'model': 'resnet34',
                'n_channels': 4,
                'pretrained': True
            }),
            'latent_size':
            128,
            'up_layers':
            3,
        }
        self.model = AutoEncoder(**self.hparams)

    def test_init(self):
        self.assertIsNotNone(self.model)

    def test_forward(self):
        batch_input = torch.randn(32, 4, 260, 346)
        out = self.model(batch_input)
        self.assertEqual(out.shape, batch_input.shape)


class TestMargiposeResnet34(unittest.TestCase):
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
            'latent_size': 128,
            'n_stages':
            4
        }
        self.model = MargiPoseModel3D(**self.hparams)

    def test_init(self):
        self.assertIsNotNone(self.model)

    def test_forward(self):
        batch_input = torch.randn(32, 1, 260, 346)
        out = self.model(batch_input)
        self.assertIsNotNone(out)

        
class TestMargiposeResnet50(unittest.TestCase):
    def setUp(self):
        self.hparams = {
            'n_joints':
            13,
            'in_cnn':
            get_feature_extractor({
                'model': 'resnet50',
                'n_channels': 1,
                'pretrained': True
            }),
            'latent_size': 256,
            'n_stages': 3
        }
        self.model = MargiPoseModel3D(**self.hparams)

    def test_init(self):
        self.assertIsNotNone(self.model)

    def test_forward(self):
        batch_input = torch.randn(32, 1, 260, 346)
        out = self.model(batch_input)
        self.assertIsNotNone(out)
