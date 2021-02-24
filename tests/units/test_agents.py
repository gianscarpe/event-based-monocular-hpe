import unittest
from unittest import mock

import pytorch_lightning as pl
import torch
from torch import nn

from experimenting.agents import MargiposeEstimator


def _get_mocked_feature_extractor(params: dict) -> nn.Sequential:
    net = nn.Sequential(nn.Conv2d(params['n_channels'], 32, kernel_size=3, padding=1))

    return net


class TestMargiposeAgent(unittest.TestCase):
    @mock.patch(
        'experimenting.agents.base.get_feature_extractor',
        _get_mocked_feature_extractor,
    )
    @mock.patch('hydra.utils')
    def setUp(self, mocked_hydra):
        self.n_channels = 1
        mocked_core = mock.MagicMock(
            n_channels=self.n_channels,
            in_shape=(224, 224),
            n_joints=13,
            avg_torso_length=100,
        )
        mocked_loss = mock.MagicMock()

        mocked_hydra.instantiate.return_value = mocked_loss

        self.parameters = {
            'optimizer': {'type': 'SGD'},
            'loss': {},
            'lr_scheduler': {},
            'model': 'test',
            'model_zoo': '/path/to/zoo',
            'backbone': 'backbone',
            'core': mocked_core,
        }
        self.agent = MargiposeEstimator(**self.parameters)

    def test_init(self):
        self.assertIsInstance(self.agent, pl.LightningModule)

    # def test_calculate_loss(self):
    #     batch = 32
    #     mocked_loss = mock.MagicMock()
    #     mocked_loss.return_value = torch.rand(batch)
    #     mocked_batch = [
    #         torch.rand((batch, self.n_channels, 224, 224)),
    #         mock.MagicMock(),
    #     ]

    #     elf.agent.loss_func = mocked_loss
    #     results = self.agent._eval(mocked_batch)
