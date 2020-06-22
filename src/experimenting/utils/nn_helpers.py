from pathlib import Path

import torch
from torch import nn
from torch.nn import init
from torch.nn.modules.conv import _ConvNd

from kornia.geometry import spatial_softmax2d

__all__ = ['FlatSoftmax', 'get_feature_extractor', '_regular_block',
           '_up_stride_block', 'init_parameters', '_down_stride_block']


class FlatSoftmax(nn.Module):
    def __init__(self):
        super(FlatSoftmax, self).__init__()

    def forward(self, inp):
        return spatial_softmax2d(inp)


RESNET34_MID_FEATURES = 128
RESNET50_MID_FEATURES = 512


def _regular_block(in_chans, out_chans):
    return ResidualBlock(
        out_chans,
        nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
        nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False))


def _down_stride_block(in_chans, out_chans):
    return ResidualBlock(
        out_chans,
        nn.Conv2d(in_chans,
                  out_chans,
                  kernel_size=3,
                  padding=1,
                  stride=2,
                  bias=False),
        nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=2, bias=False))


def _up_stride_block(in_chans, out_chans):
    return ResidualBlock(
        out_chans,
        nn.ConvTranspose2d(in_chans,
                           out_chans,
                           kernel_size=3,
                           padding=1,
                           stride=2,
                           output_padding=(1, 1),
                           bias=False),
        nn.ConvTranspose2d(in_chans,
                           out_chans,
                           kernel_size=1,
                           stride=2,
                           output_padding=(1, 1),
                           bias=False))


def init_parameters(net):
    for m in net.modules():
        if isinstance(m, _ConvNd):
            init.kaiming_normal_(m.weight, 0, 'fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # Kaiming initialisation for linear layers
            init.normal_(m.weight, 0, sqrt(2.0 / m.weight.size(0)))
            if m.bias is not None:
                init.normal_(m.bias, 0, sqrt(2.0 / m.bias.size(0)))
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            if m.bias is not None:
                init.constant_(m.bias, 0)


class ResidualBlock(nn.Module):
    """
    From https://raw.githubusercontent.com/anibali/margipose/
    """
    def __init__(self, chans, main_conv_in, shortcut_conv_in):
        super().__init__()
        assert main_conv_in.in_channels == shortcut_conv_in.in_channels
        self.module = nn.Sequential(
            main_conv_in,
            nn.BatchNorm2d(chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(chans, chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(chans),
            nn.ReLU(inplace=True),
        )
        self.shortcut = nn.Sequential(shortcut_conv_in, nn.BatchNorm2d(chans))

    def forward(self, *inputs):
        return self.module(inputs[0]) + self.shortcut(inputs[0])

    
def get_feature_extractor(model_path):
    switch = {
        'resnet34': _get_resnet34_feature_extactor,
        'resnet50': _get_resnet50_feature_extactor
    }

    model_name = Path(model_path).with_suffix('').name

    return switch[model_name](model_path)


def _get_resnet34_feature_extactor(model_path):
    resnet = torch.load(model_path)

    net = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
    )
    return net, RESNET34_MID_FEATURES


def _get_resnet50_feature_extactor(model_path):
    resnet = torch.load(model_path)

    net = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
    )
    return net, RESNET50_MID_FEATURES
