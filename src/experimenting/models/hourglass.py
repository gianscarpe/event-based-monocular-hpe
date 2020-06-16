from math import sqrt
from pathlib import Path

import torch
from torch import nn
from torch.nn import init
from torch.nn.modules.conv import _ConvNd

from ..utils import FlatSoftmax

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


class HeatmapCombiner(nn.Module):
    def __init__(self, n_joints, out_channels):
        super().__init__()
        self.combine_block = _regular_block(n_joints, out_channels)

    def forward(self, x):
        return self.combine_block(x)


class HeatmapPredictor(nn.Module):
    """
    From https://raw.githubusercontent.com/anibali/margipose/
    """
    def __init__(self, n_joints, in_channels):
        super().__init__()
        self.n_joints = n_joints
        self.down_layers = nn.Sequential(
            _regular_block(in_channels, in_channels),
            _regular_block(in_channels, in_channels),
            _down_stride_block(in_channels, 192),
            _regular_block(192, 192),
            _regular_block(192, 192),
        )
        self.up_layers = nn.Sequential(
            _regular_block(192, 192),
            _regular_block(192, 192),
            _up_stride_block(192, in_channels),
            _regular_block(in_channels, in_channels),
            _regular_block(in_channels, self.n_joints),
        )
        init_parameters(self)

    def forward(self, inputs):
        mid_in = self.down_layers(inputs)
        return self.up_layers(mid_in)


def _get_feature_extractor(model_path):
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


class HourglassStage(nn.Module):
    def __init__(self, n_joints, mid_feature_dimension):
        super().__init__()

        self.softmax = FlatSoftmax()

    def forward(self, x):
        out = self.softmax(self.hm_predictor(x))

        return out


class HourglassModel(nn.Module):
    def __init__(self, n_stages, backbone_path, n_joints, n_channels=1):
        super().__init__()

        self.n_stages = n_stages
        self.in_cnn, self.mid_feature_dimension = _get_feature_extractor(
            backbone_path)

        self.in_channels = n_channels
        self.softmax = FlatSoftmax()
        self.n_joints = n_joints
        self.hm_combiners = nn.ModuleList()
        self.hg_stages = nn.ModuleList()
        self.softmax = FlatSoftmax()

        for t in range(self.n_stages):
            if t > 0:
                self.hm_combiners.append(
                    HeatmapCombiner(self.n_joints, self.mid_feature_dimension))
            self.hg_stages.append(
                HourglassStage(n_joints, self.mid_feature_dimension))

    def forward(self, x):
        inp = self.in_cnn(x)

        outs = []
        for t in range(self.n_stages):
            if t > 0:
                inp = inp + self.hm_combiners[t - 1](outs[-1])

            outs.append(self.hg_stages[t](inp))

        return outs


class MargiPoseStage(nn.Module):
    def __init__(self, n_joints, mid_feature_dimension, heatmap_space):
        super().__init__()

        self.n_joints = n_joints
        self.softmax = FlatSoftmax()
        self.heatmap_space = heatmap_space
        self.down_layers = nn.Sequential(
            _regular_block(128, 128),
            _regular_block(128, 128),
            _down_stride_block(128, 192),
            _regular_block(192, 192),
            _regular_block(192, 192),
        )
        self.up_layers = nn.Sequential(
            _regular_block(192, 192),
            _regular_block(192, 192),
            _up_stride_block(192, 128),
            _regular_block(128, 128),
            _regular_block(128, self.n_joints),
        )

    def forward(self, *inputs):
        mid_in = self.down_layers(inputs[0])

        size = mid_in.shape[-1]
        if self.heatmap_space == 'xy':
            mid_out = mid_in
        elif self.heatmap_space == 'zy':
            mid_out = torch.cat(
                [t.permute(0, 3, 2, 1) for t in mid_in.split(size, -3)], -3)
        elif self.heatmap_space == 'xz':
            mid_out = torch.cat(
                [t.permute(0, 2, 1, 3) for t in mid_in.split(size, -3)], -3)
        else:
            raise Exception()
        return self.up_layers(mid_out)


class MargiPoseModel3D(nn.Module):
    def __init__(self, n_stages, backbone_path, n_joints, n_channels=1):
        super().__init__()

        self.n_stages = n_stages
        self.in_cnn, self.mid_feature_dimension = _get_feature_extractor(
            backbone_path)
        self.xy_hm_cnns = nn.ModuleList()
        self.zy_hm_cnns = nn.ModuleList()
        self.xz_hm_cnns = nn.ModuleList()
        self.hm_combiners = nn.ModuleList()
        self.softmax = FlatSoftmax()
        self.n_joints = n_joints

        self._set_stages()

    def _set_stages(self):
        for t in range(self.n_stages):
            if t > 0:
                self.hm_combiners.append(
                    HeatmapCombiner(3 * self.n_joints,
                                    self.mid_feature_dimension))
            self.xy_hm_cnns.append(
                MargiPoseStage(self.n_joints,
                               self.mid_feature_dimension,
                               heatmap_space='xy'))
            self.zy_hm_cnns.append(
                MargiPoseStage(self.n_joints,
                               self.mid_feature_dimension,
                               heatmap_space='zy'))
            self.xz_hm_cnns.append(
                MargiPoseStage(self.n_joints,
                               self.mid_feature_dimension,
                               heatmap_space='xz'))

    def forward(self, inputs):
        features = self.in_cnn(inputs)
        xy_heatmaps = []
        zy_heatmaps = []
        xz_heatmaps = []

        inp = features
        for t in range(self.n_stages):
            if t > 0:
                combined_hm_features = self.hm_combiners[t - 1](torch.cat([
                    xy_heatmaps[t - 1], zy_heatmaps[t - 1], xz_heatmaps[t - 1]
                ], -3))
                inp = inp + combined_hm_features
            xy_heatmaps.append(self.softmax(self.xy_hm_cnns[t](inp)))
            zy_heatmaps.append(self.softmax(self.zy_hm_cnns[t](inp)))
            xz_heatmaps.append(self.softmax(self.xz_hm_cnns[t](inp)))

        return xy_heatmaps, zy_heatmaps, xz_heatmaps


class MargiPoseModel2D(MargiPoseModel3D):
    def __init__(self, n_stages, backbone_path, n_joints, n_channels=1):
        super().__init__(n_stages, backbone_path, n_joints, n_channels)

    def _set_stages(self):
        for t in range(self.n_stages):
            if t > 0:
                self.hm_combiners.append(
                    HeatmapCombiner(self.n_joints, self.mid_feature_dimension))
            self.xy_hm_cnns.append(
                MargiPoseStage(self.n_joints,
                               self.mid_feature_dimension,
                               heatmap_space='xy'))

    def forward(self, inputs):
        features = self.in_cnn(inputs)
        xy_heatmaps = []

        inp = features
        for t in range(self.n_stages):
            if t > 0:
                combined_hm_features = self.hm_combiners[t - 1](xy_heatmaps[t -
                                                                            1])
                inp = inp + combined_hm_features
            xy_heatmaps.append(self.softmax(self.xy_hm_cnns[t](inp)))

        return xy_heatmaps


def get_margipose_model(params):
    predict_3d = params.pop('predict_3d')
    if predict_3d:
        return MargiPoseModel3D(**params)
    else:
        return MargiPoseModel2D(**params)
