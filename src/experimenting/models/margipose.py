import torch
from torch import nn

from ..utils import (FlatSoftmax, _down_stride_block, _regular_block,
                     _up_stride_block, init_parameters,
                     get_backbone_last_dimension)


class MargiPoseStage(nn.Module):
    def __init__(self, n_joints, mid_feature_dimension, heatmap_space,
                 permute):
        super().__init__()

        self.n_joints = n_joints
        self.softmax = FlatSoftmax()
        self.heatmap_space = heatmap_space
        min_dimension = 64
        max_dimension = 128
        self.down_layers = nn.Sequential(
            _regular_block(mid_feature_dimension, min_dimension),
            _regular_block(min_dimension, min_dimension),
            _down_stride_block(min_dimension, max_dimension),
            _regular_block(max_dimension, max_dimension),
            _regular_block(max_dimension, max_dimension),
        )
        self.up_layers = nn.Sequential(
            _regular_block(max_dimension, max_dimension),
            _regular_block(max_dimension, max_dimension),
            _up_stride_block(max_dimension, min_dimension),
            _regular_block(min_dimension, min_dimension),
            _regular_block(min_dimension, self.n_joints),
        )
        self.permute = permute
        init_parameters(self)

    def forward(self, *inputs):
        mid_in = self.down_layers(inputs[0])
        if self.permute:
            size = mid_in.shape[-1]

            if self.heatmap_space == 'xy':
                mid_out = mid_in
            elif self.heatmap_space == 'zy':
                mid_out = torch.cat(
                    [t.permute(0, 3, 2, 1) for t in mid_in.split(size, -3)],
                    -3)
            elif self.heatmap_space == 'xz':

                mid_out = torch.cat(
                    [t.permute(0, 2, 1, 3) for t in mid_in.split(size, -3)],
                    -3)
            else:
                raise Exception()
        else:
            mid_out = mid_in
        return self.up_layers(mid_out)


class MargiPoseModel3D(nn.Module):
    def __init__(self,
                 n_stages,
                 in_cnn,
                 latent_size,
                 n_joints,
                 permute_axis=False):
        super().__init__()

        self.n_stages = n_stages
        self.in_cnn = in_cnn
        temp_in_shape = [1, 256, 256
                         ]  # placeholder to call get_backbone_last_dimension
        self.mid_feature_dimension = get_backbone_last_dimension(
            in_cnn, temp_in_shape)[0]
        self.xy_hm_cnns = nn.ModuleList()
        self.zy_hm_cnns = nn.ModuleList()
        self.xz_hm_cnns = nn.ModuleList()
        self.hm_combiners = nn.ModuleList()
        self.softmax = FlatSoftmax()
        self.n_joints = n_joints
        self.permute_axis = permute_axis
        self._set_stages()

    class _HeatmapCombiner(nn.Module):
        def __init__(self, n_joints, n_planes, out_channels):
            super().__init__()
            self.combine_block = _regular_block(n_joints * n_planes,
                                                out_channels)

        def forward(self, x):
            return self.combine_block(x)

    def _set_stages(self):
        for t in range(self.n_stages):
            if t > 0:
                self.hm_combiners.append(
                    MargiPoseModel3D._HeatmapCombiner(
                        self.n_joints, 3, self.mid_feature_dimension))
            self.xy_hm_cnns.append(
                MargiPoseStage(self.n_joints,
                               self.mid_feature_dimension,
                               heatmap_space='xy',
                               permute=self.permute_axis))
            self.zy_hm_cnns.append(
                MargiPoseStage(self.n_joints,
                               self.mid_feature_dimension,
                               heatmap_space='zy',
                               permute=self.permute_axis))
            self.xz_hm_cnns.append(
                MargiPoseStage(self.n_joints,
                               self.mid_feature_dimension,
                               heatmap_space='xz',
                               permute=self.permute_axis))

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

    class _HeatmapCombiner(nn.Module):
        def __init__(self, n_joints, n_planes, out_channels):
            super().__init__()
            self.combine_block = _regular_block(n_joints * n_planes,
                                                out_channels)

        def forward(self, x):
            return self.combine_block(x)

    def _set_stages(self):
        for t in range(self.n_stages):
            if t > 0:
                self.hm_combiners.append(
                    MargiPoseModel2D._HeatmapCombiner(
                        self.n_joints, self.mid_feature_dimension))
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
