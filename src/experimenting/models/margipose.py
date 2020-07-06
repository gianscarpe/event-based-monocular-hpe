import torch
from torch import nn

from ..utils import (
    FlatSoftmax,
    _down_stride_block,
    _regular_block,
    _up_stride_block,
    get_feature_extractor,
    init_parameters,
)


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
        init_parameters(self)

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
    def __init__(
        self,
        n_stages,
        feature_extractor,
        mid_features_dimension,
        n_joints
    ):
        super().__init__()

        self.n_stages = n_stages

        self.in_cnn = feature_extractor
        self.mid_feature_dimension = mid_features_dimension
        self.xy_hm_cnns = nn.ModuleList()
        self.zy_hm_cnns = nn.ModuleList()
        self.xz_hm_cnns = nn.ModuleList()
        self.hm_combiners = nn.ModuleList()
        self.softmax = FlatSoftmax()
        self.n_joints = n_joints

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
                        3, self.n_joints, self.mid_feature_dimension))
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


def _get_feature_extractor(params):
    extractor_params = {'n_channels': params.n_channels}
    if params.backbone_path is not None and os.exists(params.backbone_path):
        extractor_params['custom_model_path'] = params.backbone_path
    elif params.pretrained_imagenet is not None:
        extractor_params['pretrained'] = params.pretrained_imagenet
    feature_extractor, mid_features_dimension = get_feature_extractor(
            extractor_params)

    return feature_extractor, mid_features_dimension
        

def get_margipose_model(params):
    predict_3d = params.pop('predict_3d')
    feature_extractor, mid_features_dimension = _get_feature_extractor(params)
    params['feature_extractor'] = feature_extractor
    params['mid_features_dimension'] = mid_features_dimension

    if predict_3d:
        return MargiPoseModel3D(**params)
    else:
        return MargiPoseModel2D(**params)
