import torch
from torch import nn
from .custom import FlatSoftmax
from math import sqrt
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn import init


def _regular_block(in_chans, out_chans):
    return ResidualBlock(
        out_chans,
        nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
        nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False))

def _down_stride_block(in_chans, out_chans):
    return ResidualBlock(
        out_chans,
        nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, stride=2, bias=False),
        nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=2, bias=False))

def _up_stride_block(in_chans, out_chans):
    return ResidualBlock(
        out_chans,
        nn.ConvTranspose2d(in_chans, out_chans, kernel_size=3, padding=1, stride=2,
                           output_padding=1, bias=False),
        nn.ConvTranspose2d(in_chans, out_chans, kernel_size=1, stride=2,
                           output_padding=1, bias=False))

    
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
    From https://raw.githubusercontent.com/anibali/margipose/a9dbe5c3151d7a7e071df6275d5702c07ef5152d/src/margipose/models/margipose_model.py
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
        self.combine_block = _regular_block(n_joints,
                                            out_channels)

    def forward(self, x):
        return self.combine_block(x)

class HeatmapPredictor(nn.Module):
    """
    From https://raw.githubusercontent.com/anibali/margipose/a9dbe5c3151d7a7e071df6275d5702c07ef5152d/src/margipose/models/margipose_model.py
    """

    def __init__(self, n_joints):
        super().__init__()
        self.n_joints = n_joints
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
            _up_stride_block(128, 128),
            _regular_block(128, 128),
            _up_stride_block(128, 128),
            _regular_block(128, 128),
            _up_stride_block(128, 128),
            _regular_block(128, 128),
            _regular_block(128, self.n_joints),
        )
        init_parameters(self)


    def forward(self, inputs):
        mid_in = self.down_layers(inputs)
        return self.up_layers(mid_in)


def _get_feature_extractor(model_path, device):
    resnet = torch.load(model_path, map_location=device)
    net = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
    )
    return net
    

class HourglassStage(nn.Module):
    def __init__(self, n_joints):   
        super().__init__()

        self.softmax = FlatSoftmax()
        self.hm_predictor = HeatmapPredictor(n_joints)

        
    def forward(self, x):
        out = self.softmax(self.hm_predictor(x))
            
        return out
    
class HourglassModel(nn.Module):
    def __init__(self, n_stages, backbone_path, n_joints, device):   
        super().__init__()

        self.mid_feature_dimension = 128
        self.n_stages = n_stages
        self.in_cnn = _get_feature_extractor(backbone_path, device)
        self.softmax = FlatSoftmax()
        self.hg_stages = []
        self.hm_combiners = []
        self.n_joints = n_joints

        self.softmax = FlatSoftmax()

        for t in range(self.n_stages):
            if t > 0:
                self.hm_combiners.append(HeatmapCombiner(self.n_joints, self.mid_feature_dimension))
            self.hg_stages.append(HourglassStage(n_joints))


    def forward(self, x):
        inp = self.in_cnn(inputs)
        
        outs = []
        for t in range(self.n_stages):
            if t > 0:
                inp = inp, self.hm_combiners[t](outs[-1])], 1)
                
            outs.append(self.hg_stages[t](inp))
            
        return outs
