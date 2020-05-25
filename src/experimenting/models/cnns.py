import torch
import torch.nn as nn

import segmentation_models_pytorch as smp
from torchvision import models

from ..utils import FlatSoftmax
from .dhp19 import DHP19Model


def get_cnn(model_name, params):
    switcher = {
        'resnet18': lambda **args: _get_resnet('resnet18', **args),
        'resnet34': lambda **args: _get_resnet('resnet34', **args),
        'resnet50': lambda **args: _get_resnet('resnet50', **args),
        'unet_resnet18': lambda **args: _get_unet_resnet('resnet18', **args),
        'unet_resnet34': lambda **args: _get_unet_resnet('resnet34', **args),
        'dhp19': _get_dhp19_model
    }
    return switcher[model_name](**params)


def _get_mobilenetv2(n_channels, n_classes, pretrained=False):
    cnn = models.mobilenet_v2(pretrained)
    if n_channels != 3:
        cnn.features[0][0] = torch.nn.Conv2d(n_channels, 32, kernel_size=(3, 3),
                                             stride=(2, 2), padding=(1, 1), bias=False)

    num_ftrs = cnn.classifier[-1].in_features
    cnn.classifier[-1] = nn.Linear(num_ftrs, n_classes, bias=True)
    return cnn


def _get_resnet(resnet, n_channels, n_classes, pretrained=False):

    cnn = getattr(models, resnet)(pretrained)

    if n_channels != 3:
        cnn.conv1 = torch.nn.Conv2d(n_channels, 64, kernel_size=(7, 7),
                                    stride=(2, 2), padding=(3, 3), bias=False)

    num_ftrs = cnn.fc.in_features
    cnn.fc = nn.Linear(num_ftrs, n_classes)

    return cnn


def _get_unet_resnet(resnet, n_channels, n_classes, pretrained=False, encoder_depth=3):
    encoder_weights = 'imagenet' if pretrained else None
    encoder_depth = 3
    decoder_channels = tuple([16 * (2 ** i) for i in reversed(range(0, int(encoder_depth)))])
    model: smp.Unet = smp.Unet(resnet, encoder_weights=encoder_weights,
                               encoder_depth=encoder_depth,
                               decoder_channels=decoder_channels,
                               classes=n_classes, activation=None)

    model.encoder.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7),
                                    stride=(2, 2), padding=(3, 3), bias=False)
    model.segmentation_head[-1] = FlatSoftmax()

    return model


def _get_dhp19_model(n_channels, n_classes):
    return DHP19Model(n_channels, n_classes)
