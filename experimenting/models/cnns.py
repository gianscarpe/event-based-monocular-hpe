import torch
import torchvision
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

            
def get_cnn(model_name, params):
    switcher = {
        'resnet18': _get_resnet18,
        'resnet34': _get_resnet34,
        'unet_resnet34' : _get_unet_resnet34
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

def _get_vgg19(n_channels, n_classes, pretrained=False):
    cnn = models.vgg19(pretrained)
    
    if n_channels != 3:
        l = torch.nn.Conv2d(n_channels, 64, kernel_size=(3, 3),
                            stride=(1, 1), padding=(1, 1), bias=False)
        nn.init.xavier_normal_(l.weight)
        
        cnn.features[0] = l
        num_ftrs = cnn.classifier[-1].in_features
        cnn.classifier[-1] = nn.Linear(num_ftrs, n_classes)

    return cnn


def _get_resnet50(n_channels, n_classes, pretrained=False):

    cnn = models.resnet50(pretrained)

    if n_channels != 3:
        cnn.conv1 = torch.nn.Conv2d(n_channels, 64, kernel_size=(7, 7),
                                    stride=(2, 2), padding=(3, 3), bias=False)

    num_ftrs = cnn.fc.in_features
    cnn.fc = nn.Linear(num_ftrs, n_classes)

    return cnn

def _get_resnet34(n_channels, n_classes, pretrained=False):

    cnn = models.resnet34(pretrained)

    if n_channels != 3:
        l = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.xavier_normal_(l.weight)
        
        cnn.conv1 = l

    num_ftrs = cnn.fc.in_features
    cnn.fc = nn.Linear(num_ftrs, n_classes)

    return cnn


def _get_resnet18(n_channels, n_classes, pretrained=False):

    cnn = models.resnet18(pretrained)

    if n_channels != 3:
        l = torch.nn.Conv2d(n_channels, 64, kernel_size=(7, 7),
                            stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(l.weight, mode='fan_in')
        
        cnn.conv1 = l
        num_ftrs = cnn.fc.in_features
        cnn.fc = nn.Linear(num_ftrs, n_classes)
        nn.init.kaiming_normal_(cnn.fc.weight, mode='fan_in')

    return cnn

def _get_inceptionv3(n_channels, n_classes, pretrained=False):

    cnn = torchvision.models.inception_v3(pretrained)
    
    if n_channels != 3:
        cnn.Conv2d_1a_3x3.conv = torch.nn.Conv2d(n_channels, 32, kernel_size=(3, 3),
                                                 stride=(2, 2), bias=False)

    num_ftrs = cnn.fc.in_features
    cnn.fc = nn.Linear(num_ftrs, n_classes)
    return cnn

def _get_unet_resnet34(n_channels, n_classes, pretrained=False):
    encoder_weights = 'imagenet' if pretrained else None
    model : smp.Unet = smp.Unet('resnet34', encoder_weights=encoder_weights,
                                classes=n_classes,activation=None)
    
    model.encoder.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7),
                                    stride=(2, 2), padding=(3, 3), bias=False)


    return model
        

