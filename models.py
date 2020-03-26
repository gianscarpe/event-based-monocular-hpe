import torch
import torchvision
from torchvision import models
import torch.nn as nn

def vgg19(n_channels, num_classes, pretrained=False):
    cnn = models.vgg19(pretrained)
    
    if n_channels != 3:        
        cnn.features[0] = torch.nn.Conv2d(n_channels, 64, kernel_size=(3, 3),
                                    stride=(1, 1), padding=(1, 1), bias=False)
    num_ftrs = cnn.classifier[-1].in_features
    cnn.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    return cnn


def resnet50(n_channels, num_classes, pretrained=False):

    cnn = models.resnet50(pretrained)

    if n_channels != 3:
        cnn.conv1 = torch.nn.Conv2d(n_channels, 64, kernel_size=(7, 7),
                                          stride=(2, 2), padding=(3, 3), bias=False)

    num_ftrs = cnn.fc.in_features
    cnn.fc = nn.Linear(num_ftrs, num_classes)

    return cnn

def resnet18(n_channels, num_classes, pretrained=False):

    cnn = models.resnet18(pretrained)

    if n_channels != 3:
        cnn.conv1 = torch.nn.Conv2d(n_channels, 64, kernel_size=(7, 7),
                                          stride=(2, 2), padding=(3, 3), bias=False)

    num_ftrs = cnn.fc.in_features
    cnn.fc = nn.Linear(num_ftrs, num_classes)

    return cnn

    
