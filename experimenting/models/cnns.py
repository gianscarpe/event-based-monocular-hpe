import torch
import torchvision
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from .custom import FlatSoftmax
            
def get_cnn(model_name, params):
    switcher = {
        'resnet18': _get_resnet18,
        'resnet34': _get_resnet34,
        'unet_resnet18' : lambda **args: _get_unet_resnet('resnet18', **args),
        'unet_resnet34' : lambda **args: _get_unet_resnet('resnet34', **args),
        'dhp19' : _get_dhp19_model
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


def _get_unet_resnet(resnet, n_channels, n_classes, pretrained=False, encoder_depth=3):
    encoder_weights = 'imagenet' if pretrained else None
    model : smp.Unet = smp.Unet(resnet, encoder_weights=encoder_weights,
                                encoder_depth=encoder_depth,
                                decoder_channels=(64, 32, 16),
                                classes=n_classes,activation=None)
    
    model.encoder.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7),
                                    stride=(2, 2), padding=(3, 3), bias=False)
    model.segmentation_head[-1] = nn.ReLU()
                            
    #model.segmentation_head[-1] = nn.Sequential(nn.Conv2d(n_classes, n_classes,
    #                                kernel_size=1),
    #                                            nn.ReLU())
    
    return model



    

class DHP19Model(nn.Module):
    def __init__(self, n_channels, n_joints):
        super(DHP19Model, self).__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.block1 = nn.Sequential(nn.Conv2d(in_channels=n_channels,
                                              out_channels=16, kernel_size=3,
                                              padding=1),
                                    nn.LeakyReLU())
        


        self.block2 = nn.Sequential(nn.Conv2d(in_channels=16,
                                              out_channels=32, kernel_size=3,
                                              padding=1),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(in_channels=32,
                                              out_channels=32, kernel_size=3,
                                              padding=1),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(in_channels=32,
                                              out_channels=32, kernel_size=3,
                                              padding=1),
                                    nn.LeakyReLU())

        self.block3 = nn.Sequential(nn.Conv2d(in_channels=32,
                                              out_channels=64, kernel_size=3,
                                              padding=1),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(in_channels=64,
                                              out_channels=64, kernel_size=3,
                                              padding=1),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(in_channels=64,
                                              out_channels=64, kernel_size=3,
                                              padding=1),
                                    nn.LeakyReLU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(in_channels=64,
                                                    out_channels=32,
                                                    stride=2,
                                                    padding=1,
                                                    output_padding=1,
                                                    kernel_size=3),
                                 nn.LeakyReLU())
        self.block4 = nn.Sequential(nn.Conv2d(in_channels=32,
                                              out_channels=32, kernel_size=3,
                                              padding=1),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(in_channels=32,
                                              out_channels=32, kernel_size=3,
                                              padding=1),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(in_channels=32,
                                              out_channels=32, kernel_size=3,
                                              padding=1),
                                    nn.LeakyReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(in_channels=32,
                                                    out_channels=16,
                                                    stride=2,
                                                    padding=1,
                                                    output_padding=1,
                                                    kernel_size=3),
                                 nn.LeakyReLU())
        
        self.block5 = nn.Sequential(nn.Conv2d(in_channels=16,
                                              out_channels=16, kernel_size=3,
                                              padding=1),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(in_channels=16,
                                              out_channels=16, kernel_size=3,
                                              padding=1),
                                    nn.LeakyReLU())
        self.head = nn.Sequential(nn.Conv2d(in_channels=16,
                                              out_channels=n_joints, kernel_size=3,
                                              padding=1),
                                    nn.ReLU())
    def forward(self, x):

        x1 = self.block1(x)
        x = self.max_pool(x1)
        x2 = self.block2(x)
        x = self.max_pool(x2)
        x3 = self.block3(x)
        
        x4 = self.up1(x3)
        x = x2 + x4
        x5 = self.block4(x)
        x6 = self.up2(x5)
        x = x6 + x1
        x7 = self.block5(x)
        out = self.head(x7)
        return out

def _get_dhp19_model(n_channels, n_classes):
    return DHP19Model(n_channels, n_classes)
    

