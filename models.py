import torch
import torchvision
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F



def mobilenetv2(n_channels, num_classes, pretrained=False):
    cnn = models.mobilenet_v2(pretrained)
    if n_channels != 3:        
        cnn.features[0][0] = torch.nn.Conv2d(n_channels, 32, kernel_size=(3, 3),
                                          stride=(2, 2), padding=(1, 1), bias=False)

    num_ftrs = cnn.classifier[-1].in_features
    cnn.classifier[-1] = nn.Linear(num_ftrs, num_classes, bias=True)
    return cnn

def vgg19(n_channels, num_classes, pretrained=False):
    cnn = models.vgg19(pretrained)
    
    if n_channels != 3:
        l = torch.nn.Conv2d(n_channels, 64, kernel_size=(3, 3),
                                    stride=(1, 1), padding=(1, 1), bias=False)
        nn.init.xavier_normal_(l.weight)
        
        cnn.features[0] = l
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

def resnet34(n_channels, num_classes, pretrained=False):

    cnn = models.resnet34(pretrained)

    if n_channels != 3:
        l = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.xavier_normal_(l.weight)
        
        cnn.conv1 = l

    num_ftrs = cnn.fc.in_features
    cnn.fc = nn.Linear(num_ftrs, num_classes)

    return cnn


def resnet18(n_channels, num_classes, pretrained=False):

    cnn = models.resnet18(pretrained)

    if n_channels != 3:
        l = torch.nn.Conv2d(n_channels, 64, kernel_size=(7, 7),
                                          stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(l.weight, mode='fan_in')
        
        cnn.conv1 = l
    num_ftrs = cnn.fc.in_features
    cnn.fc = nn.Linear(num_ftrs, num_classes)
    nn.init.kaiming_normal_(cnn.fc.weight, mode='fan_in')

    return cnn

def inceptionv3(n_channels, num_classes, pretrained=False):

    cnn = torchvision.models.inception_v3(pretrained)
    
    if n_channels != 3:
        cnn.Conv2d_1a_3x3.conv = torch.nn.Conv2d(n_channels, 32, kernel_size=(3, 3),
                                          stride=(2, 2), bias=False)

    num_ftrs = cnn.fc.in_features
    cnn.fc = nn.Linear(num_ftrs, num_classes)
    return cnn
  
    
def ownmodel1(n_channels, num_classes, pretrained=False):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.cnn1 = nn.Conv2d(in_channels=n_channels, out_channels=32,
                                  kernel_size=(3, 3), padding=1)
            self.bn1 = nn.BatchNorm2d()
            self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64,
                                  kernel_size=(3, 3), padding=1)
            self.bn2 = nn.BatchNorm2d()
            self.cnn3 = nn.Conv2d(in_channels=64, out_channels=128,
                                  kernel_size=(3,3), padding=1)
            self.bn3 = nn.BatchNorm2d()
            self.cnn4 = nn.Conv2d(in_channels=128, out_channels=256,
                                  kernel_size=(3,3), padding=1)
            

            self.max_pool2d = nn.MaxPool2d((2, 2))


            # 3 max pooling -> 224/16 = 14 * 256 * 256
            self.fc_in = 56 **2 * 64
            self.fc1 = nn.Linear(self.fc_in, 128, bias=True)
            self.fc2 = nn.Linear(128, num_classes, bias=True)

        def forward(self, x):
            x = F.relu(self.cnn1(x))
            x = self.max_pool2d(x)
            x = F.relu(self.cnn2(x))
            x = self.max_pool2d(x)
            
            x = x.view(-1, self.fc_in)

            x = self.fc1(x)
            out = self.fc2(x)
            return out
    cnn = Model()
    return cnn

            
                                  
