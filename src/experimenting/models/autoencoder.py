import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import init_parameters, get_cnn


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, pretrained, latent_size):
        super(AutoEncoder, self).__init__()

        self.in_channels = in_channels
        self.dec_channels = 32
        
        self.in_cnn, self.mid_feature_dimension = get_backbone(in_channels,
                                                               pretrained)
        
        self.latent_size = latent_size

        self.encoder_head = nn.Linear(self.mid_feature_dimension * 32 * 32,
                                      latent_size)

        ###############
        # DECODER
        ##############

        self.d_fc_1 = nn.Linear(latent_size, self.dec_channels * 16 * 32 * 32) 

        self.d_conv_1 = nn.Conv2d(self.dec_channels * 16,
                                  self.dec_channels * 8,
                                  kernel_size=(4, 4),
                                  stride=(1, 1),
                                  padding=0)
        self.d_bn_1 = nn.BatchNorm2d(self.dec_channels * 8)

        self.d_conv_2 = nn.Conv2d(self.dec_channels * 8,
                                  self.dec_channels * 4,
                                  kernel_size=(4, 4),
                                  stride=(1, 1),
                                  padding=0)
        self.d_bn_2 = nn.BatchNorm2d(self.dec_channels * 4)

        self.d_conv_3 = nn.Conv2d(self.dec_channels * 4,
                                  1,
                                  kernel_size=(4, 4),
                                  stride=(1, 1),
                                  padding=0)
        
        init_parameters(self)

    def encode(self, x):
        x = self.in_cnn(x)

        x = x.view(-1, self.mid_feature_dimension * 32 * 32)
        x = self.encoder_head(x)
        return x

    def decode(self, x):

        # h1
        #x = x.view(-1, self.latent_size, 1, 1)
        x = self.d_fc_1(x)

        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = x.view(-1, self.dec_channels * 16, 32, 32)

        # h2
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_1(x)

        # h3
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_2(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_2(x)

        # out
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_3(x)
        x = torch.sigmoid(x)

        return x

    def forward(self, x):
        
        z = self.encode(x)
        decoded = self.decode(z)
        return decoded


def get_backbone(n_channels, pretrained):
    resnet = get_cnn('resnet34', {
        'n_channels': n_channels,
        'n_classes': 1,
        'pretrained': pretrained
    })

    net = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
    )
    return net, 128
