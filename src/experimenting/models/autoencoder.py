import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_feature_extractor, init_parameters


class AutoEncoder(nn.Module):

    def __init__(self, in_channels, backbone_path, latent_size):
        super(AutoEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.in_cnn, self.mid_feature_dimension = self.get_feature_extractor(
            backbone_path)
        self.dec_channels = dec_channels
        self.latent_size = latent_size
       
        self.encoder_head = nn.Linear(self.mid_feature_dimension, latent_size)

        ###############
        # DECODER
        ##############
        
        self.d_fc_1 = nn.Linear(latent_size, self.mid_feature_dimension)

        self.d_conv_1 = nn.Conv2d(dec_channels*16, dec_channels*8, 
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        self.d_bn_1 = nn.BatchNorm2d(dec_channels*8)

        self.d_conv_2 = nn.Conv2d(dec_channels*8, dec_channels*4, 
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        self.d_bn_2 = nn.BatchNorm2d(dec_channels*4)

        self.d_conv_3 = nn.Conv2d(dec_channels*4, dec_channels*2, 
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        self.d_bn_3 = nn.BatchNorm2d(dec_channels*2)

        self.d_conv_4 = nn.Conv2d(dec_channels*2, dec_channels, 
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        self.d_bn_4 = nn.BatchNorm2d(dec_channels)
        
        self.d_conv_5 = nn.Conv2d(dec_channels, in_channels, 
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        
        init_parameters(self)

    def encode(self, x):
        x = self.in_cnn(x)
        #fc
        x = x.view(-1, self.dec_channels*16*4*4)
        x = self.encoder_head(x)
        return x

    def decode(self, x):
        
        # h1
        #x = x.view(-1, self.latent_size, 1, 1)
        x = self.d_fc_1(x)
        
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)  
        x = x.view(-1, self.dec_channels*16, 4, 4) 

        
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
        
        # h4
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_3(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_3(x)  

        # h5
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_4(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_4(x)
        
        
        # out
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_5(x)
        x = torch.sigmoid(x)
        
        return x

    def forward(self, x):
        z = self.encode(x)
        decoded = self.decode(z)
        return z, decoded
