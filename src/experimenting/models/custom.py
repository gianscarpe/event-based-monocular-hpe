import torch
import torch.nn as nn
from operator import mul
from functools import reduce
from kornia.geometry import spatial_softmax2d

class FlatSoftmax(nn.Module):
    def __init__(self):
        super(FlatSoftmax, self).__init__()

    def forward(self, inp):
        return spatial_softmax2d(inp)
