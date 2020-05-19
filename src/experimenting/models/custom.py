import torch
import torch.nn as nn
from operator import mul
from functools import reduce

class FlatSoftmax(nn.Module):
    def __init__(self):
        super(FlatSoftmax, self).__init__()

    def forward(self, inp):
        """Compute the softmax with all but the first two tensor dimensions
        combined.
        https://github.com/anibali/margipose/blob/a9dbe5c3151d7a7e071df6275d5702c07ef5152d/src/margipose/dsntnn.py#L123
        """

        orig_size = inp.size()
        flat = inp.view(-1, reduce(mul, orig_size[2:]))
        flat = torch.nn.functional.softmax(flat, -1)
        return flat.view(*orig_size)
