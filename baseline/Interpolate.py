import torch
from torch import nn
from torch.nn import functional as F


class Interpolate(nn.Module):
    def __init__(self, scale_factor=4, mode='bilinear', align_corners=False):
        super(Interpolate, self).__init__()
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, 
                          align_corners=self.align_corners)
        
        return x