import torch
import torch.nn as nn
from blocks import ConvBlock


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.layers = nn.Sequential(
            # ConvBlock(3, 64,  kernel_size=9, padding=4),
            nn.Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=4),
            nn.Conv2d(3, 64, 256, kernel_size=(1, 1),stride=(1, 1), bias=False),
            nn.Conv2d()
            ConvBlock(64, 32, kernel_size=1, padding=0),
            ConvBlock(32, 3,  kernel_size=5, padding=2, activation=None))
    
    def forward(self, x):
        return self.layers(x)
