import torch
import torch.nn as nn
from models.diffusion.unet import UNet

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.unet = UNet(in_channels=3, out_channels=3)

    def forward(self, x):
        return self.unet(x)