import torch
import torch.nn as nn
from .unet import UNet

class DiffusionModel(nn.Module):
    def __init__(self, num_timesteps=1000):
        super(DiffusionModel, self).__init__()
        self.num_timesteps = num_timesteps
        self.unet = UNet()

    def forward(self, x, t):
        noise = torch.randn_like(x) * (t / self.num_timesteps)
        return self.unet(x + noise)