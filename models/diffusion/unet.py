import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super(UNet, self).__init__()
        self.enc1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.enc2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1)
        self.enc3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1)
        
        self.dec3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, padding=1)
        self.dec2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 3, padding=1)
        self.dec1 = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(self, x):
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(F.avg_pool2d(x1, 2)))
        x3 = F.relu(self.enc3(F.avg_pool2d(x2, 2)))

        x3 = F.interpolate(F.relu(self.dec3(x3)), scale_factor=2)
        x2 = F.interpolate(F.relu(self.dec2(x3 + x2)), scale_factor=2)
        x1 = torch.tanh(self.dec1(x2 + x1))
        
        return x1