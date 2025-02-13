import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import LowLightDataset
from models.diffusion.diffusion_model import DiffusionModel
from models.gan.generator import Generator
from models.gan.discriminator import Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load dataset
dataset = LowLightDataset(low_light_dir="../data/lol_dataset/our485/low", normal_dir="../data/lol_dataset/our485/high")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# Initialize models
diffusion_model = DiffusionModel().to(device)
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_diffusion = optim.Adam(diffusion_model.parameters(), lr=1e-4)
optimizer_gan_g = optim.Adam(generator.parameters(), lr=2e-4)
optimizer_gan_d = optim.Adam(discriminator.parameters(), lr=2e-4)

# Losses
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()

# Training Loop
for epoch in range(10):
    for low_light, normal in dataloader:  
        low_light, normal = low_light.to(device), normal.to(device)

        # Stage 1: Diffusion Model
        diffusion_output = diffusion_model(low_light, torch.tensor(500).to(device))

        # Stage 2: GAN Enhancement
        gen_output = generator(diffusion_output)
        
        # Train Discriminator
        real_labels = torch.ones_like(discriminator(normal))
        fake_labels = torch.zeros_like(discriminator(gen_output.detach()))

        d_real_loss = bce_loss(discriminator(normal), real_labels)
        d_fake_loss = bce_loss(discriminator(gen_output.detach()), fake_labels)
        d_loss = (d_real_loss + d_fake_loss) / 2
        optimizer_gan_d.zero_grad()
        d_loss.backward()
        optimizer_gan_d.step()

        # Train Generator
        g_loss = mse_loss(gen_output, normal) + bce_loss(discriminator(gen_output), real_labels)
        optimizer_gan_g.zero_grad()
        g_loss.backward()
        optimizer_gan_g.step()

    print(f"Epoch {epoch+1}: Diffusion Loss={d_loss:.4f}, GAN Loss={g_loss:.4f}")
