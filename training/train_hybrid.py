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
import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing Device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"CUDA Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.cuda.empty_cache()

    dataset = LowLightDataset(
        low_light_dir=os.path.abspath("../data/lol_dataset/our485/low"),
        normal_dir=os.path.abspath("../data/lol_dataset/our485/high")
    )
    
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=2, 
        pin_memory=True if torch.cuda.is_available() else False
    )

    diffusion_model = DiffusionModel().to(device)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    optimizer_diffusion = optim.Adam(diffusion_model.parameters(), lr=1e-4)
    optimizer_gan_g = optim.Adam(generator.parameters(), lr=2e-4)
    optimizer_gan_d = optim.Adam(discriminator.parameters(), lr=2e-4)
    
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    
    if torch.cuda.is_available():
        print("\nPre-warming CUDA Kernels...")
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256).to(device)
            _ = diffusion_model(dummy_input, torch.tensor(500).to(device))
            _ = generator(dummy_input)
            _ = discriminator(dummy_input)
        print("CUDA Kernels Pre-warmed!\n")
    
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(10):
        print(f"\nStarting Epoch {epoch+1}")
        epoch_loss = 0.0

        for batch_idx, (low_light, normal) in enumerate(dataloader):
            low_light, normal = low_light.to(device), normal.to(device)
            
            diffusion_output = diffusion_model(low_light, torch.tensor(500).to(device))
            gen_output = generator(diffusion_output)
            
            real_labels = torch.ones_like(discriminator(normal))
            fake_labels = torch.zeros_like(discriminator(gen_output.detach()))
            
            d_real_loss = bce_loss(discriminator(normal), real_labels)
            d_fake_loss = bce_loss(discriminator(gen_output.detach()), fake_labels)
            d_loss = (d_real_loss + d_fake_loss) / 2
            
            optimizer_gan_d.zero_grad()
            d_loss.backward()
            optimizer_gan_d.step()
            
            g_loss = mse_loss(gen_output, normal) + bce_loss(discriminator(gen_output), real_labels)
            optimizer_gan_g.zero_grad()
            g_loss.backward()
            optimizer_gan_g.step()
            
            torch.cuda.synchronize()
            epoch_loss += (d_loss.item() + g_loss.item())

            if batch_idx % 60 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Diffusion Loss: {d_loss:.4f} | GAN Loss: {g_loss:.4f}")
                print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"CUDA Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'diffusion_model_state_dict': diffusion_model.state_dict(),
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_diffusion_state_dict': optimizer_diffusion.state_dict(),
            'optimizer_gan_g_state_dict': optimizer_gan_g.state_dict(),
            'optimizer_gan_d_state_dict': optimizer_gan_d.state_dict()
        }, checkpoint_path)
        print(f"Model saved at {checkpoint_path}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_path = os.path.join(save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'diffusion_model_state_dict': diffusion_model.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_diffusion_state_dict': optimizer_diffusion.state_dict(),
                'optimizer_gan_g_state_dict': optimizer_gan_g.state_dict(),
                'optimizer_gan_d_state_dict': optimizer_gan_d.state_dict()
            }, best_model_path)
            print(f"    Best model updated at {best_model_path}")

        print(f"Epoch {epoch+1} Completed! Diffusion Loss: {d_loss:.4f} | GAN Loss: {g_loss:.4f}")
