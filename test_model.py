import torch
import cv2
from models.diffusion.diffusion_model import DiffusionModel
from models.gan.generator import Generator
from utils.utils import save_image

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained models
diffusion_model = DiffusionModel().to(device)
generator = Generator().to(device)

# Load pre-trained weights
diffusion_model.load_state_dict(torch.load("training/checkpoints/diffusion_model.pth", map_location=device))
generator.load_state_dict(torch.load("training/checkpoints/generator.pth", map_location=device))

# Set models to evaluation mode
diffusion_model.eval()
generator.eval()

def preprocess_image(image_path):
    """Load and preprocess the input image."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
    return img

def enhance_image(image_path, output_path):
    """Apply Hybrid Enhancement (Diffusion + GAN) to a low-light image."""
    with torch.no_grad():
        low_light_img = preprocess_image(image_path)

        # Step 1: Diffusion Model Enhancement
        diffusion_output = diffusion_model(low_light_img, torch.tensor(500).to(device))  # Timestep = 500

        # Step 2: GAN Refinement
        enhanced_image = generator(diffusion_output)

        # Save the final enhanced image
        save_image(enhanced_image, output_path)

# Test image path
test_image = "data/test/low/test1.jpg"  # Modify path accordingly
output_image = "results/test1_enhanced.jpg"

# Run enhancement
enhance_image(test_image, output_image)

print(f"Enhanced image saved at: {output_image}")