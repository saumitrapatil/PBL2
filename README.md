# Low-Light Image Enhancement using Hybrid Diffusion-GAN Model

This project implements a hybrid deep learning approach for low-light image enhancement, combining the strengths of Diffusion Models and Generative Adversarial Networks (GANs). The system takes low-light images as input and produces high-quality enhanced images with improved visibility and detail preservation.

## Features

- Two-stage enhancement pipeline:
  1. Initial enhancement using a Diffusion Model
  2. Refinement using a GAN-based approach
- High-quality image enhancement with preserved details
- Support for batch processing
- Comprehensive evaluation metrics (PSNR, SSIM, MSE)
- GPU acceleration support

## Project Structure

```
.
├── data/
│   └── dataset.py          # Dataset loading and preprocessing
├── models/
│   ├── diffusion/          # Diffusion model implementation
│   └── gan/                # GAN model implementation
├── training/
│   └── train_hybrid.py     # Training script for hybrid model
├── test.py                 # Testing and evaluation script
└── requirements.txt        # Project dependencies
```

## Requirements

- Python 3.10+
- TensorFlow 2.17.0
- PyTorch
- CUDA 12.4 (for GPU acceleration)
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a conda environment using the provided requirements:
```bash
conda create --name <env-name> --file requirements.txt
conda activate <env-name>
```

## Usage

### Training

To train the hybrid model:

```bash
python training/train_hybrid.py
```

The training process will:
1. Load and preprocess the dataset
2. Train the diffusion model
3. Train the GAN model
4. Save checkpoints periodically

### Testing

To test the model on a single image:

```bash
python test.py --input <path-to-input-image> --output <path-to-save-enhanced-image>
```

The script will:
1. Load the trained models
2. Process the input image
3. Generate the enhanced output
4. Display evaluation metrics
5. Save the enhanced image

## Evaluation Metrics

The model performance is evaluated using:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MSE (Mean Squared Error)
- Discriminator scores

## Dataset

The model is trained on the LOL (Low-Light) dataset, which contains pairs of low-light and corresponding normal-light images. The dataset should be organized as follows:

```
data/
└── lol_dataset/
    ├── our485/
    │   ├── low/    # Low-light images
    │   └── high/   # Normal-light images
    └── eval15/
        ├── low/    # Low-light images for evaluation
        └── high/   # Normal-light images for evaluation
```

## Model Architecture

### Diffusion Model
- Input: Low-light image (256x256x3)
- Output: Initial enhanced image
- Architecture: Custom CNN with multiple convolutional layers

### GAN Model
- Generator: U-Net based architecture
- Discriminator: PatchGAN architecture
- Loss functions: Binary cross-entropy and L1 loss
