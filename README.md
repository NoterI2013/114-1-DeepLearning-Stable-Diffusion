# Deep Learning Cup 3: Text-to-Image Latent Diffusion Model

This project implements a **Latent Diffusion Model (LDM)** for text-to-image generation using TensorFlow/Keras. It combines a Pretrained VAE (Variational Autoencoder) for latent space compression and a UNet with Cross-Attention for the diffusion process.

## Project Structure

```
cup3/
├── cup3.py             # Main entry point for training and execution
├── input_pipeline.py   # Data loading, preprocessing, and augmentation
├── trainer.py          # Training loops (VAETrainer, LatentDiffusionTrainer) and Metrics (KID)
├── model/              # Model architectures
│   ├── vae.py          # VAE Encoder/Decoder definitions
│   ├── unet.py         # UNet architecture with Cross-Attention
│   ├── text_encoder.py # Text Encoder wrappers
│   └── common.py       # Shared layers and utilities
└── README.md           # Project documentation
```

## Features

- **Latent Diffusion**: Trains a diffusion model in the latent space of a VAE for efficiency.
- **Custom VAE**: Supports training/fine-tuning a VAE with GAN loss (PatchGAN Discriminator) and Perceptual Loss (LPIPS).
- **Text Conditioning**: Uses text embeddings to condition the generation process.
- **Evaluation**: Integrated KID (Kernel Inception Distance) metric for tracking generation quality during training.
- **Multi-GPU Support**: Includes hardware setup for GPU selection and memory growth.

## Requirements

- Python 3.x
- **TensorFlow** (2.x) & **Keras**
- NumPy
- Pandas
- Matplotlib
- tqdm

Install dependencies (example):
```bash
pip install tensorflow numpy pandas matplotlib tqdm
```

## Usage

### 1. Data Setup
The project expects a dataset directory structure at `./dataset` (configurable in `cup3.py`) containing:
- `text2ImgData.pkl`: Training data (image paths and text IDs).
- `dictionary/`: Vocabulary files (`id2Word.npy`, `word2Id.npy`).

### 2. Training
The main entry point is `cup3.py`. By default, it is configured to run the **Diffusion Training** loop.

```bash
python cup3.py
```

**Workflow in `cup3.py`:**
1.  **Hardware Setup**: Configures GPU usage.
2.  **Preprocessing**: Checks/generates text embeddings (optional/commented out by default).
3.  **Data Loading**: Uses `input_pipeline.py` to create TensorFlow datasets with augmentation.
4.  **Model Loading**:
    - Downloads/Loads pretrained weights for VAE Encoder/Decoder (from HuggingFace).
    - Freezes the VAE.
    - Initializes the UNet.
5.  **Training**: Runs `LatentDiffusionTrainer` which handles the noise schedule, denoising steps, and EMA (Exponential Moving Average) updates.
6.  **Checkpoints**: Saves model weights based on the best KID score.

### 3. VAE Training (Optional)
If you need to train the VAE from scratch or fine-tune it:
- Uncomment `run_vae_training(vae)` in `cup3.py`.
- This uses `VAETrainer` in `trainer.py` which combines Reconstruction Loss, KL Divergence, Perceptual Loss, and GAN Loss.

## Monitoring
- **Console**: Progress bars via `tqdm` and Keras default output.
- **Visualizations**: The trainer saves/plots generated images at the end of epochs to visualize progress.
- **Metrics**: Tracks `loss`, `recon_loss`, `gen_loss`, `disc_loss`, and `kid`.

## Core Components

### `trainer.py`
- **`VAETrainer`**: Custom Keras Model for VAE training. Implements the complex loss function (Recon + KL + LPIPS + GAN).
- **`LatentDiffusionTrainer`**: Custom Keras Model for diffusion. Handles:
    - `diffusion_schedule`: Cosine schedule for noise/signal rates.
    - `train_step`: Adds noise to latents and trains UNet to predict noise/image.
    - `generate_images`: Runs the reverse diffusion process (sampling).

### `input_pipeline.py`
- Handles text-to-ID conversion.
- Performs image augmentation:
    - Random Resizing/Cropping
    - random Flip
    - Brightness/Contrast adjustment
- Creates efficient `tf.data.Dataset` pipelines.