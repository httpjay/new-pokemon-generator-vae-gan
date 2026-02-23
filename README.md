# Pokémon Image Generation using VAE and GAN

This project implements a Variational Autoencoder (VAE) and a Generative Adversarial Network (GAN) from scratch in PyTorch to generate 64x64 Pokémon-style images.

## 🚀 Features
- Custom PyTorch implementation of:
  - DCGAN with spectral normalization
  - Variational Autoencoder with β-VAE support
- Apple Silicon GPU acceleration using MPS backend
- Data augmentation for small dataset stabilization
- TTUR (Two Time-Scale Update Rule)
- Instance noise for GAN stabilization

## 📂 Dataset
The models are trained on the **Complete Pokémon Image Dataset** containing ~2,500 high-quality images.
- **Source:** [Pokémon Images Dataset (Kaggle)](https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset)
- **Format:** 64x64 RGB Images
- **Structure:** Images are organized into generational subfolders (Gen1, Gen2, etc.), which are parsed recursively by the data loader.

## 🏗️ Project Structure
```text
Pokemon_VAE_GAN/
├── main.py            # Main entry point for training
├── vae.py             # VAE architecture & loss function
├── gan.py             # GAN (Generator/Discriminator) architecture
├── utils.py           # Data loaders & image saving utilities
├── results_vae/       # Reconstructions & training progress images
├── results_gan/       # Generated Pokémon samples
├── models/            # Saved .pth model weights
├── requirements.txt   # List of dependencies
├── README.md          # Project documentation
└── .gitignore         # Files excluded from GitHub (weights/data)
```

## 🧠 Models

### VAE
- Encoder → μ and logσ²
- Reparameterization trick
- Decoder with Sigmoid output
- β-VAE loss

### GAN
- Generator with transposed convolutions
- Discriminator with spectral normalization
- BCEWithLogitsLoss
- Label smoothing
- Extra generator step
- Instance noise

## 📈 Results

GAN samples after 200 epochs:

![GAN Sample](results_gan/sample_example.png)

VAE samples:

![VAE Sample](results_vae/sample_example.png)

## ⚙️ Run

```bash
python main.py
