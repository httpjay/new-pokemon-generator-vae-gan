# gan.py
import os
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from utils import save_grid

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        # Project latent vector to 4x4 feature map
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)

        self.net = nn.Sequential(
            # Layer 1: 4x4 -> 8x8 (Switch to Upsample to avoid checkerboard)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            # Layer 2: 8x8 -> 16x16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            # Layer 3: 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            # Layer 4: 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh() # Output pixels in [-1, 1]
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 512, 4, 4)
        return self.net(h)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Spectral Norm helps stabilize training on small datasets
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),  # 32x32
            nn.LeakyReLU(0.2, True),
            
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), # 16x16
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),# 8x8
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),# 4x4
            nn.LeakyReLU(0.2, True),
        )
        self.fc = spectral_norm(nn.Linear(512 * 4 * 4, 1))

    def forward(self, x):
        h = self.net(x).view(x.size(0), -1)
        return self.fc(h)  # Output logits


def train_gan(G, D, loader, device, epochs=300, lr=2e-4, latent_dim=100, save_dir="results_gan"):
    os.makedirs(save_dir, exist_ok=True)
    # Standard GAN hyperparameters: betas=(0.5, 0.999)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        loss_g_total, loss_d_total = 0.0, 0.0

        for real in loader:
            real = real.to(device)
            bsz = real.size(0)

            # ----- Train Discriminator -----
            opt_d.zero_grad()
            
            # Real images with label smoothing (0.9)
            real_labels = torch.full((bsz, 1), 0.9, device=device)
            # Add small instance noise to real images to help stability
            real_noisy = real + 0.02 * torch.randn_like(real)
            out_real = D(real_noisy)
            loss_d_real = bce(out_real, real_labels)

            # Fake images
            z = torch.randn(bsz, latent_dim, device=device)
            fake = G(z).detach() # Detach so we don't update G here
            fake_labels = torch.zeros((bsz, 1), device=device)
            out_fake = D(fake)
            loss_d_fake = bce(out_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            # ----- Train Generator (1:1 Ratio) -----
            opt_g.zero_grad()
            
            z = torch.randn(bsz, latent_dim, device=device)
            fake = G(z)
            # We want D to think these are real (1.0)
            fool_labels = torch.ones((bsz, 1), device=device)
            out_g = D(fake)
            loss_g = bce(out_g, fool_labels)

            loss_g.backward()
            opt_g.step()

            loss_d_total += loss_d.item()
            loss_g_total += loss_g.item()

        avg_d = loss_d_total / len(loader)
        avg_g = loss_g_total / len(loader)
        print(f"[GAN] Epoch {epoch}/{epochs}  D_loss={avg_d:.4f}  G_loss={avg_g:.4f}")

        # Save samples periodically
        if epoch == 1 or epoch % 10 == 0:
            with torch.no_grad():
                G.eval()
                z = torch.randn(64, latent_dim, device=device)
                samples = G(z)
                save_grid(samples, f"{save_dir}/samples_epoch_{epoch}.png", nrow=8, denorm=True)
                G.train()