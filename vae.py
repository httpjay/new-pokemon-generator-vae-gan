# vae.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import save_grid


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: (B,3,64,64) -> (B,256,4,4)
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # 32x32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 16x16
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), # 8x8
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),# 4x4
            nn.ReLU(True),
        )

        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        # Decoder: latent -> (B,3,64,64)
        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 8x8
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16x16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 32x32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # 64x64
            nn.Sigmoid()  # output [0,1]
        )

    def encode(self, x):
        h = self.enc(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)          # IMPORTANT: real epsilon
        return mu + std * eps

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), 256, 4, 4)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1.0):
    # BCE over pixels
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    # KL
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
    return recon_loss + beta * kl, recon_loss, kl


def train_vae(model, loader, device, epochs=10, lr=2e-4, beta=1.0, save_dir="results_vae"):
    os.makedirs(save_dir, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total, total_rec, total_kl = 0.0, 0.0, 0.0

        for x in loader:
            x = x.to(device)

            recon, mu, logvar = model(x)
            loss, rec, kl = vae_loss(recon, x, mu, logvar, beta=beta)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            total_rec += rec.item()
            total_kl += kl.item()

        n = len(loader.dataset)
        print(f"[VAE] Epoch {epoch}/{epochs}  loss={total/n:.4f}  rec={total_rec/n:.4f}  kl={total_kl/n:.4f}")

        # save recon samples
        if epoch == 1 or epoch % 10 == 0:
            with torch.no_grad():
                sample = recon[:64]
                save_grid(sample, f"{save_dir}/recon_epoch_{epoch}.png", nrow=8, denorm=False)

                z = torch.randn(64, model.latent_dim, device=device)
                gen = model.decode(z)
                save_grid(gen, f"{save_dir}/gen_epoch_{epoch}.png", nrow=8, denorm=False)

       