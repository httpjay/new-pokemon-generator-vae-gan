# main.py
import os
import torch
from utils import vae_dataloader, gan_dataloader, save_grid
from vae import VAE, train_vae
from gan import Generator, Discriminator, train_gan
from plots import plot_vae_losses, plot_gan_losses


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    images_dir = "pokemon-images-and-types/images"

    os.makedirs("models", exist_ok=True)
    os.makedirs("debug", exist_ok=True)
    os.makedirs("results_vae", exist_ok=True)
    os.makedirs("results_gan", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    debug_loader = gan_dataloader(images_dir, batch_size=64, num_workers=0)
    x = next(iter(debug_loader))  # x is in [-1,1]
    save_grid(x[:64], "debug/real_batch_gan_denorm.png", denorm=True)
    print("Debug batch shape:", x.shape)
    print("Debug min/max:", x.min().item(), x.max().item())
    print("Saved: debug/real_batch_gan_denorm.png")

    # # ---- VAE ----
    # vae_loader = vae_dataloader(images_dir, batch_size=64, num_workers=0)
    # vae = VAE(latent_dim=128).to(device)

    # vae_path = "models/vae_pokemon.pth"
    # if os.path.exists(vae_path):
    #     print(f"✅ Found existing VAE. Resuming from {vae_path}...")
    #     vae.load_state_dict(torch.load(vae_path, map_location=device))

    # vae_train_hist = train_vae(
    #     vae, vae_loader,
    #     device=device,
    #     epochs=200,
    #     lr=2e-4,
    #     beta=0.1,
    #     save_dir="results_vae"
    # )
    # torch.save(vae.state_dict(), "models/vae_pokemon.pth")
    # plot_vae_losses(train_losses=vae_train_hist, val_losses=vae_train_hist, save_path='./results/vae.png')

    # ---- GAN ----
    gan_loader = gan_dataloader(images_dir, batch_size=64, num_workers=0)
    G = Generator(latent_dim=100).to(device)
    D = Discriminator().to(device)

    g_path = "models/gan_G_pokemon.pth"
    d_path = "models/gan_D_pokemon.pth"
    if os.path.exists(g_path) and os.path.exists(d_path):
        print(f"✅ Found existing GAN models. Resuming training...")
        G.load_state_dict(torch.load(g_path, map_location=device))
        D.load_state_dict(torch.load(d_path, map_location=device))

    g_hist, d_hist = train_gan(
        G, D, gan_loader,
        device=device,
        epochs=300,
        lr=1e-4,
        latent_dim=100,
        save_dir="results_gan"
    )

    plot_gan_losses(g_hist, d_hist, save_path='./results/gan.png')
    torch.save(G.state_dict(), "models/gan_G_pokemon.pth")
    torch.save(D.state_dict(), "models/gan_D_pokemon.pth")

    print("Training Complete! All models saved in /models")


if __name__ == "__main__":
    main()