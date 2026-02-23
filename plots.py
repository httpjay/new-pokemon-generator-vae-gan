# plots.py
import matplotlib.pyplot as plt
import os

def plot_vae_losses(train_losses, val_losses, save_path='./results/vae.png'):
    """
    Plot VAE training and validation losses.

    Args:
        train_losses (dict): Dictionary with 'total', 'recon', 'kl' losses
        val_losses (dict): Dictionary with 'total', 'recon', 'kl' losses
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(train_losses['total']) + 1)

    # Total loss
    axes[0].plot(epochs, train_losses['total'], 'b-', label='Train')
    axes[0].plot(epochs, val_losses['total'], 'r-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Reconstruction loss
    axes[1].plot(epochs, train_losses['recon'], 'b-', label='Train')
    axes[1].plot(epochs, val_losses['recon'], 'r-', label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True)

    # KL divergence
    axes[2].plot(epochs, train_losses['kl'], 'b-', label='Train')
    axes[2].plot(epochs, val_losses['kl'], 'r-', label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"VAE loss plot saved to {save_path}")
    plt.close()


def plot_gan_losses(g_losses, d_losses, save_path='./results/gan.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    """
    Plot GAN training losses.

    Args:
        g_losses (list): Generator losses
        d_losses (list): Discriminator losses
        save_path (str): Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(g_losses) + 1)

    ax.plot(epochs, g_losses, 'b-', label='Generator Loss')
    ax.plot(epochs, d_losses, 'r-', label='Discriminator Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('GAN Training Losses')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"GAN loss plot saved to {save_path}")
    plt.close()
