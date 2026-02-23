# utils.py
import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from torchvision.utils import save_image




class ImageOnlyFolder(Dataset):
    def __init__(self, folder: str, transform=None):
        self.folder = folder
        self.transform = transform
        # Fixed: Added "*" wildcard for glob extensions
        valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
        self.files = []
        for ext in valid_exts:
            pattern = os.path.join(folder, "**", ext)
            self.files.extend(glob.glob(pattern, recursive=True))

        if len(self.files) == 0:
            raise ValueError(f"No image files found in: {folder}")
        print(f"Dataset initialized: {len(self.files)} images found.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

def vae_dataloader(images_dir: str, batch_size=64, num_workers=0):
    """
    VAE expects pixels in [0,1] because decoder ends with Sigmoid().
    """
    tf = transforms.Compose([
        transforms.Resize((72, 72)),
        transforms.RandomCrop((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),  # LAST for VAE (no Normalize)
    ])
    ds = ImageOnlyFolder(images_dir, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

def gan_dataloader(images_dir: str, batch_size=64,num_workers=0):
    tf = transforms.Compose([
        transforms.Resize((72, 72)),
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # New: Helps the GAN generalize shapes
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Crucial for Tanh
    ])
    ds = ImageOnlyFolder(images_dir, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers ,drop_last=True)

def save_grid(images: torch.Tensor, path: str, nrow: int = 8, denorm: bool = False):
    """
    Save a grid of images.
    denorm=True converts [-1,1] back to [0,1] for saving.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imgs = images.detach().cpu()
    if denorm:
        imgs = (imgs + 1) / 2
        imgs = imgs.clamp(0, 1)
    save_image(imgs, path, nrow=nrow)