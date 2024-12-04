# src/data_preprocessing.py

import os
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import torch

class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, img) for img in os.listdir(mask_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        img = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        # Apply transformations
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask

def get_datasets(image_dir, mask_dir, image_size=256, train_ratio=0.8):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Create dataset
    dataset = MedicalImageDataset(image_dir, mask_dir, transform=transform)

    # Split dataset into training and validation sets
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset

if __name__ == "__main__":
    image_dir = '../data/images'
    mask_dir = '../data/masks'
    train_dataset, val_dataset = get_datasets(image_dir, mask_dir)

    # Optionally, save datasets (not commonly done with PyTorch Datasets)
    # torch.save(train_dataset, '../data/train_dataset.pt')
    # torch.save(val_dataset, '../data/val_dataset.pt')
