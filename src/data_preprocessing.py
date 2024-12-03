# data_preprocessing.py

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        self.mask_paths = [os.path.join(mask_dir, img) for img in os.listdir(mask_dir)]
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        
        return img, mask

def get_datasets(image_dir, mask_dir):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    dataset = ISICDataset(image_dir, mask_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset
