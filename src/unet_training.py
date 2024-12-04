# src/unet_training.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from neural_networks import UNet
from data_preprocessing import get_datasets
from tqdm import tqdm
import os

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = 1 - ((2. * intersection + smooth) /
                (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    return loss.mean()

def train_unet(epochs=50, batch_size=8, device='cuda'):
    # Get datasets
    train_dataset, val_dataset = get_datasets('../data/images', '../data/masks')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs('../results/segmentation_outputs', exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, masks in tqdm(train_loader):
            inputs = inputs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, masks in val_loader:
                inputs = inputs.to(device)
                masks = masks.to(device)
                outputs = model(inputs)
                loss = dice_loss(outputs, masks)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')

        # Save model checkpoint
        torch.save(model.state_dict(), f'../models/unet_epoch_{epoch+1}.pth')

    # Save final model
    torch.save(model.state_dict(), '../models/unet_final.pth')

if __name__ == "__main__":
    train_unet()
