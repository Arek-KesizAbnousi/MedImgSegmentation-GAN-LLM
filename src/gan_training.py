# src/gan_training.py

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from neural_networks import Generator, Discriminator
from data_preprocessing import get_datasets
from tqdm import tqdm

def train_gan(epochs=100, batch_size=64, noise_dim=100, device='cuda'):
    # Get datasets (we only need images for GAN)
    train_dataset, _ = get_datasets('../data/images', '../data/masks')
    
    # We can ignore masks and only use images
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize models
    generator = Generator(noise_dim=noise_dim).to(device)
    discriminator = Discriminator().to(device)
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)
    
    criterion = torch.nn.BCELoss()
    
    os.makedirs('../results/generated_images', exist_ok=True)
    
    for epoch in range(epochs):
        for real_imgs, _ in tqdm(train_loader):  # We can ignore masks
            real_imgs = real_imgs.to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            noise = torch.randn(real_imgs.size(0), noise_dim, device=device)
            fake_imgs = generator(noise)
            
            real_labels = torch.ones(real_imgs.size(0), device=device)
            fake_labels = torch.zeros(real_imgs.size(0), device=device)
            
            outputs_real = discriminator(real_imgs)
            outputs_fake = discriminator(fake_imgs.detach())
            
            loss_real = criterion(outputs_real, real_labels)
            loss_fake = criterion(outputs_fake, fake_labels)
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            outputs = discriminator(fake_imgs)
            loss_G = criterion(outputs, real_labels)
            loss_G.backward()
            optimizer_G.step()
        
        # Save generated images
        with torch.no_grad():
            fake_imgs = fake_imgs[:25]
            fake_imgs = (fake_imgs + 1) / 2  # Rescale to [0,1]
            save_image(fake_imgs, f'../results/generated_images/epoch_{epoch+1}.png', nrow=5)
        print(f'Epoch [{epoch+1}/{epochs}] | Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}')
    
    # Save models
    torch.save(generator.state_dict(), '../models/generator.pth')
    torch.save(discriminator.state_dict(), '../models/discriminator.pth')

if __name__ == "__main__":
    train_gan()
