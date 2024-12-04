# src/neural_networks.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# U-Net Model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Define U-Net architecture
        self.enc1 = self.contracting_block(3, 64)
        self.enc2 = self.contracting_block(64, 128)
        self.enc3 = self.contracting_block(128, 256)
        self.enc4 = self.contracting_block(256, 512)

        self.bottom = self.contracting_block(512, 1024)

        self.upconv4 = self.expansive_block(1024, 512)
        self.upconv3 = self.expansive_block(512, 256)
        self.upconv2 = self.expansive_block(256, 128)
        self.upconv1 = self.expansive_block(128, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        return block

    def expansive_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            ),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        return block

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2, stride=2))

        # Bottleneck
        bottom = self.bottom(F.max_pool2d(enc4, kernel_size=2, stride=2))

        # Decoder
        dec4 = self.upconv4(bottom)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)

        output = self.final_conv(dec1)
        output = torch.sigmoid(output)
        return output

# GAN Models
class Generator(nn.Module):
    def __init__(self, noise_dim=100, img_channels=3):
        super(Generator, self).__init__()
        self.fc = nn.Linear(noise_dim, 256 * 8 * 8)
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 8, 8)
        x = self.gen(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),           # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=8),                                 # 1x1
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.disc(x)
        x = x.view(-1)
        return x
