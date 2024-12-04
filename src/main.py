# src/main.py

import os
from gan_training import train_gan
from unet_training import train_unet
from llm_integration import generate_report
from utils import visualize_segmentation

def main():
    # Set up directories
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    os.makedirs('../results/generated_images', exist_ok=True)
    os.makedirs('../results/segmentation_outputs', exist_ok=True)
    os.makedirs('../results/reports', exist_ok=True)

    # Data Preprocessing is handled within training scripts now

    # GAN Training
    train_gan(epochs=50, batch_size=32, noise_dim=100, device='cuda')

    # U-Net Training
    train_unet(epochs=25, batch_size=4, device='cuda')

    # Generate Segmentation Outputs
    # Code to generate segmentation outputs would go here

    # LLM Integration
    segmentation_output_path = '../results/segmentation_outputs/segmentation_info.json'
    report_output_path = '../results/reports/diagnostic_report.txt'
    generate_report(segmentation_output_path, report_output_path)

    # Visualization
    # Code to visualize segmentation results would go here

if __name__ == "__main__":
    main()
