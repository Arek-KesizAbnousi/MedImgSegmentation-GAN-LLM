# src/main.py

import os
from data_preprocessing import load_images, split_dataset
from gan_training.py import train_gan
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

    # Data Preprocessing
    image_size = 256
    images, masks = load_images('../data/images', '../data/masks', image_size)
    X_train, X_val, y_train, y_val = split_dataset(images, masks)

    # Save preprocessed data
    np.save('../data/X_train.npy', X_train)
    np.save('../data/X_val.npy', X_val)
    np.save('../data/y_train.npy', y_train)
    np.save('../data/y_val.npy', y_val)

    # GAN Training
    train_gan(epochs=50, batch_size=32, noise_dim=100, device='cuda')

    # U-Net Training
    train_unet(epochs=25, batch_size=4, device='cuda')

    # Generate Segmentation Outputs
    # Load test images and make predictions (code not shown for brevity)
    # Save predictions to '../results/segmentation_outputs/'

    # LLM Integration
    segmentation_output_path = '../results/segmentation_outputs/segmentation_info.json'
    report_output_path = '../results/reports/diagnostic_report.txt'
    generate_report(segmentation_output_path, report_output_path)

    # Visualization
    # visualize_segmentation(image, mask, prediction, save_path='../results/segmentation_outputs/visualization.png')

if __name__ == "__main__":
    main()
