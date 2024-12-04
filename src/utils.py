# src/utils.py

import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_segmentation(image, mask, prediction, save_path=None):
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.title('Original Image')

    plt.subplot(1,3,2)
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.title('Ground Truth Mask')

    plt.subplot(1,3,3)
    plt.imshow(prediction.squeeze(), cmap='gray')
    plt.title('Predicted Mask')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_training_history(history, save_path=None):
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
