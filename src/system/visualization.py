"""
Visualization utilities for images and results.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from sklearn.metrics import ConfusionMatrixDisplay

def plot_synapse_detection(
    original_image: np.ndarray,
    worm_mask: np.ndarray,
    maxima: List[Tuple[int, int]],
    title: str = 'Synapse Detection Results'
) -> None:
    """
    Plot synapse detection results.
    
    Args:
        original_image: Original microscopy image
        worm_mask: Binary mask of segmented worm
        maxima: List of synapse coordinates
        graph: NetworkX graph of synapses
        title: Plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    fig.suptitle(title, fontsize=16)
    
    # Plot original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title(f'Original Image : {title}')
    axes[0].axis('off')
    
    # Plot worm mask
    axes[1].imshow(worm_mask, cmap='gray')
    axes[1].set_title('Worm Segmentation')
    axes[1].axis('off')
    
    # Plot maxima on original image
    axes[2].imshow(original_image, cmap='gray')
    if maxima:
        maxima = np.array(maxima)
        axes[2].scatter(maxima[:, 1], maxima[:, 0], c='red', s=3) 
    axes[2].set_title('Detected Synapses')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_images(images: List[np.ndarray], 
                titles: Optional[List[str]] = None,
                cmap: str = 'gray',
                figsize: Tuple[int, int] = (12, 4)) -> None:
    """
    Plot multiple images in a row.
    
    Args:
        images: List of images to plot
        titles: Optional list of titles for each image
        cmap: Colormap to use
        figsize: Figure size
    """
    n_images = len(images)
    if titles is None:
        titles = [f'Image {i+1}' for i in range(n_images)]
        
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    if n_images == 1:
        axes = [axes]
        
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

def plot_heatmap(data_df: pd.DataFrame,):
    """
    Plot heatmap of model accuracy for different scaling methods.
    This function visualizes the accuracy of different models with various scaling methods.

    Args:
        data_df (pd.DataFrame): DataFrame containing model accuracies.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(data_df.T * 100, annot=True, fmt='.2f', cmap='viridis',
                xticklabels=data_df.index, yticklabels=data_df.columns)
    plt.title('Model Accuracy (%) for Different Scaling Methods')
    plt.ylabel('Scaling Method')
    plt.xlabel('Model Type')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_and_learning_curve(cm, train_sizes, train_mean, train_std, test_mean, test_std):
    """
    Plot confusion matrix and learning curve.
    """
    # Subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    # Matrice de confusion
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axs[0], cmap='Blues', colorbar=False)
    axs[0].set_title(f"Confusion Matrix")
    # Courbe d'apprentissage
    axs[1].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
    axs[1].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="orange")
    axs[1].plot(train_sizes, train_mean, 'o-', color="blue", label="Training score")
    axs[1].plot(train_sizes, test_mean, 'o-', color="orange", label="Cross-validation score")
    axs[1].set_title("Learning Curve")
    axs[1].set_xlabel("Training Examples")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend(loc="best")
    axs[1].grid(True)
    fig.tight_layout()
    plt.show()



