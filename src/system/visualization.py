"""
Visualization utilities for images and results.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from sklearn.metrics import ConfusionMatrixDisplay

def plot_synapse_detection(
                            original_image: np.ndarray,
                            worm_mask: np.ndarray,
                            maxima: List[Tuple[int, int]],
                            title: str = 'Synapse Detection Results',
                            display: bool = True
                        ) -> None:
    """
    Plot synapse detection results, including the original image, worm mask, and detected synapses.
    
    This function generates a three-panel figure to visualize the different stages
    of synapse detection: the original input image, the segmented worm mask, and
    the detected synapse locations overlaid on the original image.

    Args:
        original_image (np.ndarray): The original microscopy image, expected to be grayscale.
        worm_mask (np.ndarray): The binary mask representing the segmented worm.
        maxima (List[Tuple[int, int]]): A list of (row, column) coordinates for each detected synapse.
        title (str): The main title for the entire plot. Defaults to 'Synapse Detection Results'.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    fig.suptitle(title, fontsize=16)
    
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title(f'Original Image : {title}')
    axes[0].axis('off')

    axes[1].imshow(worm_mask, cmap='gray')
    axes[1].set_title('Worm Segmentation')
    axes[1].axis('off')

    axes[2].imshow(original_image, cmap='gray')
    if maxima:
        maxima = np.array(maxima)
        axes[2].scatter(maxima[:, 1], maxima[:, 0], c='red', s=3)
    axes[2].set_title('Detected Synapses')
    axes[2].axis('off')

    plt.tight_layout()

    # Save the plot to an in-memory bytes buffer
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)  # Close the figure to free memory
    
    # Convert buffer to PIL image
    image = Image.open(buf)
    
    if display:
        image.show()
        
    return image

def plot_images(images: List[np.ndarray], 
                titles: Optional[List[str]] = None,
                cmap: str = 'gray',
                figsize: Tuple[int, int] = (12, 4)) -> None:
    """
    Plot a list of images side-by-side in a single row.
    
    This utility function is useful for visualizing a sequence of images,
    such as different stages of an image processing pipeline.

    Args:
        images (List[np.ndarray]): A list of images to be displayed.
        titles (Optional[List[str]]): An optional list of titles for each image.
                                      If None, default titles 'Image 1', 'Image 2', etc., are used.
        cmap (str): The colormap to use for plotting the images. Defaults to 'gray'.
        figsize (Tuple[int, int]): The size of the entire figure. Defaults to (12, 4).
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
    Plots a heatmap of model accuracy for different scaling methods.
    
    This function visualizes the accuracy of various machine learning models
    with different data scaling techniques, providing a clear comparison.
    The values in the heatmap are presented as percentages.

    Args:
        data_df (pd.DataFrame): A DataFrame where the index represents the models,
                                the columns represent the scaling methods, and the
                                cell values are the accuracy scores.
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
    Plots a confusion matrix and a learning curve side-by-side.
    
    This function is a useful tool for model evaluation, showing both the
    classification performance (confusion matrix) and the model's learning
    dynamics as the training set size increases (learning curve).

    Args:
        cm (np.ndarray): The confusion matrix as a numpy array.
        train_sizes (np.ndarray): The number of training examples used to generate
                                  the learning curve.
        train_mean (np.ndarray): The mean accuracy scores on the training sets.
        train_std (np.ndarray): The standard deviation of the training scores.
        test_mean (np.ndarray): The mean accuracy scores on the cross-validation sets.
        test_std (np.ndarray): The standard deviation of the cross-validation scores.
    """
    # Subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axs[0], cmap='Blues', colorbar=False)
    axs[0].set_title(f"Confusion Matrix")
    # Learning curve
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



