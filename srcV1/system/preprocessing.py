"""
Image preprocessing utilities for synapse detection.
"""
import cv2
import numpy as np
import skimage as ski
import networkx as nx
from ultralytics import YOLO
from skimage.morphology import binary_closing, disk
        
from config import DATA_DIR, MODELS_DIR
from .graph_analysis import get_synapses_graph

def worm_segmentation(img: np.ndarray) -> np.ndarray:
    """
    Segment worm from background using either YOLO or traditional filtering.
    
    Args:
        img: Input image
        
    Returns:
        Binary mask of segmented worm
    """
    method = "YOLO"  # "Filter" or "YOLO"
    
    if method == "YOLO":
        model_path = MODELS_DIR / "YOLO_segmentation.pt"
        if not model_path.exists():
            print("YOLO model not found. Using filter method instead.")
            method = "Filter"
        else:
            model = YOLO(str(model_path))
            image = img.copy()
            
            # Normalize image for YOLO
            threshold = 3000
            image = np.clip(image, 0, threshold).astype(np.uint16)
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            # Save temporary image
            temp_path = DATA_DIR / "temp_converted_image.png"
            cv2.imwrite(str(temp_path), image)
            
            # Predict
            prediction = model.predict(source=str(temp_path), save=False, verbose=False)
            temp_path.unlink()  # Remove temp file
            
            masks = prediction[0].masks
            if masks is not None:
                mask_array = masks.data
                worm_mask = mask_array[0].cpu().numpy()
                worm_mask = cv2.resize(worm_mask, (image.shape[1], image.shape[0]))
                
                # Keep largest component 
                # TODO 
                labeled_mask = ski.measure.label(worm_mask)
                largest_component = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
                worm_mask = (labeled_mask == largest_component).astype(np.uint8)
                
                if np.sum(worm_mask) > 0:
                    return worm_mask
                    
            method = "Filter"
            print("No worm detected with YOLO. Using filter method.")
    
    if method == "Filter":
        # Apply vessel enhancement filter
        img = ski.filters.meijering(img, sigmas=range(8, 14, 2), black_ridges=False)
        binary_mask = img > np.mean(img)
        
        # Clean up mask
        cleaned_mask = remove_small_objects(binary_mask)
        cleaned_mask = cleaned_mask.astype(bool)
        
        # Fill holes and close gaps
        worm_mask = ski.morphology.remove_small_holes(cleaned_mask, area_threshold=50)
        worm_mask = binary_closing(worm_mask, disk(20))
        
        # Keep largest component
        labeled_mask = ski.measure.label(worm_mask)
        largest_component = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
        worm_mask = (labeled_mask == largest_component).astype(np.uint8)
        
        return worm_mask

def is_coiled_worm(worm_mask: np.ndarray) -> bool:
    """
    Determine if worm is coiled based on shape analysis.
    
    Args:
        worm_mask: Binary mask of segmented worm
        
    Returns:
        True if worm is coiled, False otherwise
    """
    if np.sum(worm_mask) == 0:
        return True
        
    labeled_mask = ski.measure.label(worm_mask)
    regions = ski.measure.regionprops(labeled_mask)
    
    if not regions:
        return True
        
    region = regions[0]
    return region.major_axis_length / region.minor_axis_length <= 1.5



def get_synapse_using_graph(image: np.ndarray, worm_mask: np.ndarray) -> tuple:
    """
    Detect synapses using graph-based approach.
    
    Args:
        image: Input image
        worm_mask: Binary mask of the segmented worm
        
    Returns:
        Tuple of (maxima coordinates, graph, median width, slice difference measure,
                point segment difference measure)
    """
    try:
        # Preprocess image
        img, local_max = find_local_maxima(image)

        # Get synapses graph
        maxima, G, median_width, diff_slice, diff_segment, head_mask_1, head_mask_2 = get_synapses_graph(
            worm_mask,
            local_max
        )

        
        # HEAD
        """
        mean_intensity_1 = round(np.mean(img[head_mask_1 == 1]), 2) if np.any(head_mask_1 == 1) else float('inf')
        mean_intensity_2 = round(np.mean(img[head_mask_2 == 1]), 2) if np.any(head_mask_2 == 1) else float('inf')
        median_intensity_1 = round(np.median(img[head_mask_1 == 1]), 2) if np.any(head_mask_1 == 1) else float('inf')
        median_intensity_2 = round(np.median(img[head_mask_2 == 1]), 2) if np.any(head_mask_2 == 1) else float('inf')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.imshow(worm_mask, cmap='gray', alpha=0.8)
        plt.imshow(img, cmap='gray', alpha=0.8)
        plt.imshow(np.ma.masked_where(head_mask_1 == 0, head_mask_1), 
                cmap='Blues', alpha=0.6, vmin=0, vmax=1)
        plt.imshow(np.ma.masked_where(head_mask_2 == 0, head_mask_2), 
                cmap='Greens', alpha=0.6, vmin=0, vmax=1)
        plt.title('Worm Head Detection Analysis', fontsize=14)
        plt.legend()
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.6, label=f'Head Mask 1 (mean : {mean_intensity_1} ; median : {median_intensity_1})'),
            Patch(facecolor='green', alpha=0.6, label=f'Head Mask 2 (mean : {mean_intensity_2} ; median : {median_intensity_2})'),
            Patch(facecolor='white', alpha=0.8, label='Worm Mask')
        ]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.axis('off')  # Remove axes for cleaner look
        plt.tight_layout()
        plt.show()"""

        maxima = list(map(tuple, maxima))

        return maxima, G, median_width, diff_slice, diff_segment
        
    except Exception as e:
        print(f"Error in synapse detection: {str(e)}")
        empty_graph = nx.Graph()
        return [], empty_graph, 0, 0, 0


# Utils for get_synapse_using_graph
def find_local_maxima(img: np.ndarray) -> tuple:
    """
    Preprocess image for graph-based analysis.
    
    Args:
        img: Input image
        
    Returns:
        Tuple of (processed image, local maxima coordinates)
    """
    # Apply Frangi filter
    frangi_response = ski.filters.frangi(
        img,
        black_ridges=False,
        sigmas=range(1, 3, 1),
        alpha=0.5,
        beta=0.5,
        gamma=70
    )
    frangi_response = ski.filters.apply_hysteresis_threshold(frangi_response, 0.01, 0.2)
    
    # Clean up response
    frangi_response = remove_small_objects(frangi_response)
    
    # Keep line-like components
    labeled_image = ski.measure.label(frangi_response)
    components = ski.measure.regionprops(labeled_image)
    label_components = np.zeros_like(labeled_image)
    
    # Don't keep line-like components
    for component in components:
        if component.major_axis_length / component.minor_axis_length > 4:
            label_components[labeled_image == component.label] = 1
            
    frangi_response = label_components
    
    # Normalize response
    frangi_response = (frangi_response - frangi_response.min()) / (frangi_response.max() - frangi_response.min())
    
    # Create mask
    threshold = np.percentile(frangi_response, 95)
    mask = frangi_response > threshold
    
    # Apply mask
    masked_img = img.copy()
    masked_img[~mask] = 0
    
    # Find local maxima
    local_max = ski.feature.peak_local_max(
        masked_img,
        min_distance=5,
        threshold_abs=0,
        exclude_border=False
    )
    
    return img, local_max

def remove_small_objects(binary_image: np.ndarray) -> np.ndarray:
    """
    Remove small objects from binary image.
    
    Args:
        image: Input binary image
        
    Returns:
        Cleaned binary image
    """

    labeled_image = ski.measure.label(binary_image)
    regions = ski.measure.regionprops(labeled_image)
    large_regions = np.zeros_like(labeled_image)
        
    for region in regions:
        if region.area > 25:
            large_regions[labeled_image == region.label] = 1
    return large_regions.astype(np.uint16)

