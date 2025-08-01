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
from src.system.graph_analysis import get_synapses_graph

class Preprocessing():
    """
    A collection of utility methods for image preprocessing, focusing on
    worm segmentation and synapse detection in microscopy images.

    This class encapsulates various image processing algorithms, including
    deep learning-based segmentation (YOLO) and traditional filter-based
    methods, as well as functions for analyzing worm morphology and
    identifying potential synapses.
    """
    def __init__(self):
        """
        Initialize preprocessing utilities.
        """
        pass
    
    def worm_segmentation(self, img: np.ndarray) -> np.ndarray:
        """
        Segments the worm from the background using either a YOLO model or traditional
        image filtering techniques.

        The function first attempts to use a pre-trained YOLO segmentation model.
        If the model is not found or fails to detect a worm, it falls back to
        a traditional method involving a Meijering vessel enhancement filter
        and morphological operations.

        Args:
            img (np.ndarray): The input image, typically a grayscale microscopy image.
            
        Returns:
            np.ndarray: A binary mask (with dtype np.uint8) of the segmented worm.
                        Returns an empty mask if no worm is detected by either method.
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
            cleaned_mask = self.remove_small_objects(binary_mask)
            cleaned_mask = cleaned_mask.astype(bool)
            
            # Fill holes and close gaps
            worm_mask = ski.morphology.remove_small_holes(cleaned_mask, area_threshold=50)
            worm_mask = binary_closing(worm_mask, disk(20))
            
            # Keep largest component
            labeled_mask = ski.measure.label(worm_mask)
            largest_component = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
            worm_mask = (labeled_mask == largest_component).astype(np.uint8)
            
            return worm_mask

    def is_coiled_worm(self, worm_mask: np.ndarray) -> bool:
        """
        Determines if a worm is in a coiled state based on its shape properties.

        The method calculates the ratio of the major axis length to the minor
        axis length of the segmented worm. A low ratio (e.g., less than or
        equal to 1.5) is indicative of a more circular, coiled shape.

        Args:
            worm_mask (np.ndarray): The binary mask of the segmented worm.
            
        Returns:
            bool: True if the worm is classified as coiled, False otherwise.
        """
        if np.sum(worm_mask) == 0:
            return True
            
        labeled_mask = ski.measure.label(worm_mask)
        regions = ski.measure.regionprops(labeled_mask)
        
        if not regions:
            return True
            
        region = regions[0]
        return region.major_axis_length / region.minor_axis_length <= 1.5

    def get_synapse_using_graph(self, image: np.ndarray, worm_mask: np.ndarray) -> tuple:
        """
        Detects synapses in an image using a graph-based approach.

        This function first preprocesses the image to find potential local maxima
        (synapse candidates), then builds a graph connecting these candidates
        based on their proximity and relationship to the worm's shape.
        It returns the final synapse coordinates and various metrics from the analysis.

        Args:
            image (np.ndarray): The input image containing the worm.
            worm_mask (np.ndarray): The binary mask of the segmented worm.
            
        Returns:
            Tuple: A tuple containing:
                - List[Tuple[int, int]]: The (row, column) coordinates of detected synapses.
                - nx.Graph: The NetworkX graph representing the synapse network.
                - float: The median width of the worm.
                - float: A measure of the difference in worm slices.
                - float: A measure of the difference in point segments.
                - int: The number of cords (branches) in the worm skeleton.
        """
        try:
            # Preprocess image
            img, local_max = self.find_local_maxima(image)
            
            # Get synapses graph
            maxima, G, median_width, diff_slice, diff_segment, NUMBER_OF_CORDS = get_synapses_graph(
                worm_mask,
                local_max
            )

            maxima = list(map(tuple, maxima))

            return maxima, G, median_width, diff_slice, diff_segment, NUMBER_OF_CORDS
            
        except Exception as e:
            print(f"Error in synapse detection: {str(e)}")
            empty_graph = nx.Graph()
            return [], empty_graph, 0, 0, 0, 0

    # Utils for get_synapse_using_graph
    def find_local_maxima(self, img: np.ndarray) -> tuple:
        """
        Preprocesses an image to identify potential synapse locations by finding
        local maxima after applying a Frangi filter.

        The process involves enhancing line-like structures (e.g., neural cords)
        with the Frangi filter, cleaning up the response, and then using a
        peak-finding algorithm to locate the most intense points.

        Args:
            img (np.ndarray): The input image.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - np.ndarray: The original image (unmodified).
                - np.ndarray: An array of (row, column) coordinates of the local maxima.
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
        frangi_response = self.remove_small_objects(frangi_response)
        
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

    def remove_small_objects(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Removes small connected components from a binary image.

        This is a common noise reduction step in image processing. It
        labels all connected components and discards those below a
        certain area threshold.

        Args:
            binary_image (np.ndarray): The input binary image.
            
        Returns:
            np.ndarray: A new binary image with small objects removed.
        """

        labeled_image = ski.measure.label(binary_image)
        regions = ski.measure.regionprops(labeled_image)
        large_regions = np.zeros_like(labeled_image)
            
        for region in regions:
            if region.area > 25:
                large_regions[labeled_image == region.label] = 1
        return large_regions.astype(np.uint16)

