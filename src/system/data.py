import numpy as np
import networkx as nx

class Data:
    """
    A class for managing and storing data related to an analysis, likely of
    biological images (e.g., worms).

    This class holds various attributes such as image filenames, labels,
    mutant types, and extracted features, providing methods to access and
    manipulate this data.
    """
    def __init__(self):
        """
        Initializes a new Data object with default, empty attributes.
        """
        self.filename = []
        self.label = []
        self.mutant_type = [] 
        self.original_image = [] 
        self.coiled = False
        self.worm_mask = [] 
        self.maxima = [] 
        self.graph = nx.Graph() 
        self.median_width = 0 
        self.diff_slice = 0 
        self.diff_segment = 0 
        self.features = []
        self.feature_names = []
        self.features_selected = []
        self.features_names_selected = []
        self.number_of_cords = 1
        
    def get_original_data(self):
        """
        Retrieves the original image and its associated label.
        
        Returns:
            tuple: A tuple containing the original image data and the label.
        """
        return self.original_image, self.label
    
    def get_features(self):
        """
        Retrieves the complete set of features and their corresponding names.
        
        Returns:
            tuple: A tuple containing the numpy array of all features and a list
                   of their names.
        """
        return self.features, self.feature_names
    
    def get_features_selected(self):
        """
        Retrieves the subset of selected features and their corresponding names.
        
        Returns:
            tuple: A tuple containing the numpy array of selected features and a list
                   of their names.
        """
        print(f"DEBUG: Data.get_features_selected called for {self.filename}")
        # print(f"DEBUG: Returning {len(self.features_selected) if self.features_selected is not None else 'None'} features")
        return self.features_selected, self.features_names_selected
    
    def set_features(self, features: np.ndarray, feature_names: list):
        """
        Sets the full set of features and their names for the dataset.
        
        Args:
            features (np.ndarray): The numpy array containing all the features.
            feature_names (list): The list of strings for the feature names.
        """
        self.features = features
        self.feature_names = feature_names
        
    def set_features_selected(self, features: np.ndarray, feature_names: list):
        """
        Sets the selected subset of features and their corresponding names.
        
        Args:
            features (np.ndarray): The numpy array containing the selected features.
            feature_names (list): The list of strings for the selected feature names.
        """
        self.features_selected = features
        self.features_names_selected = feature_names

    def print_data(self):
        """
        Prints the data in a human-readable format.
        """
        print("Filename:", self.filename)
        print("Label:", self.label)
        print("Mutant Type:", self.mutant_type)
        print("Original Image:", self.original_image)
        print("Coiled:", self.coiled)
        print("Worm Mask:", self.worm_mask)
        print("Maxima:", self.maxima)
        print("Graph:", self.graph)
        print("Median Width:", self.median_width)
        print("Diff Slice:", self.diff_slice)
        print("Diff Segment:", self.diff_segment)
        print("Features:", self.features)
        print("Feature Names:", self.feature_names)
        print("Features Selected:", self.features_selected)
        print("Features Names Selected:", self.features_names_selected)
        print("Number of Cords:", self.number_of_cords)  
        
