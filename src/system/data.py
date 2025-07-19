import numpy as np
import networkx as nx

class Data:
    """
    Class for managing the data.
    """
    def __init__(self):
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
        
    def get_original_data(self):
        """
        Get the original image and label.
        
        Returns:
            Original image and label.
        """
        return self.original_image, self.label
    
    def get_features(self):
        """
        Get the features and feature names.
        
        Returns:
            Features and feature names.
        """
        return self.features, self.feature_names
    
    def get_features_selected(self):
        """
        Get the selected features and feature names.
        """
        return self.features_selected, self.features_names_selected
    
    def set_features(self, features: np.ndarray, feature_names: list):
        """
        Set the features and feature names.
        
        Args:
            features: Numpy array of features.
            feature_names: List of feature names.
        """
        self.features = features
        self.feature_names = feature_names
        
    def set_features_selected(self, features: np.ndarray, feature_names: list):
        """
        Set the selected features and feature names.
        Args:
            features: Numpy array of selected features.
            feature_names: List of selected feature names.
        """
        self.features_selected = features
        self.features_names_selected = feature_names

        
        
