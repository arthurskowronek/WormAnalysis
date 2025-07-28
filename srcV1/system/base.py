"""
Base model class that defines the interface for all models.
"""
from abc import ABC, abstractmethod
import joblib
from pathlib import Path
import numpy as np
from typing import Optional, Tuple, Dict, Any

from config import MODELS_DIR, DEFAULT_RANDOM_STATE

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE):
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseModel':
        """
        Fit the model to the data.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional arguments for the model
            
        Returns:
            self: The fitted model
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        pass
    
    def save(self, filename: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filename: Name of the file to save to
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        save_path = Path(MODELS_DIR) / filename
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'random_state': self.random_state
        }
        joblib.dump(model_data, save_path)
        
    def load(self, filename: str) -> 'BaseModel':
        """
        Load a saved model from disk.
        
        Args:
            filename: Name of the file to load from
            
        Returns:
            self: The loaded model
        """
        load_path = Path(MODELS_DIR) / filename
        model_data = joblib.load(load_path)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.random_state = model_data['random_state']
        self.is_fitted = True
        
        return self
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get the model's parameters.
        
        Returns:
            Dictionary of parameters
        """
        pass
    
    @abstractmethod
    def set_params(self, **params) -> 'BaseModel':
        """
        Set the model's parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            self: The model with updated parameters
        """
        pass
    
    def get_feature_importance(self) -> Optional[Tuple[np.ndarray, list]]:
        """
        Get feature importance if the model supports it.
        
        Returns:
            Tuple of (importance scores, feature names) or None if not supported
        """
        return None 