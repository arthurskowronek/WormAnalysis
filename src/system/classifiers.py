"""
Implementation of various classifiers.
"""
import shap
import optuna
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from typing import Dict, Any, Optional, Tuple
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier as SKHistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier as SklearnMLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict, learning_curve
from sklearn.metrics import confusion_matrix


from src.system.base import BaseModel
from config import DEFAULT_RANDOM_STATE, DEFAULT_CV_FOLDS
from src.system.visualization import plot_confusion_matrix_and_learning_curve

class ClassifierFactory:
    """Factory class for creating different types of classifiers."""
    
    @staticmethod
    def create(classifier_type: str, **kwargs) -> 'BaseClassifier':
        """
        Create a classifier of the specified type.
        
        Args:
            classifier_type: Type of classifier to create
            **kwargs: Additional arguments for the classifier
            
        Returns:
            Instantiated classifier
        """
        classifiers = {
            'random_forest': RFClassifier,
            'hist_gradient_boosting': HistGradientBoostingClassifier,
            'svm': SVMClassifier,
            'knn': KNNClassifier,
            'mlp': MLPClassifier
        }
        
        if classifier_type not in classifiers:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
            
        return classifiers[classifier_type](**kwargs)

class BaseClassifier(BaseModel):
    """
    Abstract base class for all classification models.

    Extends BaseModel by adding support for hyperparameter optimization via Optuna.
    Specific classifier implementations must implement `_get_trial_params()` to define
    the search space for their hyperparameters.
    """

    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
        cv: int = DEFAULT_CV_FOLDS
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Optimize the model's hyperparameters using Optuna with cross-validation.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target vector of shape (n_samples,).
            n_trials (int): Number of trials to run in the optimization.
            cv (int): Number of cross-validation folds (e.g., 5 or 10).

        Returns:
            Tuple[float, Dict[str, Any]]:
                - best_value: The best cross-validation score found.
                - best_params: The hyperparameters corresponding to the best score.
        
        Raises:
            NotImplementedError: If `_get_trial_params()` is not implemented in a subclass.
        """

        def objective(trial):
            """
            Objective function for Optuna trial.

            This function defines how each trial is evaluated by:
            - Sampling hyperparameters via `_get_trial_params()`
            - Setting them using `set_params()`
            - Running cross-validation and returning the mean accuracy
            """
            params = self._get_trial_params(trial)
            self.set_params(**params)

            # Stratified k-fold CV to preserve class distribution
            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

            # Evaluate cross-validation accuracy
            scores = cross_val_score(self.model, X, y, cv=kf, scoring='accuracy', n_jobs=-1)
            return np.mean(scores)

        # Create a study aiming to maximize accuracy
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        # Save visualizations to HTML files
        optuna.visualization.plot_optimization_history(study).write_html("results/optuna/optuna_optimization_history.html")
        optuna.visualization.plot_param_importances(study).write_html("results/optuna/optuna_param_importances.html")
        optuna.visualization.plot_parallel_coordinate(study).write_html("results/optuna/optuna_parallel_coordinate.html")
        optuna.visualization.plot_contour(study).write_html("results/optuna/optuna_contour.html")
        optuna.visualization.plot_slice(study).write_html("results/optuna/optuna_slice.html")

        # Apply the best found parameters to the model
        self.set_params(**study.best_params)

        return study.best_value, study.best_params

    def _get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the hyperparameter search space for Optuna.

        This method must be implemented by any subclass that uses `optimize_hyperparameters`.

        Args:
            trial (optuna.Trial): The Optuna trial object to sample parameters from.

        Returns:
            Dict[str, Any]: A dictionary of hyperparameters to test.

        Raises:
            NotImplementedError: This method is abstract and must be overridden in subclasses.
        """
        raise NotImplementedError

class RFClassifier(BaseClassifier):
    """
    Random Forest classifier implementation based on BaseClassifier.

    This class wraps around sklearn's RandomForestClassifier and supports
    hyperparameter optimization via Optuna as well as standard model utilities
    like fitting, predicting, saving, loading, and retrieving feature importance.

    Attributes:
        model (RandomForestClassifier): The underlying sklearn classifier.
    """

    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE, **kwargs):
        """
        Initialize the Random Forest classifier with default or user-supplied hyperparameters.

        Args:
            random_state (int): Random seed for reproducibility.
            **kwargs: Additional parameters to override defaults (e.g., n_estimators, max_depth).
        """
        super().__init__(random_state)

        # Default hyperparameters
        default_params = {
            'n_estimators': 80,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'bootstrap': True
        }

        # Allow override of any defaults via kwargs
        default_params.update(kwargs)

        # Initialize the RandomForestClassifier with combined parameters
        self.model = RandomForestClassifier(random_state=random_state, **default_params)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'RFClassifier':
        """
        Fit the Random Forest classifier to the training data.

        Args:
            X (np.ndarray): Training feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target vector of shape (n_samples,).
            **kwargs: Optional arguments passed to the underlying `fit()` method.

        Returns:
            RFClassifier: The fitted classifier instance.
        """
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for new data using the trained classifier.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(X)

    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Get current hyperparameters of the underlying Random Forest model.

        Args:
            deep (bool): If True, will return parameters of nested estimators.

        Returns:
            Dict[str, Any]: Dictionary of model parameters.
        """
        return self.model.get_params(deep=deep)

    def set_params(self, **params) -> 'RFClassifier':
        """
        Set hyperparameters of the underlying Random Forest model.

        Args:
            **params: Arbitrary keyword arguments of parameters to update.

        Returns:
            RFClassifier: The classifier instance with updated parameters.
        """
        self.model.set_params(**params)
        return self

    def get_feature_importance(self) -> Optional[Tuple[np.ndarray, list]]:
        """
        Get feature importance scores if available.

        Returns:
            Optional[Tuple[np.ndarray, list]]: Tuple containing feature importance
            scores and feature names, or None if unavailable.
        """
        if self.is_fitted and self.feature_names:
            return self.model.feature_importances_, self.feature_names
        return None

    def _get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the hyperparameter search space for Optuna trials.

        Args:
            trial (optuna.Trial): Trial object used to suggest hyperparameters.

        Returns:
            Dict[str, Any]: Dictionary of sampled hyperparameters.
        """
        return {
            'n_estimators': trial.suggest_int('n_estimators', 30, 80),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }

class SVMClassifier(BaseClassifier):
    """
    Support Vector Machine (SVM) classifier implementation based on BaseClassifier.

    This class wraps around sklearn's SVC and supports hyperparameter optimization 
    using Optuna, as well as basic model functionalities like fitting, predicting, 
    and parameter management.

    Attributes:
        model (SVC): The underlying scikit-learn Support Vector Classifier.
    """

    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE, probability: bool = True, **kwargs):
        """
        Initialize the SVM classifier with default or user-supplied hyperparameters.

        Args:
            random_state (int): Seed for reproducibility.
            probability (bool): Whether to enable probability estimates via Platt scaling.
            **kwargs: Additional keyword arguments to override the default hyperparameters.
        """
        super().__init__(random_state)

        # Default hyperparameters for SVC
        default_params = {
            'C': 11.3,                      # Regularization strength
            'kernel': 'rbf',                # Kernel type (radial basis function)
            'gamma': 0.12,                  # Kernel coefficient
            'class_weight': 'balanced'      # Handle class imbalance
        }

        # Allow user-specified overrides
        default_params.update(kwargs)

        # Instantiate the SVC model with combined parameters
        self.model = SVC(random_state=random_state, probability=probability, **default_params)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SVMClassifier':
        """
        Fit the SVM model to the training data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target vector of shape (n_samples,).
            **kwargs: Additional arguments passed to the underlying `fit()` method.

        Returns:
            SVMClassifier: The fitted model instance.
        """
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for new input samples.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels.
        """
        return self.model.predict(X)

    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Get the current hyperparameters of the SVM model.

        Args:
            deep (bool): If True, will return parameters of nested estimators.

        Returns:
            Dict[str, Any]: A dictionary of the model’s parameters.
        """
        return self.model.get_params(deep=deep)

    def set_params(self, **params) -> 'SVMClassifier':
        """
        Set hyperparameters of the SVM model.

        Args:
            **params: Keyword arguments representing parameters to update.

        Returns:
            SVMClassifier: The model instance with updated parameters.
        """
        self.model.set_params(**params)
        return self

    def _get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the hyperparameter search space for Optuna optimization.

        Args:
            trial (optuna.Trial): Trial object used by Optuna to suggest parameters.

        Returns:
            Dict[str, Any]: Dictionary of sampled hyperparameters.
        """
        return {
            'C': trial.suggest_float('C', 11, 12, log=True),             # Search around base value 11.3
            'gamma': trial.suggest_float('gamma', 0.09, 0.2, log=True),  # Search range for RBF kernel width
            'kernel': trial.suggest_categorical('kernel', ['rbf'])       # Fixed to RBF (can be expanded)
        }

class KNNClassifier(BaseClassifier):
    """
    K-Nearest Neighbors (KNN) classifier implementation based on BaseClassifier.

    This class wraps around sklearn's KNeighborsClassifier and includes functionality
    for fitting, prediction, hyperparameter tuning using Optuna, and parameter management.

    Attributes:
        model (KNeighborsClassifier): The underlying scikit-learn KNN model.
    """

    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE, **kwargs):
        """
        Initialize the KNN classifier with default or user-provided hyperparameters.

        Args:
            random_state (int): Not used in KNN directly, but stored for consistency.
            **kwargs: Additional parameters to override default KNN hyperparameters.
        """
        super().__init__(random_state)

        # Default KNN hyperparameters
        default_params = {
            'n_neighbors': 7,     # Number of neighbors to use for classification
            'weights': 'uniform', # Weight function used in prediction ('uniform' or 'distance')
            'p': 1                # Power parameter for the Minkowski metric (1=Manhattan, 2=Euclidean)
        }

        # Allow user to override default parameters
        default_params.update(kwargs)

        # Initialize the KNN model with the chosen parameters
        self.model = KNeighborsClassifier(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'KNNClassifier':
        """
        Fit the KNN model to the training data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target vector of shape (n_samples,).
            **kwargs: Additional arguments passed to the `fit()` method.

        Returns:
            KNNClassifier: The fitted model instance.
        """
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the input samples.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels.
        """
        return self.model.predict(X)

    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Get the current hyperparameters of the KNN model.

        Args:
            deep (bool): If True, return parameters of nested estimators.

        Returns:
            Dict[str, Any]: Dictionary of the model’s parameters.
        """
        return self.model.get_params(deep=deep)

    def set_params(self, **params) -> 'KNNClassifier':
        """
        Set new hyperparameters for the KNN model.

        Args:
            **params: Key-value pairs of parameters to update.

        Returns:
            KNNClassifier: The model instance with updated parameters.
        """
        self.model.set_params(**params)
        return self

    def _get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the search space for Optuna hyperparameter tuning.

        Args:
            trial (optuna.Trial): Optuna trial object for suggesting parameters.

        Returns:
            Dict[str, Any]: Suggested hyperparameters for the current trial.
        """
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', 5, 10),           # Number of neighbors to try
            'weights': trial.suggest_categorical('weights', ['uniform']),     # Fixed to 'uniform' for now
            'p': trial.suggest_int('p', 1, 2)                                  # Distance metric: 1=Manhattan, 2=Euclidean
        }

class MLPClassifier(BaseClassifier):
    """
    Multi-layer Perceptron (MLP) classifier implementation using scikit-learn.

    This class extends a base classifier and wraps `sklearn.neural_network.MLPClassifier`.
    It supports hyperparameter configuration, fitting, prediction, and Optuna-based tuning.

    Attributes:
        model (SklearnMLPClassifier): The underlying scikit-learn MLP model.
    """

    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE, **kwargs):
        """
        Initialize the MLP classifier with default or user-defined hyperparameters.

        Args:
            random_state (int): Seed for reproducibility.
            **kwargs: Additional keyword arguments to override default MLP hyperparameters.
        """
        super().__init__(random_state)

        # Default hyperparameters for the MLPClassifier
        default_params = {
            'hidden_layer_sizes': (20,),       # Structure of the hidden layers (e.g., one layer with 20 neurons)
            'activation': 'relu',              # Activation function ('relu', 'tanh', etc.)
            'alpha': 0.0001,                   # L2 regularization term (weight decay)
            'learning_rate': 'constant',       # Learning rate schedule ('constant', 'adaptive', etc.)
            'max_iter': 30000,                 # Maximum number of training iterations
            'solver': 'lbfgs'                  # Optimization algorithm ('adam', 'sgd', 'lbfgs')
        }

        # Override defaults with any provided parameters
        default_params.update(kwargs)

        # Initialize the underlying scikit-learn MLPClassifier
        self.model = SklearnMLPClassifier(random_state=random_state, **default_params)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'MLPClassifier':
        """
        Fit the MLP model to the training data.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).
            **kwargs: Additional arguments passed to the underlying `fit()`.

        Returns:
            MLPClassifier: The fitted model instance.
        """
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels.
        """
        return self.model.predict(X)

    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Retrieve current hyperparameters of the model.

        Args:
            deep (bool): If True, includes nested parameters.

        Returns:
            Dict[str, Any]: Model hyperparameters.
        """
        return self.model.get_params(deep=deep)

    def set_params(self, **params) -> 'MLPClassifier':
        """
        Update the model's hyperparameters.

        Args:
            **params: Key-value pairs of parameters to update.

        Returns:
            MLPClassifier: The model instance with updated parameters.
        """
        self.model.set_params(**params)
        return self

    def _get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest a set of hyperparameters for Optuna optimization.

        Args:
            trial (optuna.Trial): Optuna trial object used to sample hyperparameters.

        Returns:
            Dict[str, Any]: A dictionary of suggested hyperparameters.
        """
        return {
            'hidden_layer_sizes': trial.suggest_categorical(
                'hidden_layer_sizes',
                [(10,), (20,), (30,), (30, 15)]  # Number of neurons per layer and layer depth
            ),
            'activation': trial.suggest_categorical(
                'activation',
                ['relu', 'tanh']  # Common activation functions for MLP
            ),
            'alpha': trial.suggest_float(
                'alpha',
                1e-4, 1e-2, log=True  # L2 regularization strength
            ),
            'learning_rate': trial.suggest_categorical(
                'learning_rate',
                ['constant', 'adaptive']  # Learning rate strategies
            )
        }

class HistGradientBoostingClassifier(BaseClassifier):
    """
    Wrapper for scikit-learn's HistGradientBoostingClassifier.

    This class provides a standardized interface for the Histogram Gradient Boosting
    classifier, including hyperparameter optimization capabilities via Optuna.
    """

    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE, **kwargs):
        """
        Initializes the HistGradientBoostingClassifier.

        Args:
            random_state (int): Seed for random number generation to ensure reproducibility.
            **kwargs: Additional keyword arguments to pass to the underlying
                      SKHistGradientBoostingClassifier constructor.
        """
        super().__init__(random_state)
        
        # Set default hyperparameters for the model
        default_params = {
            'learning_rate': 0.1,
            'max_iter': 150,
            'max_depth': 5,
            'min_samples_leaf': 2
        }
        
        # Override defaults with any user-provided kwargs
        default_params.update(kwargs)
        
        # Initialize the scikit-learn model with the specified parameters
        self.model = SKHistGradientBoostingClassifier(random_state=random_state, **default_params)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'HistGradientBoostingClassifier':
        """
        Fits the HistGradientBoostingClassifier to the training data.

        Args:
            X (np.ndarray): The training input samples.
            y (np.ndarray): The target values.
            **kwargs: Additional keyword arguments to pass to the `fit` method of the
                      underlying scikit-learn model.

        Returns:
            HistGradientBoostingClassifier: The fitted classifier instance.
        """
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for samples in X.

        Args:
            X (np.ndarray): The input samples.

        Returns:
            np.ndarray: Predicted class labels.
        """
        return self.model.predict(X)

    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Gets parameters for the underlying scikit-learn estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and
                         contained subobjects that are estimators.

        Returns:
            Dict[str, Any]: Parameter names mapped to their values.
        """
        return self.model.get_params(deep=deep)

    def set_params(self, **params) -> 'HistGradientBoostingClassifier':
        """
        Sets the parameters of the underlying scikit-learn estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            HistGradientBoostingClassifier: The instance with updated parameters.
        """
        self.model.set_params(**params)
        return self

    def _get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggests hyperparameters for Optuna optimization.

        Args:
            trial (optuna.Trial): An Optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary of suggested hyperparameters.
        """
        return {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_iter': trial.suggest_int('max_iter', 30, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 1.0, log=True),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 5, 20)
        }

def plot_shap_summary(model, X, class_index=1, max_background=100):
    """
    Plot the SHAP summary for a given model, compatible with scikit-learn MLPClassifier.

    Args:
        model: The model to plot the SHAP summary for
        X: The features to plot the SHAP summary for
        class_index: Index of the class to explain (for multiclass)
        max_background: Number of samples to use as background for KernelExplainer
    """
    # Wrapper management
    if hasattr(model, "model"):
        model_to_explain = model.model
    else:
        model_to_explain = model

    # Case for scikit-learn MLP or other natively unsupported models
    model_name = type(model_to_explain).__name__.lower()
    try:
        if "mlp" in model_name:
            # Use KernelExplainer with background data
            background = X[np.random.choice(X.shape[0], min(max_background, X.shape[0]), replace=False)]
            # For binary or multiclass classification
            if hasattr(model_to_explain, "predict_proba"):
                def predict_fn(x):
                    proba = model_to_explain.predict_proba(x)
                    # For binary classification, use proba[:, 1]
                    if proba.shape[1] == 2:
                        return proba[:, 1]
                    # For multiclass, explain the class_index class
                    else:
                        return proba[:, class_index]
            else:
                predict_fn = model_to_explain.predict
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X, nsamples=100)
            # For multiclass, shap_values is a list
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[class_index], X, show=True)
            else:
                shap.summary_plot(shap_values, X, show=True)
        else:
            # Try to use the automatic explainer
            explainer = shap.Explainer(model_to_explain, X)
            shap_values = explainer(X)
            shap.summary_plot(shap_values, X, show=True)
    except Exception as e:
        print(f"[WARN] Unable to create SHAP explainer: {e}")
        return

def evaluate_models_with_scalers(
    X, y, model_types, scaler_dict, classifier_factory, cv=5, scoring='accuracy',
    optimize_hyperparams=False, n_trials=30, cv_optimize=5, verbose=False, shap_analysis=False
):
    """
    Evaluate each combination of scaler + model via cross-validation.
    Can optimize hyperparameters if requested.
    Returns a DataFrame of average scores, the best trained model and the best combination of scaler and model.
    
    Args:
        X: Training features
        y: Training labels
        model_types: List of model types to evaluate
        scaler_dict: Dictionary of scalers to evaluate
        classifier_factory: Factory for creating classifiers
        cv: Number of cross-validation folds
        optimize_hyperparams: Whether to optimize hyperparameters
        n_trials: Number of trials for hyperparameter optimization
        cv_optimize: Number of cross-validation folds for hyperparameter optimization
        verbose: Whether to print verbose output
        shap_analysis: Whether to perform SHAP analysis on the best model
    Returns:
        results: DataFrame of average scores
        best_model: Best trained model
        best_combo: Best combination of scaler and model
    """
    results = {}
    best_score = -float('inf')
    best_combo = (None, None)
    
    for scaler_name, scaler in scaler_dict.items():
        if scaler_name == 'NoScaler':
            X_scaled = X.copy()
        else:
            X_scaled = scaler.fit_transform(X)
        scaler_results = {}
        for model_type in model_types:
            clf = classifier_factory.create(model_type)
            # Hyperparameter optimization if requested and possible
            if optimize_hyperparams and hasattr(clf, 'optimize_hyperparameters'):
                try:
                    clf.optimize_hyperparameters(
                        X_scaled, y, n_trials=n_trials, cv=cv_optimize
                    )
                except Exception as e:
                    print(f"[WARN] Optimization failed for {model_type} with {scaler_name}: {e}")
            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
            scores = cross_val_score(clf, X_scaled, y, cv=kf, scoring=scoring)
            mean_score = scores.mean()
            scaler_results[model_type] = mean_score

            if mean_score > best_score:
                best_score = mean_score
                best_combo = (scaler_name, model_type)
        
        results[scaler_name] = scaler_results

    # Retrain only the best model on the complete data
    best_scaler_name, best_model_type = best_combo
    best_scaler = scaler_dict[best_scaler_name]
    if best_scaler_name == 'NoScaler':
        X_best_scaled = X.copy()
    else:
        X_best_scaled = best_scaler.fit_transform(X)
    best_model = classifier_factory.create(best_model_type)
    # Optimization on the entire dataset if requested
    if optimize_hyperparams and hasattr(best_model, 'optimize_hyperparameters'):
        try:
            best_model.optimize_hyperparameters(
                X_best_scaled, y, n_trials=n_trials, cv=cv_optimize
            )
        except Exception as e:
            print(f"[WARN] Optimization failed for the best model: {e}")
    best_model.fit(X_best_scaled, y)

    if len(model_types) == 1 and len(scaler_dict) == 1:
        if verbose:
            # Display for the best model/scaler only
            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
            # Use best_model.model if best_model is a wrapper, otherwise use best_model directly
            model_for_cv = getattr(best_model, "model", best_model)
            y_pred_cv = cross_val_predict(model_for_cv, X_best_scaled, y, cv=kf, n_jobs=-1)
            cm = confusion_matrix(y, y_pred_cv)

            # Learning curve
            train_sizes, train_scores, test_scores = learning_curve(
                model_for_cv, X_best_scaled, y, cv=kf, scoring='accuracy', n_jobs=-1
            )
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            plot_confusion_matrix_and_learning_curve(cm, train_sizes, train_mean, train_std, test_mean, test_std)

            # Affichage des indices mal classés
            misclassified_indices = np.where(y != y_pred_cv)[0]
            print("Misclassified samples indices:", misclassified_indices)

        if shap_analysis:
            print("Launching SHAP analysis for the best model...")
            plot_shap_summary(best_model, X_best_scaled)

    return pd.DataFrame(results), best_model, best_scaler_name, best_model_type, best_score
