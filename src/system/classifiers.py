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
from sklearn.metrics import confusion_matrix, make_scorer, recall_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import FunctionTransformer
from boruta import BorutaPy

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
        cv: int = DEFAULT_CV_FOLDS,
        scoring: str = 'accuracy',
        direction: str = 'maximize'
    ) -> Tuple[float, Dict[str, Any]]:
        ...
        def objective(trial):
            params = self._get_trial_params(trial)
            self.set_params(**params)
            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(self.model, X, y, cv=kf, scoring=scoring, n_jobs=-1)
            return np.mean(scores)

        # Create a study aiming to maximize accuracy
        study = optuna.create_study(direction=direction)
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
    X, y, model_types, scaler_dict, classifier_factory,
    cv=5, scoring='accuracy',
    optimize_hyperparams=False, n_trials=30, cv_optimize=5,
    verbose=False, shap_analysis=False,
    feature_selection_method='lasso', k_features=20,
    random_state=42
):
    """
    Évalue chaque combinaison scaler + feature_selector + model via cross-validation
    (la sélection de features est refaite dans chaque fold).
    Retourne DataFrame des scores moyens, le pipeline entraîné (scaler+selector+model),
    le meilleur scaler & modèle, et le meilleur score.
    """
    results = {}
    best_score = -float('inf')
    best_combo = (None, None)
    best_pipeline = None

    # set scorer
    if scoring == 'all_mutants':
        scorer = make_scorer(recall_score, pos_label=1)
    elif scoring == 'all_wts':
        scorer = make_scorer(recall_score, pos_label=0)
    elif scoring == 'accuracy':
        scorer = 'accuracy'
    else:
        scorer = scoring

    for scaler_name, scaler in scaler_dict.items():
        scaler_results = {}
        for model_type in model_types:
            clf = classifier_factory.create(model_type)

            # build pipeline: scaler + selector + classifier
            selector = make_feature_selector(method=feature_selection_method, k=k_features, random_state=random_state)
            steps = []
            # NoScaler may be a FunctionTransformer(identity) — still ok in pipeline
            steps.append(('scaler', scaler))
            steps.append(('selector', selector))
            steps.append(('clf', clf))
            pipe = Pipeline(steps)

            # optionally optimize hyperparams inside the pipeline (if classifier supports)
            if optimize_hyperparams and hasattr(clf, 'optimize_hyperparameters'):
                try:
                    # caller must handle how optimization is applied to the pipeline;
                    clf.optimize_hyperparameters(X, y, n_trials=n_trials, cv=cv_optimize, scoring=scoring)
                except Exception as e:
                    print(f"[WARN] Optimization failed for {model_type} with {scaler_name}: {e}")

            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
            try:
                scores = cross_val_score(pipe, X, y, cv=kf, scoring=scorer, n_jobs=-1)
                mean_score = scores.mean()
            except Exception as e:
                print(f"[ERROR] cross_val_score failed for {model_type} with {scaler_name}: {e}")
                mean_score = np.nan

            scaler_results[model_type] = mean_score

            if mean_score is not None and not np.isnan(mean_score) and mean_score > best_score:
                best_score = mean_score
                best_combo = (scaler_name, model_type)
                best_pipeline = pipe  # pipeline, but not yet fitted

        results[scaler_name] = scaler_results

    # Retrain only the best pipeline on the complete data (fit scaler+selector+clf on full X)
    if best_pipeline is None:
        raise RuntimeError("No valid pipeline found (best_pipeline is None).")

    # If best_scaler_name is NoScaler (FunctionTransformer), pipeline still works
    # Fit best_pipeline on full dataset
    best_pipeline.fit(X, y)

    best_scaler_name, best_model_type = best_combo
    
    
    # Si on n'a testé qu'un seul scaler et un seul modèle, afficher courbes + confusion
    if verbose and len(model_types) == 1 and len(scaler_dict) == 1:
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        # cross_val_predict sur le pipeline complet
        y_pred_cv = cross_val_predict(best_pipeline, X, y, cv=kf, n_jobs=-1)
        
        # Matrice de confusion
        cm = confusion_matrix(y, y_pred_cv)
        
        # Courbe d'apprentissage
        train_sizes, train_scores, test_scores = learning_curve(
            best_pipeline, X, y, cv=kf, scoring='accuracy', n_jobs=-1
        )
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Affichage (réutiliser ta fonction ou matplotlib)
        plot_confusion_matrix_and_learning_curve(cm, train_sizes, train_mean, train_std, test_mean, test_std)

        # Afficher indices mal classés
        misclassified_indices = np.where(y != y_pred_cv)[0]
        print("Misclassified samples indices:", misclassified_indices)

        # Option SHAP
        if shap_analysis:
            print("Launching SHAP analysis for the best model...")
            plot_shap_summary(best_pipeline, X)

    
    
    
    # Return DataFrame results, trained pipeline (scaler+selector+clf), best scaler/model names and score
    return pd.DataFrame(results), best_pipeline, best_scaler_name, best_model_type, best_score

    
class LassoFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cv=5, random_state=42):
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        self.model_ = LassoCV(cv=self.cv, random_state=self.random_state, n_jobs=-1).fit(X, y)
        self.mask_ = self.model_.coef_ != 0
        return self

    def transform(self, X):
        return X[:, self.mask_]

    def get_support(self, indices=False):
        if indices:
            return np.where(self.mask_)[0]
        return self.mask_

class ElasticNetFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cv=5, random_state=42, l1_ratio=[.1, .5, .7, .9, .95, .99, 1]):
        self.cv = cv
        self.random_state = random_state
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        self.model_ = ElasticNetCV(
            cv=self.cv,
            l1_ratio=self.l1_ratio,
            random_state=self.random_state
        ).fit(X, y)
        # mask of selected features
        self.mask_ = self.model_.coef_ != 0
        return self

    def transform(self, X):
        return X[:, self.mask_]

    def get_support(self, indices=False):
        if indices:
            return np.where(self.mask_)[0]
        return self.mask_

class BorutaFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, rf_estimator, n_estimators='auto', verbose=0, random_state=42):
        self.rf_estimator = rf_estimator
        self.n_estimators = n_estimators
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y):
        self.boruta_ = BorutaPy(
            self.rf_estimator,
            n_estimators=self.n_estimators,
            verbose=self.verbose,
            random_state=self.random_state
        )
        self.boruta_.fit(X, y)
        self.mask_ = self.boruta_.support_
        return self

    def transform(self, X):
        return X[:, self.mask_]

    def get_support(self, indices=False):
        if indices:
            return np.where(self.mask_)[0]
        return self.mask_

def make_feature_selector(method='lasso', k=20, random_state=42):
    """
    Retourne un objet transformer compatible sklearn selon la méthode.
    method: 'kbest', 'lasso', 'elasticnet', 'boruta', 'none'
    """
    method = method.lower()
    if method == 'kbest':
        return SelectKBest(score_func=f_classif, k=k)
    elif method == 'lasso':
        return LassoFeatureSelector(cv=5, random_state=random_state)
    elif method == 'elasticnet':
        return ElasticNetFeatureSelector(cv=5, random_state=random_state)
    elif method == 'boruta':
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_jobs=-1, random_state=random_state, max_depth=5)
        return BorutaFeatureSelector(rf_estimator=rf, n_estimators='auto', verbose=0, random_state=random_state)
    elif method == 'none':
        # Pas de sélection : identite
        return FunctionTransformer(lambda X: X, validate=False)
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    