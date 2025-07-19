"""
Implementation of various classifiers.
"""
import shap
import optuna
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from typing import Dict, Any, Optional, Tuple
from sklearn.feature_selection import chi2
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
    """Base class for all classifiers."""
    
    def optimize_hyperparameters(self, 
                               X: np.ndarray, 
                               y: np.ndarray,
                               n_trials: int = 50,
                               cv: int = DEFAULT_CV_FOLDS) -> Tuple[float, Dict[str, Any]]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X: Training features
            y: Training labels
            n_trials: Number of optimization trials
            cv: Number of cross-validation folds
            
        Returns:
            Tuple of (best score, best parameters)
        """
        def objective(trial):
            params = self._get_trial_params(trial)
            self.set_params(**params)
            
            # Perform cross-validation
            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(self.model, X, y, cv=kf, scoring='accuracy', n_jobs=-1)
            
            return np.mean(scores)
            
        # Create and run study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        # Sauvegarde des visualisations dans des fichiers HTML
        optuna.visualization.plot_optimization_history(study).write_html("results/optuna/optuna_optimization_history.html")
        optuna.visualization.plot_param_importances(study).write_html("results/optuna/optuna_param_importances.html")
        optuna.visualization.plot_parallel_coordinate(study).write_html("results/optuna/optuna_parallel_coordinate.html")
        optuna.visualization.plot_contour(study).write_html("results/optuna/optuna_contour.html")
        optuna.visualization.plot_slice(study).write_html("results/optuna/optuna_slice.html")

        
        # Set best parameters
        self.set_params(**study.best_params)
        
        return study.best_value, study.best_params
    
    def _get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get parameters for a trial. To be implemented by subclasses."""
        raise NotImplementedError

class RFClassifier(BaseClassifier):
    """Random Forest classifier implementation."""
    
    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE, **kwargs):
        super().__init__(random_state)
        default_params = {
            'n_estimators': 80,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'bootstrap': True
        }
        default_params.update(kwargs)
        self.model = RandomForestClassifier(random_state=random_state, **default_params)
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'RFClassifier':
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def get_params(self, deep=True) -> Dict[str, Any]:
        return self.model.get_params(deep=deep)
        
    def set_params(self, **params) -> 'RFClassifier':
        self.model.set_params(**params)
        return self
        
    def get_feature_importance(self) -> Optional[Tuple[np.ndarray, list]]:
        if self.is_fitted and self.feature_names:
            return self.model.feature_importances_, self.feature_names
        return None
        
    def _get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 30, 80),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }

class SVMClassifier(BaseClassifier):
    """Support Vector Machine classifier implementation."""

    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE, probability: bool = True, **kwargs):
        super().__init__(random_state)
        default_params = {
            'C': 11.3,
            'kernel': 'rbf',
            'gamma': 0.12,
            'class_weight': 'balanced'
        }
        default_params.update(kwargs)
        self.model = SVC(random_state=random_state, probability=probability, **default_params)
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SVMClassifier':
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def get_params(self, deep=True) -> Dict[str, Any]: 
        return self.model.get_params(deep=deep)
        
    def set_params(self, **params) -> 'SVMClassifier':
        self.model.set_params(**params)
        return self
        
    def _get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'C': trial.suggest_float('C', 11, 12, log=True),
            'gamma': trial.suggest_float('gamma', 0.09, 0.2, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf'])
        }

class KNNClassifier(BaseClassifier):
    """K-Nearest Neighbors classifier implementation."""
        
    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE, **kwargs):
        super().__init__(random_state)
        default_params = {
            'n_neighbors': 7,
            'weights': 'uniform',
            'p': 1
        }
        default_params.update(kwargs)
        self.model = KNeighborsClassifier(**default_params)
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'KNNClassifier':
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def get_params(self, deep=True) -> Dict[str, Any]:
        return self.model.get_params(deep=deep)
        
    def set_params(self, **params) -> 'KNNClassifier':
        self.model.set_params(**params)
        return self
        
    def _get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', 5, 10),
            'weights': trial.suggest_categorical('weights', ['uniform']),
            'p': trial.suggest_int('p', 1, 2)
        }

class MLPClassifier(BaseClassifier):
    """Multi-layer Perceptron classifier implementation."""
    
    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE, **kwargs):
        super().__init__(random_state)
        default_params = {
            'hidden_layer_sizes': (20,),
            'activation': 'relu',
            'alpha': 0.0001,
            'learning_rate': 'constant',
            'max_iter': 30000,
            'solver': 'lbfgs'
        }
        default_params.update(kwargs)
        self.model = SklearnMLPClassifier(random_state=random_state, **default_params)
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'MLPClassifier':
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def get_params(self, deep=True) -> Dict[str, Any]:
        return self.model.get_params(deep=deep)
        
    def set_params(self, **params) -> 'MLPClassifier':
        self.model.set_params(**params)
        return self
        
    def _get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'hidden_layer_sizes': trial.suggest_categorical(
                'hidden_layer_sizes',
                [(10,), (20,), (30,), (30, 15)]
            ),
            'activation': trial.suggest_categorical(
                'activation',
                ['relu', 'tanh']
            ),
            'alpha': trial.suggest_float('alpha', 1e-4, 1e-2, log=True),
            'learning_rate': trial.suggest_categorical(
                'learning_rate',
                ['constant', 'adaptive']
            )
        }

class HistGradientBoostingClassifier(BaseClassifier):
    """Histogram Gradient Boosting classifier implementation."""

    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE, **kwargs):
        super().__init__(random_state)
        default_params = {
            'learning_rate': 0.1,
            'max_iter': 150,
            'max_depth': 5,
            'min_samples_leaf': 2
        }
        default_params.update(kwargs)
        self.model = SKHistGradientBoostingClassifier(random_state=random_state, **default_params)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'HistGradientBoostingClassifier':
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_params(self, deep=True) -> Dict[str, Any]:
        return self.model.get_params(deep=deep)

    def set_params(self, **params) -> 'HistGradientBoostingClassifier':
        self.model.set_params(**params)
        return self

    def _get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
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
    # Gestion des wrappers
    if hasattr(model, "model"):
        model_to_explain = model.model
    else:
        model_to_explain = model

    # Cas des modèles scikit-learn MLP ou autres non supportés nativement
    model_name = type(model_to_explain).__name__.lower()
    try:
        if "mlp" in model_name:
            # Utiliser KernelExplainer avec un background data
            background = X[np.random.choice(X.shape[0], min(max_background, X.shape[0]), replace=False)]
            # Pour la classification binaire ou multiclasse
            if hasattr(model_to_explain, "predict_proba"):
                def predict_fn(x):
                    proba = model_to_explain.predict_proba(x)
                    # Pour la classification binaire, proba[:, 1]
                    if proba.shape[1] == 2:
                        return proba[:, 1]
                    # Pour le multiclasse, expliquer la classe class_index
                    else:
                        return proba[:, class_index]
            else:
                predict_fn = model_to_explain.predict
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X, nsamples=100)
            # Pour le multiclasse, shap_values est une liste
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[class_index], X, show=True)
            else:
                shap.summary_plot(shap_values, X, show=True)
        else:
            # Essayer d'utiliser l'explainer automatique
            explainer = shap.Explainer(model_to_explain, X)
            shap_values = explainer(X)
            shap.summary_plot(shap_values, X, show=True)
    except Exception as e:
        print(f"[WARN] Impossible de créer un explainer SHAP : {e}")
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
            # Optimisation des hyperparamètres si demandé et si possible
            if optimize_hyperparams and hasattr(clf, 'optimize_hyperparameters'):
                try:
                    clf.optimize_hyperparameters(
                        X_scaled, y, n_trials=n_trials, cv=cv_optimize
                    )
                except Exception as e:
                    print(f"[WARN] Optimisation échouée pour {model_type} avec {scaler_name}: {e}")
            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
            scores = cross_val_score(clf, X_scaled, y, cv=kf, scoring=scoring)
            mean_score = scores.mean()
            scaler_results[model_type] = mean_score

            if mean_score > best_score:
                best_score = mean_score
                best_combo = (scaler_name, model_type)
        
        results[scaler_name] = scaler_results

    # Réentraîner uniquement le meilleur modèle sur les données complètes
    best_scaler_name, best_model_type = best_combo
    best_scaler = scaler_dict[best_scaler_name]
    if best_scaler_name == 'NoScaler':
        X_best_scaled = X.copy()
    else:
        X_best_scaled = best_scaler.fit_transform(X)
    best_model = classifier_factory.create(best_model_type)
    # Optimisation sur tout le jeu si demandé
    if optimize_hyperparams and hasattr(best_model, 'optimize_hyperparameters'):
        try:
            best_model.optimize_hyperparameters(
                X_best_scaled, y, n_trials=n_trials, cv=cv_optimize
            )
        except Exception as e:
            print(f"[WARN] Optimisation échouée pour le meilleur modèle: {e}")
    best_model.fit(X_best_scaled, y)

    if len(model_types) == 1 and len(scaler_dict) == 1:
        if verbose:
            # Affichage pour le meilleur modèle/scaler uniquement
            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
            # Utiliser best_model.model si best_model est un wrapper, sinon best_model directement
            model_for_cv = getattr(best_model, "model", best_model)
            y_pred_cv = cross_val_predict(model_for_cv, X_best_scaled, y, cv=kf, n_jobs=-1)
            cm = confusion_matrix(y, y_pred_cv)

            # Courbe d'apprentissage
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
            print("Lancement de l'analyse SHAP pour le meilleur modèle...")
            plot_shap_summary(best_model, X_best_scaled)

    return pd.DataFrame(results), best_model, best_scaler_name, best_model_type, best_score
