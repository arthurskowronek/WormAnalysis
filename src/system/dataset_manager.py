"""
Dataset management.
"""
import os
import math
import joblib
import shutil
import random
import pathlib
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from skimage.transform import resize
from typing import Dict, List, Tuple, Optional
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, QuantileTransformer, FunctionTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

from src.system.data import Data
from src.system.visualization import plot_heatmap, plot_synapse_detection
from src.system.preprocessing import Preprocessing
from src.system.classifiers import ClassifierFactory, evaluate_models_with_scalers
from src.system.outlier import MahalanobisOutlierDetector
from src.system.features import FeatureExtractor

from config import DATA_DIR, TRAINING_DIR, IMAGE_SIZE, DEFAULT_PKL_NAME, MODELS_DIR, DATE_FORMAT

class Dataset_Manager:
    """
    A class for managing dataset loading, preprocessing, and model training.

    This class handles the entire workflow from loading raw images and
    pre-processed data, to extracting features, performing feature selection,
    and training and evaluating machine learning models.
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        """
        Initializes the Dataset_Manager.

        Args:
            data_dir (Path): The root directory where the dataset is stored.
        """
        # The main directory containing the raw data.
        self.data_dir = Path(data_dir)
        # The directory for storing cached, processed dataset files (e.g., .pkl).
        self.dataset_pkl_dir = Path(data_dir) / "Dataset_pkl"
        # A list to hold the Data objects, each representing a single sample.
        self.data = []

    def add_data(self, data_item: 'Data'):
        """
        Adds a single Data item to the dataset.

        Args:
            data_item (Data): An instance of the Data class to be added.
        """
        self.data.append(data_item)
        
    def load_images(self, 
                compute: bool = True, 
                test_mode: bool = False,
                model_name = None,
                visualize: bool = False,
                training: bool = False,
                validation: bool = False,
                name_dataset: str = DEFAULT_PKL_NAME,
                verbose: bool = False) -> 'Dataset_Manager':
        """
        Loads and preprocesses images from the specified data directories.

        - If model_name is None: behave exactly as before (use self.data_dir and existing logic).
        - If model_name is not None and corresponds to a folder inside TRAINING_DIR (or self.TRAINING_DIR
        if present), use that folder as the base. That folder should contain 'Mutant' and 'WT'.
        In that case, the function will ensure a 'validation' subfolder exists inside the model folder
        and populate it by copying a deterministic subset of images from 'Mutant' and 'WT' (20% by default,
        at least 1 per class). After that the loading logic is the same (it will read Mutant/WT or validation
        depending on the flags).
        - If the named model folder does not exist or is invalid, the function falls back to the original
        behavior and prints a warning.
        """
        moved_from_mutant: List[str] = []
        moved_from_wt: List[str] = []
    
        pkl_path = self.dataset_pkl_dir / (name_dataset + ".pkl")
        
        # Attempt to load cached data unless 'compute' is forced to True.
        if not compute and pkl_path.exists():
            try:
                self.data = joblib.load(pkl_path)
                print("Loaded cached data from", pkl_path)
            except Exception as e:
                print(f"Error loading cached data: {e}")
                print("Proceeding with fresh data loading...")
                # Fallback to fresh computation if loading fails.
                compute = True
                name_dataset = DEFAULT_PKL_NAME
                pkl_path = self.dataset_pkl_dir / (name_dataset + ".pkl")
        else:
            if training:
                print("No cached data found, proceeding with fresh data loading...")
            compute = True
            name_dataset = DEFAULT_PKL_NAME
            pkl_path = self.dataset_pkl_dir / (name_dataset + ".pkl")

        # Decide base directory depending on model_name (SIMPLE: only detect, do NOT create validation)
        use_model_folder = False
        model_base_dir = None
        if model_name is not None:
            candidate = TRAINING_DIR / str(model_name)
            if candidate.exists() and candidate.is_dir():
                # require both classes to exist to be considered a valid model folder
                if (candidate / "Mutant").exists() and (candidate / "WT").exists():
                    use_model_folder = True
                    model_base_dir = candidate
                    print(f"Using model folder as base directory: {model_base_dir}")
                else:
                    print(f"Warning: model folder {candidate} found but missing 'Mutant' or 'WT' subfolders. Falling back to {self.data_dir}.")
            else:
                print(f"Warning: model folder {candidate} not found. Falling back to {self.data_dir}.")

        if compute:
            print("Acquiring data...")
            # Define which directories to process based on 'training' mode.
            if not training:
                label_dirs = ['Unclassified']
            else:
                label_dirs = ['Mutant', 'WT']
            if validation:
                label_dirs = ['validation']

            # choose the base directory to look into: either model_base_dir (if used) or self.data_dir
            base_dir = model_base_dir if use_model_folder and model_base_dir is not None else Path(self.data_dir)

            ind=0
            for label_dir in label_dirs:
                dir_path = base_dir / label_dir

                if not dir_path.exists():
                    # skip missing directories (preserves previous behavior)
                    continue

                # Iterate through all .tif images in the directory.
                
                for img_path in dir_path.glob('*.tif'):
                    print(f"Processing image {ind + 1} : {img_path}")
                    ind +=1
                    try:
                        # Basic image loading and preprocessing
                        img = imread(img_path)
                        img = self._preprocess_image(img)
                        preprocessing = Preprocessing()

                        # Get worm mask
                        worm_mask = preprocessing.worm_segmentation(img)

                        # Skip coiled worms if requested
                        if not preprocessing.is_coiled_worm(worm_mask):
                            # Get synapse data
                            maxima, graph, median_width, diff_slice, diff_segment, NUMBER_OF_CORDS = preprocessing.get_synapse_using_graph(img, worm_mask, verbose=verbose)
                            if len(maxima) <= 10:
                                coiled = True
                            else:
                                coiled = False
                        else:
                            if verbose: print(f"Skipping coiled worm in {img_path}")
                            empty_graph = nx.Graph()
                            maxima = []
                            graph = empty_graph
                            median_width = 0
                            diff_slice = 0
                            diff_segment = 0 
                            coiled = True
                            NUMBER_OF_CORDS = 1
                            
                        if validation:
                            # When reading from validation directory, attempt to infer label from filename:
                            # fall back to the label_dir if inference fails.
                            inferred_label = 'Mutant' if 'Mut' in img_path.name else 'WT'
                            label_for_data = inferred_label if inferred_label in ('Mutant', 'WT') else label_dir
                        else:
                            label_for_data = label_dir

                        # Create a new Data object and populate its attributes.
                        new_data = Data()
                        new_data.label = label_for_data
                        new_data.filename = img_path.name
                        new_data.original_image = img
                        if new_data.label == "Mutant":
                            # Extract mutant type from the filename (preserve original behaviour).
                            # keep the same slice indices as before
                            new_data.mutant_type = img_path.name[5:9]
                        new_data.worm_mask = worm_mask
                        new_data.maxima = maxima
                        new_data.graph = graph
                        new_data.median_width = median_width
                        new_data.diff_slice = diff_slice
                        new_data.diff_segment = diff_segment
                        new_data.coiled = coiled
                        new_data.number_of_cords = NUMBER_OF_CORDS
                        self.add_data(new_data)

                        if test_mode and len(self.data) >= 2:
                            break

                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")

        # Save processed dataset to a pickle file for caching.
        if not test_mode and training and compute:
            try:
                joblib.dump(self.data, pkl_path)
            except Exception as e:
                print(f"Error saving dataset: {e}")

        # Visualize preprocessing results if requested.
        enhanced_img = None
        if visualize:
            print("\nVisualizing preprocessing results...")
            count = 0
            for data in self.data:
                img_name = data.filename
                img = data.original_image
                mask = data.worm_mask
                max_data = data.maxima
                enhanced_img = plot_synapse_detection(
                    original_image=img,
                    worm_mask=mask,
                    maxima=max_data,
                    title=f'{img_name}',
                    display=False
                )
                count += 1
                if count >= 3:
                    break    

        print("Data acquired successfully.")
        return self, moved_from_mutant, moved_from_wt, enhanced_img

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the original images and their corresponding labels from the dataset.
        
        Raises:
            ValueError: If the dataset is empty.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing a list of original images
                                           and a list of their labels.
        """
        if not self.data:
            raise ValueError("Get_data : Dataset is empty. Call load_images() first.")
            
        X = [] # original images
        y = [] # labels
        for item in self.data:
            img, label = item.get_original_data()
            X.append(img)
            y.append(label)
        
        return X, y
    
    def update_label_by_filename(self, filename: str, new_label: str, new_filename: str) -> bool:
        """
        Updates the label and filename of a data item.
        
        Args:
            filename (str): The current filename of the image to update.
            new_label (str): The new label to assign to the data item.
            new_filename (str): The new filename to assign to the data item.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        for item in self.data:
            print(item.filename)
            if item.filename == filename:
                old_label = item.label
                item.label = new_label
                item.filename = new_filename
                print(f"Label updated for '{filename}': '{old_label}' → '{new_label}'")
                return True
        print(f"Filename '{filename}' not found in dataset.")
        return False
 
    def get_coiled_worms(self) -> List[int]:
        """
        Retrieves the indices of all coiled worms in the dataset.
        
        Raises:
            ValueError: If the dataset is empty.

        Returns:
            List[int]: A list of integer indices corresponding to the coiled worm data items.
        """
        if not self.data:
            raise ValueError("Get_coiled : Dataset is empty. Call load_images() first.")
            
        coiled_worms = []
        # get index of coiled worms
        for i, item in enumerate(self.data):
            if item.coiled:
                coiled_worms.append(i)
        
        return coiled_worms
  
    def get_median_width(self) -> List[float]:
        """
        Retrieves the median width for each worm in the dataset.
        
        Raises:
            ValueError: If the dataset is empty.

        Returns:
            List[float]: A list of median width values.
        """
        if not self.data:
            raise ValueError("Get_median : Dataset is empty. Call load_images() first.")
            
        median_widths = []
        for item in self.data:
            median_widths.append(item.median_width)
        
        return median_widths
    
    def get_features(self) -> Tuple[np.ndarray, list]:
        """
        Retrieves the complete set of computed features and their names.
        
        Returns:
            Tuple[np.ndarray, list]: A tuple containing a numpy array of all
                                     features and a list of their names.
        """
        features = []

        for item in self.data:
            features.append(item.get_features()[0])
            
        features = np.array(features) 
        feature_names = self._get_feature_names_selected()
                
        return features, feature_names
    
    def get_features_selected(self) -> Tuple[np.ndarray, list]:
        """
        Retrieves the subset of selected features and their names.
        
        Returns:
            Tuple[np.ndarray, list]: A tuple containing a numpy array of the
                                     selected features and a list of their names.
        """
        features = []
        
        for item in self.data:
            if item.coiled == False:
                features.append(item.get_features_selected()[0])
   
        features = np.array(features)
        feature_names = self._get_feature_names_selected()
        
        return features, feature_names
    
    def _get_feature_names_selected(self):
        for item in self.data:
            try:
                fn = item.get_features_selected()[1]
                if fn is not None and len(fn) > 0:
                    return fn
            except Exception:
                continue
        raise ValueError("No feature names found (no non-coiled item with selected features).")
    
    def set_features(self, 
                     compute: bool = True, 
                     save_features: bool = False, 
                     name_dataset: str = DEFAULT_PKL_NAME,  
                     feature_reduction: bool = False, 
                     selection_method: str = "none", 
                     verbose: bool = False) -> None:
        """
        Computes, selects, and sets the features for the dataset.

        This method handles feature extraction from raw data, applies feature
        selection if requested, and stores the results in the dataset.

        Args:
            compute (bool): If True, re-computes features from the raw data.
                            Defaults to True.
            save_features (bool): If True, saves the computed features to a
                                  pickle file. Defaults to False.
            name_dataset (str): The name of the dataset file. Defaults to DEFAULT_PKL_NAME.
            feature_reduction (bool): DEPRECATED. Use `selection_method` instead.
                                      Defaults to False.
            selection_method (str): The method to use for feature selection.
                                    Options include 'saved', 'none'. But 'none' has to be always used. 
                                    Because the selection is made in the pipeline
            verbose (bool): If True, prints additional information during the process.
                            Defaults to False.
        
        Raises:
            ValueError: If an unknown feature selection method is specified.
        """
        pkl_path = self.dataset_pkl_dir / (name_dataset + "_features.pkl")
        pkl_big_dataset_path = self.dataset_pkl_dir / (name_dataset + ".pkl")
        # Attempt to load cached features unless 'compute' is forced.
        if not compute and (pkl_path.exists() or pkl_big_dataset_path.exists()):
            try:
                self.data = joblib.load(pkl_path)
                feature_extractor = FeatureExtractor(self.data)
                print("Loaded cached data from", pkl_path)
            except Exception as e:
                compute = True
                print(f"Error loading cached data: {e}")
                print("Proceeding with fresh data loading...")
        else:
            if verbose: 
                print("No cached data found, proceeding with fresh data loading...")
            compute = True
            name_dataset = DEFAULT_PKL_NAME
            pkl_path = self.dataset_pkl_dir / (name_dataset + "_features.pkl")
        
                    
        if compute:
            print("Computing features...")
            # Initialize a feature extractor and compute features.
            feature_extractor = FeatureExtractor()
            feature_extractor.set_features(self)
            features, feature_names = feature_extractor.get_features()
            # Assign computed features to each data item.
            for item, feature_vector in zip(self.data, features):
                if not item.coiled:
                    item.set_features(feature_vector, feature_names)
            # Save the updated dataset if requested.
            if save_features:
                try:
                    joblib.dump(self.data, self.dataset_pkl_dir / (name_dataset + "_features.pkl"))
                except Exception as e:
                    print(f"Error saving dataset with features: {e}")
        
        # -------------------------
        # Handle feature selection
        # -------------------------
        # NOTE: Selection methods that use labels (kbest, lasso, boruta, mRMR, elasticnet)
        # MUST be applied during training inside the pipeline (evaluate_models_with_scalers)
        # to avoid data leakage. 
        # set_features() will only:
        #  - compute raw features (done above),
        #  - optionally apply 'none' (i.e. keep all features) for bookkeeping,
        #  - or apply 'saved' to reproduce a previously stored subset (for inference).
        if feature_reduction:
            print("Applying feature reduction...")
            _, y = self.get_data()
            indices_coiled = self.get_coiled_worms()
            feature_extractor.feature_reduction(y, indices_coiled)
        elif selection_method == 'none':
            # No selection: just compute scaled features for storage (optional)
            _, y = self.get_data()
            indices_coiled = self.get_coiled_worms()
            features, feature_names, y = feature_extractor._process_features(y, indices_coiled)

            # handle NaNs then scale
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
            features = np.nan_to_num(features)

            count = 0
            for item in self.data:
                if (not item.coiled) and (item.coiled is not None):
                    item.set_features_selected(features[count], feature_names)
                    count += 1
            return self
        elif selection_method == 'saved':
            # Load a pre-computed list of selected indices and apply them (for inference reproduction)
            if verbose: print("Loading saved features...")
            _, y = self.get_data()
            indices_coiled = self.get_coiled_worms()
            features, feature_names, y = feature_extractor._process_features(y, indices_coiled)

            features = np.nan_to_num(features)
            scaler = StandardScaler()
            features = scaler.fit_transform(features)

            script_path = pathlib.Path(__file__).resolve()
            root_dir = script_path.parent.parent.parent
            file_path = root_dir / 'models' / 'selected_features.txt'
            if not file_path.exists():
                raise FileNotFoundError(f"selected_features.txt not found at {file_path}")
            with open(file_path, 'r') as f:
                selected_indices = [int(line.strip()) for line in f]

            features = features[:, selected_indices]
            feature_names = [feature_names[i] for i in selected_indices]
            count = 0
            for item in self.data:
                if not item.coiled:
                    item.set_features_selected(features[count], feature_names)
                    count += 1
            return self
        else:
            # If user asked for an in-training selection method, skip it here and instruct.
            if selection_method in ['kbest', 'boruta', 'mRMR', 'elasticnet', 'lasso']:
                print(f"Selection method '{selection_method}' must be applied during training (get_model) inside the pipeline to avoid data leakage. Skipping selection in set_features().")
                return self
            raise ValueError(f"Unknown feature selection method: {selection_method}")
  
    def make_unsupervised_selector(self, method: str, X_fit: np.ndarray, k: int = 20, var_threshold: float = 1e-5):
        """
        Retourne un selector non-supervisé fit sur X_fit.
        method: 'none', 'variance', 'pca'
        - 'variance' : VarianceThreshold(threshold=var_threshold)
        - 'pca'      : PCA(n_components=k)
        """
        method = (method or 'none').lower()
        if method in ('none', ''):
            return None
        if method == 'variance':
            sel = VarianceThreshold(threshold=var_threshold)
            sel.fit(X_fit)
            return sel
        if method == 'pca':
            n_comp = min(k, X_fit.shape[1])
            sel = PCA(n_components=n_comp)
            sel.fit(X_fit)
            
            # a) Variance expliquée par composante
            explained_variance_ratio = sel.explained_variance_ratio_
            # b) Variance cumulée
            cumulative_variance = np.cumsum(explained_variance_ratio)
            # c) Génération du graphique
            plt.figure(figsize=(10, 6))
            # Graphique de l'éboulis des valeurs propres (variance par composante)
            plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.6, 
                    color='g', label='Variance expliquée individuelle')
            # Graphique de la variance cumulée
            plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance, 
                    marker='o', linestyle='--', color='r', label='Variance expliquée cumulée')
            plt.title('Analyse de la Variance Expliquée pour la Détermination de n_components')
            plt.xlabel('Numéro de la Composante Principale')
            plt.ylabel('Proportion de Variance Expliquée')
            plt.xticks(range(1, len(explained_variance_ratio) + 1), rotation=45)
            plt.grid(axis='y', linestyle='--')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            return sel
        if method == 'variance_pca':
            # --- Définir les étapes du Pipeline ---
            variance_step = ('variance_threshold', VarianceThreshold(threshold=var_threshold))
            n_comp = min(k, X_fit.shape[1]) 
            pca_step = ('pca', PCA(n_components=n_comp))
            
            # --- Créer et Entraîner le Pipeline ---
            pipeline = Pipeline(steps=[
                variance_step,
                pca_step
            ])
            pipeline.fit(X_fit)
            
            # --- Générer le graphique ---
            
            # 1. Extraire l'objet PCA entraîné du pipeline
            pca_model = pipeline.named_steps['pca']
            
            # 2. Récupérer les données de variance
            explained_variance_ratio = pca_model.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            # 3. Génération du graphique d'éboulis
            plt.figure(figsize=(10, 6))

            # Graphique de l'éboulis des valeurs propres (variance par composante)
            # Note : On affiche les composantes qui existent après le VarianceThreshold
            x_axis = range(1, len(explained_variance_ratio) + 1)
            
            plt.bar(x_axis, explained_variance_ratio, alpha=0.6, 
                    color='g', label='Variance expliquée individuelle')

            # Graphique de la variance cumulée
            plt.plot(x_axis, cumulative_variance, 
                    marker='o', linestyle='--', color='r', label='Variance expliquée cumulée')

            plt.title('Analyse de la Variance Expliquée (après Variance Threshold)')
            plt.xlabel('Numéro de la Composante Principale')
            plt.ylabel('Proportion de Variance Expliquée')
            # Limiter les ticks si le nombre de composantes est très grand
            if len(x_axis) <= 50:
                plt.xticks(x_axis, rotation=45)
            
            plt.grid(axis='y', linestyle='--')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            return pipeline
        raise ValueError(f"Unknown unsupervised feature selection method: {method}")

    def get_model(self, 
                  compute: bool = False, 
                  retrain: bool = True,
                  model_type: str = 'classifier', 
                  outlier_type: str = 'mahalanobis_chi2',
                  classifier_type: List[str] = ['hist_gradient_boosting', 'svm', 'random_forest', 'knn', 'mlp'], 
                  scaler: List[str] = ['NoScaler','StandardScaler','RobustScaler','MinMaxScaler','MaxAbsScaler','Normalizer','QuantileTransformer'],
                  optimizing: bool = False,
                  verbose: bool = False,
                  verbose_plot = True,
                  shap_analysis: bool = False,
                  feature_selection_method: str = 'lasso',
                  model_name = None):
        """
        Trains or loads a machine learning model.

        This method prepares the data, trains a classifier or outlier detection
        model, and can perform hyperparameter optimization and SHAP analysis.

        Args:
            compute (bool): If True, forces the training of a new model. If False,
                            attempts to load a cached model. Defaults to False.
            retrain (bool): If True, forces retraining even if the dataset size
                            hasn't changed. Defaults to True.
            model_type (str): The type of model to train. Either 'classifier' or
                              'outlier'. Defaults to 'classifier'.
            outlier_type (str): The method for outlier detection.
                                Options: 'elliptic_envelope' or 'mahalanobis_chi2'.
                                Defaults to 'mahalanobis_chi2'.
            classifier_type (List[str]): A list of classifiers to evaluate.
                                         Defaults to a list of common classifiers.
            scaler (List[str]): A list of data scalers to apply during evaluation.
                                Defaults to a list of common scalers.
            optimizing (bool): If True, performs hyperparameter optimization on
                               the best model. Defaults to False.
            verbose (bool): If True, prints detailed training and evaluation results.
                            Defaults to False.
            shap_analysis (bool): If True, performs SHAP analysis on the best
                                  trained model. Defaults to False.

        Returns:
            Any: The best trained model (e.g., a scikit-learn estimator).

        Raises:
            ValueError: If an unknown model or outlier type is specified.
        """
        
        if compute:
            X_all, _ = self.get_features_selected()
            y_labels = self.get_y_without_coiled_worm()  # strings 'Mutant'/'WT'

            if model_type == 'outlier':
                mask_WT = np.array([lbl == 'WT' for lbl in y_labels])
                X = X_all[mask_WT]  # on garde uniquement les WT pour entraîner l'outlier detector
            else:
                X = X_all.copy()
                label_mapping = {'Mutant': 1, 'WT': 0}
                y = np.array([label_mapping[label] for label in y_labels])
            
                # Balance the dataset to prevent class imbalance issues.
                indices_class_0 = np.where(y == 0)[0]
                indices_class_1 = np.where(y == 1)[0]
                min_size = min(len(indices_class_0), len(indices_class_1))
                balanced_indices_0 = np.random.choice(indices_class_0, min_size, replace=False)
                balanced_indices_1 = np.random.choice(indices_class_1, min_size, replace=False)
                balanced_indices = np.concatenate([balanced_indices_0, balanced_indices_1])
                X_balanced = X[balanced_indices]
                y_balanced = y[balanced_indices]
                X, y = shuffle(X_balanced, y_balanced, random_state=42)
            
            
                # Check if the dataset has new samples to decide whether to retrain.
                csv_path = Path(MODELS_DIR) / "best_model_tracking.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                else:
                    df = pd.DataFrame(columns=['date','best_scaler_name','best_model_name','best_score','len_y'])      
                max_len_y = df['len_y'].max()
                if len(y) <= max_len_y and not retrain:
                    print("Dataset has not changed significantly. No need to retrain the model.")
                    model = joblib.load(MODELS_DIR / "model_prediction.pkl")
                    return model, 0
    

        if not compute:
            try:
                if model_name is not None:
                    model = joblib.load(MODELS_DIR / f"model_prediction_{model_name}.pkl")
                    print("Loaded cached model from", MODELS_DIR / f"model_prediction_{model_name}.pkl")
                else:
                    model = joblib.load(MODELS_DIR / f"model_prediction.pkl")
                    print("Loaded cached model from", MODELS_DIR / f"model_prediction.pkl")
                return model, 0
            except Exception as e:
                print(f"Error loading cached model: {e}")
                return None, 0
        else:
            if model_type == 'classifier':
                print("Training classifier...")
                # Define and filter the scalers to be used.
                scaler_dict_complete = {
                    'NoScaler': FunctionTransformer(func=None),
                    'StandardScaler': StandardScaler(),
                    'RobustScaler': RobustScaler(),
                    'MinMaxScaler': MinMaxScaler(),
                    'MaxAbsScaler': MaxAbsScaler(),
                    'Normalizer': Normalizer(),
                    'QuantileTransformer': QuantileTransformer()
                }
                scaler_dict = {name: scaler_dict_complete[name] for name in scaler if name in scaler_dict_complete}

                # scoring can be : "accuracy", "all_mutants", "all_wts"
                results_df, model, best_scaler_name, best_model_name, best_score = evaluate_models_with_scalers(
                    X, y, classifier_type, scaler_dict, ClassifierFactory,
                    scoring='accuracy', optimize_hyperparams=optimizing, verbose=verbose,
                    shap_analysis=shap_analysis,
                    feature_selection_method=feature_selection_method,
                    random_state=42
                )
                if model_name is not None:
                    joblib.dump(model, MODELS_DIR / f"model_prediction_{model_name}.pkl") 
                else:
                    joblib.dump(model, MODELS_DIR / f"model_prediction.pkl") 
                    
                # --- Sauvegarde des features sélectionnées (si pas "none") ---
                if feature_selection_method != 'none':
                    selector = model.named_steps.get('selector', None)
                    if selector is not None and hasattr(selector, 'get_support'):
                        support = selector.get_support(indices=True)
                        
                        script_path = pathlib.Path(__file__).resolve()
                        root_dir = script_path.parent.parent.parent
                        file_path = root_dir / 'models' / 'selected_features.txt'
                        
                        with open(file_path, 'w') as f:
                            for idx in support:
                                f.write(f"{idx}\n")
                        
                        print(f"[INFO] Saved {len(support)} selected feature indices to {file_path}")

                
                # Visualize results with a heatmap if multiple models or scalers were tested.
                if (len(classifier_type) > 1 or len(scaler) > 1) and verbose_plot == True:
                    plot_heatmap(results_df)

            elif model_type == 'outlier':
                best_model_name = 'Outlier Detection'
                best_score = 0

                # --- non-supervised feature selection settings ---
                fs_method = 'variance_pca'  # 'none', 'variance', 'pca', 'variance_pca'
                pca_k = 20  # si PCA

                X_WT = X  # uniquement WT pour entraînement

                # Build selector
                selector = None
                if fs_method != 'none':
                    if fs_method not in {'variance', 'pca', 'variance_pca'}:
                        raise ValueError("Pour outliers non-supervisés, feature_selection_method doit être 'none', 'variance' ou 'pca'.")
                    selector = self.make_unsupervised_selector(fs_method, X_WT, k=pca_k, var_threshold=1e-5)
                    X_sel = selector.transform(X_WT)
                else:
                    X_sel = X_WT

                # Scaler recommandé
                best_scaler_name = 'RobustScaler'
                chosen_scaler = RobustScaler()

                if outlier_type == 'mahalanobis_chi2':
                    detector = MahalanobisOutlierDetector(contamination=0.001, scaler=chosen_scaler, regularization=1e-8)
                    # stocke le selector dans le detector
                    detector.selector_ = selector
                    # Fit sur features sélectionnées
                    model = detector.fit(X_sel)
                    joblib.dump(model, MODELS_DIR / "model_outlier_prediction.pkl")
                elif outlier_type == 'elliptic_envelope':
                    detector = EllipticEnvelope(contamination=0.001)
                    # scaler + selector
                    X_proc = chosen_scaler.fit_transform(X_sel) if chosen_scaler else X_sel
                    detector = detector.fit(X_proc)
                    detector.scaler_ = chosen_scaler
                    detector.selector_ = selector
                    model = detector
                else:
                    raise ValueError(f"Unknown outlier detection method: {outlier_type}")

            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Save model performance metrics to a CSV file.
            if model_type == 'outlier':
                saved_len = len(X)  # nombre d'exemples WT utilisés pour entraîner l'outlier detector
            else:
                saved_len = len(y)
    
            new_line = {
                'date': [pd.Timestamp.now().strftime(DATE_FORMAT)],
                'best_scaler_name': [best_scaler_name],
                'best_model_name': [best_model_name],
                'best_score': [best_score],
                'len_y': [saved_len]
            }
            df_new_results = pd.DataFrame(new_line)
            csv_path = Path(MODELS_DIR) / "best_model_tracking.csv"
            if os.path.exists(csv_path):
                df_existing_results = pd.read_csv(csv_path)
                df_combined_results = pd.concat([df_existing_results, df_new_results], ignore_index=True)
                df_combined_results.to_csv(csv_path, index=False, mode='w')
            else:
                df_new_results.to_csv(csv_path, index=False, mode='w', header=True)
                
                
                
            df = pd.read_csv(csv_path)        
            max_existing_score = df['best_score'].max()
            # Save the new model only if its performance is better than the best saved model.
            if best_score >= max_existing_score: 
                try:
                    joblib.dump(model, MODELS_DIR / "model_prediction.pkl")
                    print(f"Model saved to", MODELS_DIR / "model_prediction.pkl")
                except Exception as e:
                    print(f"Error saving model: {e}")
            
            return model, best_score
       
    def get_y_without_coiled_worm(self) -> np.ndarray:
        """
        Retrieves the labels for all data items, excluding any coiled worms.
        
        Raises:
            ValueError: If the dataset is empty.

        Returns:
            np.ndarray: A numpy array containing the labels.
        """
        if not self.data:
            raise ValueError("Get_y : Dataset is empty. Call load_images() first.")
            
        y = []
        for item in self.data:
            if not item.coiled:
                y.append(item.label)

        return np.array(y)
    
    @staticmethod
    def _preprocess_image(image: np.ndarray) -> np.ndarray:
        """
        Performs basic preprocessing on a single image.

        This static method handles multi-channel images, resizes them to a
        consistent size, and ensures a uniform data type.
        
        Args:
            image (np.ndarray): The raw input image.
            
        Returns:
            np.ndarray: The preprocessed image.
        """
        # Handle multi-channel images by selecting the second channel.
        if len(image.shape) > 2:
            image = image[1, :, :]
            
        # Resize if the image dimensions do not match the expected size.
        if image.shape != IMAGE_SIZE:
            image = resize(image, IMAGE_SIZE, preserve_range=True)
            
        # Ensure consistent data type for all images.
        return image.astype(np.uint16)
    
    def merge_with(self, other_dataset: 'Dataset_Manager', avoid_duplicates: bool = True) -> None:
        """
        Merges another Dataset_Manager object's data into the current one.

        Args:
            other_dataset (Dataset_Manager): The dataset instance to merge with.
            avoid_duplicates (bool): If True, prevents adding data items that
                                     have the same filename as an existing item.
                                     Defaults to True.

        Raises:
            TypeError: If the input is not a Dataset_Manager instance.
        """
        if not isinstance(other_dataset, Dataset_Manager):
            raise TypeError("merge_with expects a Dataset object as input.")

        # Create a set of existing filenames for efficient duplicate checking.
        existing_filenames = {data_item.filename for data_item in self.data} if avoid_duplicates else set()

        for data_item in other_dataset.data:
            if not avoid_duplicates or data_item.filename not in existing_filenames:
                self.add_data(data_item)
                
    def remove_unclassified(self):
        """
        Removes all data items with the label 'Unclassified' from the dataset.
        """
        original_count = len(self.data)
        # Use a list comprehension to filter out unclassified items.
        self.data = [item for item in self.data if item.label != 'Unclassified']
        removed_count = original_count - len(self.data)
        print(f"Removed {removed_count} unclassified data items.")