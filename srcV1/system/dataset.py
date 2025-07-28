"""
Dataset management.
"""
import os
import joblib
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from skimage.io import imread
from sklearn.utils import shuffle
from skimage.transform import resize
from typing import Dict, List, Tuple, Optional
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, QuantileTransformer, FunctionTransformer

from src.system.data import Data
from src.system.visualization import plot_heatmap, plot_synapse_detection
from src.system.preprocessing import worm_segmentation, is_coiled_worm, get_synapse_using_graph
from src.system.classifiers import ClassifierFactory, evaluate_models_with_scalers
from src.system.outlier import MahalanobisOutlierDetector
from src.system.features import FeatureExtractor

from config import DATA_DIR, IMAGE_SIZE, DEFAULT_PKL_NAME, MODELS_DIR, DATE_FORMAT


class Dataset:
    """Class for managing the dataset loading and preprocessing."""
    
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.dataset_pkl_dir = Path(data_dir) / "Dataset_pkl"
        self.data = []

    def add_data(self, data_item: Data):
        """Ajouter un élément Data au dataset."""
        self.data.append(data_item)
        
    def load_images(self, 
                   compute: bool = True, 
                   test_mode: bool = False,
                   visualize: bool = False,
                   production = True,
                   name_dataset: str = DEFAULT_PKL_NAME) -> 'Dataset':
        """
        Load images from the data directory.
        
        Args:
            compute: If True, reload images from disk instead of using cached version
            test_mode: If True, only load a small subset of images for testing
            name_dataset: Name of the dataset to load
            visualize: If True, visualize the preprocessing results
            
        Returns:
            self: The dataset instance
        """
        pkl_path = self.dataset_pkl_dir / (name_dataset + ".pkl")
        
        if not compute and pkl_path.exists():
            try:
                self.data = joblib.load(pkl_path)
                print("Loaded cached data from", pkl_path)
            except Exception as e:
                print(f"Error loading cached data: {e}")
                print("Proceeding with fresh data loading...")
                compute = True
                name_dataset = DEFAULT_PKL_NAME
                pkl_path = self.dataset_pkl_dir / (name_dataset + ".pkl")
        else:
            if not production: print("No cached data found, proceeding with fresh data loading...") 
            compute = True
            name_dataset = DEFAULT_PKL_NAME
            pkl_path = self.dataset_pkl_dir / (name_dataset + ".pkl")
        
        if compute:
            print("Acquiring data...")
            # Process Mutant and WildType directories
            if production:
                label_dirs = ['Unclassified']
            else:
                label_dirs = ['Mutant', 'WT']
            for label_dir in label_dirs:
                dir_path = self.data_dir / label_dir
                if not dir_path.exists():
                    continue
                    
                for img_path in dir_path.glob('*.tif'):
                    try:
                        # Basic image loading and preprocessing
                        img = imread(img_path)
                        img = self._preprocess_image(img)

                        # Get worm mask
                        worm_mask = worm_segmentation(img)
                        
                        # Skip coiled worms if requested
                        if is_coiled_worm(worm_mask):
                            print(f"Skipping coiled worm in {img_path}")
                            empty_graph = nx.Graph()
                            maxima = []
                            graph = empty_graph
                            median_width = 0
                            diff_slice = 0
                            diff_segment = 0 
                            coiled = True
                        else:
                            # Get synapse data
                            coiled = False
                            maxima, graph, median_width, diff_slice, diff_segment = get_synapse_using_graph(img, worm_mask)

                        # Add data to the new_data object
                        new_data = Data()
                        new_data.label = label_dir
                        new_data.filename = img_path.name
                        new_data.original_image = img
                        if new_data.label == "Mutant":
                            new_data.mutant_type = img_path.name[5:9]
                        new_data.worm_mask = worm_mask
                        new_data.maxima = maxima
                        new_data.graph = graph
                        new_data.median_width = median_width
                        new_data.diff_slice = diff_slice
                        new_data.diff_segment = diff_segment
                        new_data.coiled = coiled
                        self.add_data(new_data)
                        
                        
                        if test_mode and len(self.data) >= 2:
                            break
                            
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        
        # Save processed dataset
        if not test_mode and not production and compute:
            try:
                joblib.dump(self.data, pkl_path)
            except Exception as e:
                print(f"Error saving dataset: {e}")
            
        # Visualize preprocessing results if requested
        if visualize:
            print("\nVisualizing preprocessing results...")
            count = 0
            for data in self.data:
                img_name = data.filename
                img = data.original_image
                mask = data.worm_mask
                max = data.maxima
                plot_synapse_detection(
                    original_image=img,
                    worm_mask=mask,
                    maxima=max,
                    title=f'{img_name}'
                )
                count += 1
                if count >= 3:
                    break    
            
        print("Data acquired successfully.")
        return self
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the processed images and labels.
        
        Returns:
            Tuple of (images, labels)
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
        Update the label of a data item by its filename.
        
        Args:
            filename: The filename of the image to update.
            new_label: The new label to assign.

        Returns:
            True if the label was updated successfully, False otherwise.
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
 
    def get_coiled_worms(self) -> List[Data]:
        """
        Get indices of coiled worms in the dataset.
        Returns:
            List of indices of coiled worms
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
        Get the median width of the worms in the dataset.
        Returns:
            List of median widths
        """
        if not self.data:
            raise ValueError("Get_median : Dataset is empty. Call load_images() first.")
            
        median_widths = []
        for item in self.data:
            median_widths.append(item.median_width)
        
        return median_widths
    
    def get_features(self) -> Optional[np.ndarray]:
        """
        Get the computed features as a DataFrame.
        Returns:
            Tuple of (features, feature_names)
        """
        features = []

        for item in self.data:
            features.append(item.get_features()[0])
            
        features = np.array(features)
        feature_names = self.data[0].get_features()[1]        
                
        return features, feature_names
    
    def get_features_selected(self) -> Optional[np.ndarray]:
        """
        Get the selected features as a DataFrame.
        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        
        for item in self.data:
            if item.coiled == False:
                features.append(item.get_features_selected()[0])
   
        features = np.array(features)
        feature_names = self.data[0].get_features_selected()[1]
        
        return features, feature_names
    
    def set_features(self, compute = True, production = True, name_dataset = DEFAULT_PKL_NAME,  feature_reduction = False, selection_method = "saved", verbose = False) -> None:
        """
        Set the computed features.
        
        Args:
            compute: If True, re-import features from the dataset
            dataset_name: Name of the dataset to save features to
            feature_reduction: If True, apply feature reduction
            selection_method: Method for feature selection (e.g., 'kbest', 'boruta', 'mRMR', 'elasticnet', 'lasso')
        """
        pkl_path = self.dataset_pkl_dir / (name_dataset + "_features.pkl")
        pkl_big_dataset_path = self.dataset_pkl_dir / (name_dataset + ".pkl")
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
            if not production : print("No cached data found, proceeding with fresh data loading...")
            compute = True
            name_dataset = DEFAULT_PKL_NAME
            pkl_path = self.dataset_pkl_dir / (name_dataset + "_features.pkl")
        
                    
        if compute:
            print("Computing features...")
            feature_extractor = FeatureExtractor()
            feature_extractor.set_features(self)
            features, feature_names = feature_extractor.get_features()
            for item, feature_vector in zip(self.data, features):
                if item.coiled == False:
                    item.set_features(feature_vector, feature_names)
            # save the updated dataset
            try:
                if not production: joblib.dump(self.data, self.dataset_pkl_dir / (name_dataset + "_features.pkl"))
                
                # A SUPPRIMER ?
                """df_new = pd.DataFrame(features)  
                df_new.columns = feature_names
                df_new = df_new[~(df_new == 0).all(axis=1)] 
                if not df_new.empty:
                    excel_path = Path(DATA_DIR) / "Excel" / "features.xlsx"
                    if os.path.exists(excel_path):
                        df_existing = pd.read_excel(excel_path)
                        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                        df_combined.to_excel(excel_path, index=False)
                    else:
                        df_new.to_excel(excel_path, index=False)"""

            except Exception as e:
                print(f"Error saving dataset with features: {e}")
        
        if feature_reduction:
            print("Applying feature reduction...")
            _, y = self.get_data()
            indices_coiled = self.get_coiled_worms()
            feature_extractor.feature_reduction(y, indices_coiled)
        elif selection_method == 'none':
            _, y = self.get_data()
            indices_coiled = self.get_coiled_worms()
            features, feature_names, y = feature_extractor._process_features(y, indices_coiled)          
        
            scaler = StandardScaler() 
            features = scaler.fit_transform(features)
            features = np.nan_to_num(features)
            
            count = 0
            for item in self.data:
                if item.coiled == False:
                    item.set_features_selected(features[count], feature_names)
                    count += 1
            return self
        elif selection_method == 'saved':
            if not production: print("Loading saved features...")
            _, y = self.get_data()
            indices_coiled = self.get_coiled_worms()
            features, feature_names, y = feature_extractor._process_features(y, indices_coiled)          
        
            scaler = StandardScaler() 
            features = scaler.fit_transform(features)
            features = np.nan_to_num(features)
            
            with open('models/selected_features.txt', 'r') as f:
                selected_indices = [int(line.strip()) for line in f]
                
            features = features[:, selected_indices]
            feature_names = [feature_names[i] for i in selected_indices]
            count = 0
            for item in self.data:
                if item.coiled == False:
                    item.set_features_selected(features[count], feature_names)
                    count += 1
        elif selection_method in ['kbest', 'boruta', 'mRMR', 'elasticnet', 'lasso']:
            print(f"Applying feature selection using {selection_method}...")
            _, y = self.get_data()
            indices_coiled = self.get_coiled_worms()
            features_selected, feature_names_selected = feature_extractor.feature_selection(
                method=selection_method,
                y = y,
                coiled_worms = indices_coiled,
                verbose_features_selected=verbose
            )
            
            count = 0
            for item in self.data:
                if item.coiled == False:
                    item.set_features_selected(features_selected[count], feature_names_selected)
                    count += 1
            try:
                joblib.dump(self.data, self.dataset_pkl_dir / (name_dataset + "_features_selected.pkl"))
                # A SUPPRIMER ?
                """df = pd.DataFrame(features_selected)
                df.columns = feature_names_selected
                df.to_excel('data/Excel/features_selected.xlsx', index=False)"""
                print("Features selected saved to", self.dataset_pkl_dir / (name_dataset + "_features_selected.pkl"))
            except Exception as e:
                print(f"Error saving dataset with features selected: {e}")
        else:
            raise ValueError(f"Unknown feature selection method: {selection_method}")
      
    def get_model(self, 
                  compute: bool = False, 
                  production = True,
                  model_type: str = 'classifier', 
                  outlier_type: str = 'mahalanobis_chi2',
                  classifier_type: List[str] = ['hist_gradient_boosting', 'svm', 'random_forest', 'knn', 'mlp'], 
                  scaler: List[str] = ['NoScaler','StandardScaler','RobustScaler','MinMaxScaler','MaxAbsScaler','Normalizer','QuantileTransformer'],
                  optimizing: bool = False,
                  verbose: bool = False,
                  shap_analysis: bool = False):
        """
        Get the trained model.
        
        Args:
            compute: If True, reload model from disk
            model_type: Type of model ('classifier' or 'outlier')
            outlier_type: Type of outlier detection ('elliptic_envelope' or 'mahalanobis_chi2')
            classifier_type: List of classifier types to use
            scaler: List of scalers to apply
            optimizing: If True, optimize hyperparameters
            verbose: If True, print the results
            shap_analysis: If True, perform SHAP analysis
        Returns:
            model: Trained model (pkl file)
        """
                        
        X, _ = self.get_features_selected()
        y_labels = self.get_y_without_coiled_worm()
        label_mapping = {'Mutant': 1, 'WT': 0}
        y = np.array([label_mapping[label] for label in y_labels])

        # Get a balanced dataset
        indices_class_0 = np.where(y == 0)[0]
        indices_class_1 = np.where(y == 1)[0]
        min_size = min(len(indices_class_0), len(indices_class_1))
        balanced_indices_0 = np.random.choice(indices_class_0, min_size, replace=False)
        balanced_indices_1 = np.random.choice(indices_class_1, min_size, replace=False)
        balanced_indices = np.concatenate([balanced_indices_0, balanced_indices_1])
        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]
        X, y = shuffle(X_balanced, y_balanced, random_state=42)
        
        
        # Check if the dataset has new samples
        csv_path = Path(MODELS_DIR) / "best_model_tracking.csv"
        df = pd.read_csv(csv_path)        
        max_len_y = df['len_y'].max()
        if len(y) <= max_len_y and production == True:
            print("Dataset has not changed significantly. No need to retrain the model.")
            model = joblib.load(MODELS_DIR / "model_prediction.pkl")
            return model
    

        if not compute:
            try:
                model = joblib.load(MODELS_DIR / "model_prediction.pkl")
                print("Loaded cached model from", MODELS_DIR / "model_prediction.pkl")
                return model
            except Exception as e:
                print(f"Error loading cached model: {e}")
                return None
        else:
            if model_type == 'classifier':
                print("Training classifier...")
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

                results_df, model, best_scaler_name, best_model_name, best_score = evaluate_models_with_scalers(X, y, classifier_type, scaler_dict, ClassifierFactory, optimize_hyperparams = optimizing, verbose = verbose, shap_analysis = shap_analysis)  
                
                if len(classifier_type) > 1 or len(scaler) > 1:
                    plot_heatmap(results_df)
            elif model_type == 'outlier':
                best_model_name = 'Outlier Detection'
                best_scaler_name = 'NoScaler'
                best_score = 0
                if outlier_type == 'elliptic_envelope':
                    X_WT = X[y == 'WT']
                    detector = EllipticEnvelope(contamination=0.001) # 0.1% threshold for outliers
                    model = detector.fit(X_WT)

                elif outlier_type == 'mahalanobis_chi2':
                    X_WT = X[y == 'WT']
                    detector = MahalanobisOutlierDetector(contamination=0.001)
                    model = detector.fit(X_WT)
                else:
                    raise ValueError(f"Unknown outlier detection method: {outlier_type}")
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # save model performance to CSV
            new_line = {
                'date': [pd.Timestamp.now().strftime(DATE_FORMAT)],
                'best_scaler_name': [best_scaler_name],
                'best_model_name': [best_model_name],
                'best_score': [best_score],
                'len_y': [len(y)]
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
            if best_score >= max_existing_score: 
                try:
                    joblib.dump(model, MODELS_DIR / "model_prediction.pkl")
                    print(f"Model saved to", MODELS_DIR / "model_prediction.pkl")
                except Exception as e:
                    print(f"Error saving model: {e}")
            
            return model
    
    
    def get_y_without_coiled_worm(self) -> np.ndarray:
        """
        Get the labels for the dataset without coiled worms.
        This method extracts the labels from the dataset, excluding any coiled worms.
        
        Returns:
            Array of labels
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
        Preprocess a single image.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Handle multi-channel images
        if len(image.shape) > 2:
            image = image[1, :, :]  # Take second channel
            
        # Resize if necessary
        if image.shape != IMAGE_SIZE:
            image = resize(image, IMAGE_SIZE, preserve_range=True)
            
        # Ensure consistent data type
        return image.astype(np.uint16)
    
    def merge_with(self, other_dataset: 'Dataset', avoid_duplicates: bool = True) -> None:
        """
        Merge another Dataset object into the current one.

        Args:
            other_dataset (Dataset): The dataset to merge with.
            avoid_duplicates (bool): If True, avoid adding Data objects with the same filename.
        """
        if not isinstance(other_dataset, Dataset):
            raise TypeError("merge_with expects a Dataset object as input.")

        existing_filenames = {data_item.filename for data_item in self.data} if avoid_duplicates else set()

        for data_item in other_dataset.data:
            if not avoid_duplicates or data_item.filename not in existing_filenames:
                self.add_data(data_item)
                
    def remove_unclassified(self):
        """
        Remove all Data items with label 'Unclassified' from the dataset.
        """
        original_count = len(self.data)
        self.data = [item for item in self.data if item.label != 'Unclassified']
        removed_count = original_count - len(self.data)
        print(f"Removed {removed_count} unclassified data items.")

