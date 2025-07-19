"""
Main script demonstrating the usage of the codebase.
"""
from src.system.dataset import Dataset

def main():
    """Main function demonstrating the workflow."""
    NAME_DATASET = "big_dataset"  # Name of the dataset to be used
    dataset = Dataset()
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    dataset.load_images(compute=False,
        name_dataset=NAME_DATASET,
        production=False,
        visualize=False,  # Set to True if you want to visualize the preprocessing results
    )
    
    # 2. Extract features
    print("\nExtracting features...")
    dataset.set_features(compute = False,
        production = False,
        name_dataset=NAME_DATASET,
        feature_reduction=False,  # Set to True if you want to apply a feature reduction (PCA)
        selection_method='saved',  # Change to 'saved' or ('kbest', 'boruta', 'mRMR', 'elasticnet' or 'lasso') or 'none'
        verbose=False
    )

    # 3. Compute models
    print("\nComputing models...")
    model = dataset.get_model(compute=True,
        production = False,
        model_type = 'classifier', # 'outlier' or 'classifier'
        outlier_type = 'mahalanobis_chi2', # 'elliptic_envelope' or 'mahalanobis_chi2'
        #classifier_type = ['svm'], # ['hist_gradient_boosting', 'svm', 'random_forest', 'knn', 'mlp']
        #scaler = ['Normalizer'], # ['NoScaler','StandardScaler','RobustScaler','MinMaxScaler','MaxAbsScaler','Normalizer','QuantileTransformer']
        optimizing = False,
        verbose = False,
        shap_analysis = False
    ) 
    
    # 4. Prediction
    print("\nMaking predictions...")
    predictions = model.predict(dataset.get_features_selected()[0])
    print(len(predictions))
    print("Predictions:", predictions)
    
if __name__ == "__main__":
    main()
