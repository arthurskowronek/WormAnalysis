"""
Outlier detection using Mahalanobis distance.
"""
import numpy as np
from scipy.stats import chi2

class MahalanobisOutlierDetector:
    """
    A class for detecting outliers in a dataset using the Mahalanobis distance.

    This method assumes that the data follows a multivariate Gaussian
    distribution. Outliers are identified as points with a Mahalanobis
    distance greater than a threshold determined by a Chi-squared distribution.

    Args:
        contamination (float): The expected proportion of outliers in the
                               dataset, a value between 0 and 0.5. Defaults to 0.001.
    """
    def __init__(self, contamination=0.001):
        """
        Initializes the MahalanobisOutlierDetector.

        Args:
            contamination (float): The expected proportion of outliers in the
                                   dataset.
        """
        # The expected proportion of outliers in the training data.
        self.contamination = contamination
        # The mean of the training data.
        self.mean_ = None
        # The inverse of the covariance matrix of the training data.
        self.inv_cov_ = None
        # The threshold for the Mahalanobis distance, derived from the Chi-squared distribution.
        self.threshold_ = None
        # The degrees of freedom for the Chi-squared distribution, equal to the number of features.
        self.df_ = None

    def fit(self, X: np.ndarray):
        """
        Fits the model to the training data.

        This method calculates the mean and inverse covariance matrix of the
        training data and determines the Mahalanobis distance threshold for
        outlier detection.

        Args:
            X (np.ndarray): The training data, with shape (n_samples, n_features).

        Returns:
            MahalanobisOutlierDetector: The fitted instance of the detector.
        """
        # Calculate the mean of the training data.
        self.mean_ = np.mean(X, axis=0)
        # Calculate the covariance matrix.
        cov = np.cov(X, rowvar=False)
        # Calculate the pseudo-inverse of the covariance matrix.
        self.inv_cov_ = np.linalg.pinv(cov)
        # The degrees of freedom is the number of features.
        self.df_ = X.shape[1]
        # Calculate the threshold based on the Chi-squared distribution and contamination level.
        self.threshold_ = np.sqrt(chi2.ppf(1 - self.contamination, df=self.df_))
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the Mahalanobis distance for each sample in X.

        Args:
            X (np.ndarray): The data to compute the Mahalanobis distance for,
                            with shape (n_samples, n_features).

        Returns:
            np.ndarray: An array of Mahalanobis distances, one for each sample.
        """
        # Calculate the difference of each sample from the mean.
        diffs = X - self.mean_
        # Efficiently compute the Mahalanobis distance using numpy's einsum.
        dists = np.sqrt(np.einsum('ij,jk,ik->i', diffs, self.inv_cov_, diffs))
        return dists

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts whether each sample in X is an inlier or an outlier.

        Samples with a Mahalanobis distance greater than the threshold are
        classified as outliers.

        Args:
            X (np.ndarray): The data to predict on, with shape (n_samples, n_features).

        Returns:
            np.ndarray: A numpy array of predictions. -1 for outliers and 1 for inliers.
        """
        # Get the Mahalanobis distances for the new data.
        dists = self.decision_function(X)
        # Classify based on the threshold.
        return np.where(dists > self.threshold_, -1, 1)  # -1: outlier, 1: inlier