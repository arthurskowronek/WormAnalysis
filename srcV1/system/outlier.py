"""
Outlier detection using Mahalanobis distance.
"""
from scipy.stats import chi2
import numpy as np

class MahalanobisOutlierDetector:
    def __init__(self, contamination=0.001):
        self.contamination = contamination
        self.mean_ = None
        self.inv_cov_ = None
        self.threshold_ = None
        self.df_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        self.inv_cov_ = np.linalg.pinv(cov)
        self.df_ = X.shape[1]
        self.threshold_ = np.sqrt(chi2.ppf(1 - self.contamination, df=self.df_))
        return self

    def decision_function(self, X):
        diffs = X - self.mean_
        dists = np.sqrt(np.einsum('ij,jk,ik->i', diffs, self.inv_cov_, diffs))
        return dists

    def predict(self, X):
        dists = self.decision_function(X)
        return np.where(dists > self.threshold_, -1, 1)  # -1: outlier, 1: inlier
