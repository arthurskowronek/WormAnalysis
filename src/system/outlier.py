"""
Outlier detection using Mahalanobis distance.
"""
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import RobustScaler, StandardScaler
from scipy.stats import chi2
import numpy as np

class MahalanobisOutlierDetector:
    """
    Détecteur d'outliers basé sur la distance de Mahalanobis.
    - contamination : proportion attendue d'outliers (ex: 0.001)
    - scaler : un transformateur sklearn (p.ex. RobustScaler()) ou None.
    """
    def __init__(self, contamination=0.001, scaler = None, regularization: float = 1e-8):
        self.contamination = contamination
        self.scaler = scaler  # instance d'un scaler sklearn (fit/transform)
        self.mean_ = None
        self.inv_cov_ = None
        self.threshold_ = None
        self.df_ = None
        self.regularization = regularization
        self.selector_ = None
        self.scaler_ = None

    def fit(self, X: np.ndarray):
        """
        Fit du détecteur sur X (pré-supposé être les WT).
        Stocke le scaler si fourni, calcule covariance (MinCovDet si possible),
        calcule l'inverse régularisé et le seuil basé sur chi2.
        """
        # Appliquer / fitter le scaler si demandé
        if self.scaler is not None:
            self.scaler_ = self.scaler.fit(X)
            X_proc = self.scaler_.transform(X)
        else:
            X_proc = X.copy()

        # Estimation robuste de la covariance si possible
        try:
            mcd = MinCovDet().fit(X_proc)
            cov = mcd.covariance_
            self.mean_ = mcd.location_
        except Exception:
            # fallback : estimateur classique
            self.mean_ = np.mean(X_proc, axis=0)
            cov = np.cov(X_proc, rowvar=False)

        # régularisation numérique et pseudo-inverse
        cov += self.regularization * np.eye(cov.shape[0])
        self.inv_cov_ = np.linalg.pinv(cov)

        self.df_ = X_proc.shape[1]
        print(f"Degrees of freedom for chi2 threshold: {self.df_}")

        # Seuil : on calcule le quantile du chi2 pour la distance^2,
        #     puis on prend la racine si on travaille avec distances (sqrt).
        chi2_q = chi2.ppf(1 - self.contamination, df=self.df_)
        self.threshold_ = np.sqrt(chi2_q)
        print(f"Threshold : {self.threshold_}")

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        # 1) apply selector if exists (selector may be PCA or VarianceThreshold)
        if getattr(self, "selector_", None) is not None:
            X_proc = self.selector_.transform(X)
        elif getattr(self, "selected_indices_", None) is not None:
            X_proc = X[:, self.selected_indices_]
        else:
            X_proc = X

        # 2) apply scaler_ if fitted (fit done inside fit())
        if getattr(self, "scaler_", None) is not None:
            X_proc = self.scaler_.transform(X_proc)

        # 3) compute Mahalanobis distance
        diffs = X_proc - self.mean_
        dists = np.sqrt(np.einsum('ij,jk,ik->i', diffs, self.inv_cov_, diffs))
        return dists

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = self.decision_function(X)
        is_outlier = dists > self.threshold_
        return is_outlier.astype(int)  # 1 = outlier (Mutant), 0 = inlier (WT)

