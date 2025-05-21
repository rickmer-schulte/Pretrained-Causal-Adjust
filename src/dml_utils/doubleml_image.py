import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold

class DoubleMLPLRImage:
    """
    Double Machine Learning for Partially Linear Regression with image data.
    
    This version uses the partialling-out score for the ATE,
    and it assumes that the nuisance learners (ml_l for outcome and ml_m for treatment)
    are CNNs (or any image-based learner) implementing a scikit-learn-like API (fit/predict).
    
    Parameters
    ----------
    X : numpy.ndarray
        Array of images (n_samples, channels, height, width).
    
    y : numpy.ndarray
        Outcome variable (n_samples, ).
    
    d : numpy.ndarray
        Binary treatment variable (n_samples, ). 
    
    ml_l : estimator
        Nuisance function to predict y from X (e.g. a CNN regressor).
    
    ml_m : estimator
        Nuisance function to predict d from X (e.g. a CNN regressor or classifier).
    
    n_folds : int, default=2
        Number of folds for cross-fitting.
    
    n_rep : int, default=1
        Number of repetitions of the sample-splitting.
    """
    
    def __init__(self, X, y, d, ml_l, ml_m, n_folds=2, n_rep=1):
        self.X = X
        self.y = y
        self.d = d
        self.ml_l = ml_l
        self.ml_m = ml_m
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.coefs = None 
        self.se = None    

    def fit(self):
        rep_coef = []
        n = self.X.shape[0]
        
        # Repeat the procedure n_rep times with different splits
        for rep in range(self.n_rep):
            # Create folds (shuffle with a different seed each repetition)
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=rep)
            # Arrays for residuals (u: outcome residual, v: treatment residual)
            u_all = np.empty(n, dtype=float)
            v_all = np.empty(n, dtype=float)
            
            for train_idx, test_idx in kf.split(self.X):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]
                d_train, d_test = self.d[train_idx], self.d[test_idx]
                
                # Fit outcome model (ml_l) on training data and predict on test
                self.ml_l.fit(X_train, y_train)
                y_pred = self.ml_l.predict(X_test)
                
                # Fit treatment model (ml_m) on training data and predict on test
                self.ml_m.fit(X_train, d_train)
                d_pred = self.ml_m.predict(X_test)
                
                # Compute residuals: u = y - predicted y, v = d - predicted d
                u_all[test_idx] = y_test - y_pred
                v_all[test_idx] = d_test - d_pred

            # Compute the ATE estimate for this repetition: 
            orth_lm = sm.OLS(u_all, v_all).fit(cov_type='HC0')
            theta_rep = orth_lm.params
            rep_coef.append(theta_rep)
        
        # Average across repetitions
        self.coef = np.mean(rep_coef)
        if self.n_rep > 1:
            self.se = np.std(rep_coef, ddof=1) / np.sqrt(self.n_rep)
        else:
            self.se = orth_lm.HC0_se[0]
        return self