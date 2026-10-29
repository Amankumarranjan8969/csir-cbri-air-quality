import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class ValidatedSuperBlend:
    """
    Leakage-free SuperBlend Ensemble.
    Optimizes non-negative convex weights (summing to 1) strictly on the VALIDATION set,
    freezes the weights, and evaluates on untouched test data.
    """
    def __init__(self, model_names=None):
        self.model_names = model_names or []
        self.weights_ = None
        self.val_loss_ = None

    def fit(self, val_predictions_dict, y_val):
        """
        Optimize ensemble weights using SLSQP on Validation data only.
        
        Args:
            val_predictions_dict (dict): {model_name: np.ndarray of shape (N_val,)}
            y_val (np.ndarray): True validation labels.
        """
        self.model_names = list(val_predictions_dict.keys())
        preds_matrix = np.column_stack([val_predictions_dict[m] for m in self.model_names])
        y_v = np.asarray(y_val).ravel()
        n_models = len(self.model_names)

        def objective(w):
            pred = np.dot(preds_matrix, w)
            return np.mean((y_v - pred) ** 2)

        init_w = np.full(n_models, 1.0 / n_models)
        bounds = [(0.0, 1.0) for _ in range(n_models)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        res = minimize(objective, init_w, bounds=bounds, constraints=constraints, method='SLSQP')
        self.weights_ = res.x
        self.val_loss_ = float(res.fun)
        return self

    def predict(self, predictions_dict):
        """Compute weighted blend using frozen validation weights."""
        if self.weights_ is None:
            raise RuntimeError("SuperBlend must be fitted on validation data before predicting.")
        preds_matrix = np.column_stack([predictions_dict[m] for m in self.model_names])
        return np.dot(preds_matrix, self.weights_)

    def get_weights_summary(self):
        """Return dictionary of fitted weights per model."""
        return {m: round(float(w), 4) for m, w in zip(self.model_names, self.weights_)}


class TimeSeriesStackingEnsemble:
    """
    Leakage-free Stacking Meta-Ensemble.
    Generates out-of-fold predictions using expanding-window TimeSeriesSplit
    on the training set so the meta-learner is NEVER trained on in-sample predictions.
    """
    def __init__(self, base_models_dict, n_splits=5):
        self.base_models_dict = base_models_dict
        self.n_splits = n_splits
        self.meta_learner_ = None
        self.model_names_ = list(base_models_dict.keys())

    def fit(self, X_train, y_train):
        """
        Generate out-of-fold predictions via TimeSeriesSplit and fit RidgeCV meta-learner.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        oof_preds = {m: [] for m in self.model_names_}
        oof_indices = []

        X_arr = np.asarray(X_train)
        y_arr = np.asarray(y_train)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_arr)):
            X_f_tr, y_f_tr = X_arr[train_idx], y_arr[train_idx]
            X_f_val = X_arr[val_idx]
            oof_indices.extend(val_idx)

            for name, model_cls in self.base_models_dict.items():
                # Clone/fit a temporary fold model
                fold_model = self._create_fold_model(name)
                fold_model.fit(X_f_tr, y_f_tr)
                oof_preds[name].extend(fold_model.predict(X_f_val))

        oof_X = np.column_stack([oof_preds[m] for m in self.model_names_])
        oof_y = y_arr[oof_indices]

        # Train meta-learner on out-of-fold predictions
        self.meta_learner_ = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 20)))
        ])
        self.meta_learner_.fit(oof_X, oof_y)
        return self

    def _create_fold_model(self, name):
        """Create a lightweight clone of base model for out-of-fold cross-validation."""
        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor
        from catboost import CatBoostRegressor
        from sklearn.ensemble import RandomForestRegressor

        if name == 'CatBoost':
            return CatBoostRegressor(iterations=300, learning_rate=0.05, depth=5, verbose=0)
        elif name == 'XGBoost':
            return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, tree_method='hist')
        elif name == 'LightGBM':
            return LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, verbose=-1)
        else:
            return RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1)

    def predict(self, base_predictions_dict):
        """Predict using meta-learner on base model predictions."""
        if self.meta_learner_ is None:
            raise RuntimeError("Stacking ensemble must be fitted before predict.")
        X_meta = np.column_stack([base_predictions_dict[m] for m in self.model_names_])
        return self.meta_learner_.predict(X_meta)
