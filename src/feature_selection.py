import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr

class TrainingFittedFeatureSelector:
    """
    Research-grade feature selector fitted strictly on historical training observations.
    Combines Mutual Information Regression (non-linear dependencies) with Absolute Pearson
    Correlation (linear / inverse atmospheric dispersion relationships) inspired by state-of-the-art
    environmental modeling standards.
    
    Prevents temporal data leakage by fitting all MI scores and correlation coefficients
    strictly on training-split observations.
    """
    def __init__(self, target_col='PM2_5', min_mi_threshold=0.05, min_abs_corr_threshold=0.05, top_k=None, min_corr_threshold=None):
        self.target_col = target_col
        self.min_mi_threshold = min_mi_threshold
        self.min_abs_corr_threshold = min_corr_threshold if min_corr_threshold is not None else min_abs_corr_threshold
        self.top_k = top_k
        
        self.selected_features_ = []
        self.dropped_features_ = []
        self.mi_scores_ = {}
        self.correlations_ = {}
        self.abs_correlations_ = {}
        self.selection_summary_df_ = None
        self.is_fitted_ = False

    def fit(self, train_df, candidate_cols=None):
        """
        Fit Mutual Information and Pearson correlation metrics strictly on training data.
        """
        df = train_df.copy()
        
        if candidate_cols is None:
            num_cols = df.select_dtypes(include=[np.number]).columns
            datetime_drop = ['Date', 'date', 'Timestamp', 'timestamp', 'datetime', 'time',
                             'Month', 'month', 'Year', 'year', 'Day', 'day', 'dayofweek',
                             'quarter', 'trend', 'is_weekend', 'sin_dow', 'cos_dow', 'sin_doy', 'cos_doy']
            candidate_cols = [c for c in num_cols if c not in datetime_drop and c != self.target_col]
        else:
            datetime_drop = ['Date', 'date', 'Timestamp', 'timestamp', 'datetime', 'time',
                             'Month', 'month', 'Year', 'year', 'Day', 'day', 'dayofweek',
                             'quarter', 'trend', 'is_weekend', 'sin_dow', 'cos_dow', 'sin_doy', 'cos_doy']
            candidate_cols = [c for c in candidate_cols if c not in datetime_drop and c != self.target_col]

        X = df[candidate_cols].copy()
        y = np.asarray(df[self.target_col].values).ravel()

        # Handle any NaNs in target y
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        # Handle any residual NaNs in training predictors
        nan_counts = X.isna().sum()
        if nan_counts.sum() > 0:
            X = X.fillna(X.median()).fillna(0.0)

        # 1. Linear and Inverse Atmospheric Pearson Correlations
        self.correlations_ = {}
        self.abs_correlations_ = {}
        
        for col in candidate_cols:
            col_vals = X[col].values
            std = float(np.std(col_vals))
            if std > 1e-6:
                r, _ = pearsonr(col_vals, y)
                r_val = float(r) if not np.isnan(r) else 0.0
            else:
                r_val = 0.0
            self.correlations_[col] = r_val
            self.abs_correlations_[col] = abs(r_val)

        # 2. Non-linear Mutual Information Scores (with small sample fallback)
        if len(X) <= 5:
            mi_values = [self.abs_correlations_.get(c, 0.0) for c in candidate_cols]
        else:
            n_neighbors = min(3, max(1, len(X) - 2))
            mi_values = mutual_info_regression(X, y, n_neighbors=n_neighbors, random_state=42)
        self.mi_scores_ = dict(zip(candidate_cols, [float(v) for v in mi_values]))

        # 3. Dual-Criteria Selection Logic
        self.selected_features_ = []
        self.dropped_features_ = []
        summary_records = []

        for feat in candidate_cols:
            mi_score = self.mi_scores_.get(feat, 0.0)
            r_val = self.correlations_.get(feat, 0.0)
            abs_r = self.abs_correlations_.get(feat, 0.0)
            col_vals = X[feat].values
            std = float(np.std(col_vals))

            # Rule A: Zero variance / constant feature (e.g. constant rainfall = 0)
            if std <= 1e-6:
                status = "DROPPED"
                reason = "Constant / Zero Variance (No informative signal)"
                self.dropped_features_.append(feat)

            # Rule B: Sufficient Mutual Information OR Strong Absolute Correlation
            elif mi_score >= self.min_mi_threshold or abs_r >= self.min_abs_corr_threshold:
                status = "SELECTED"
                if r_val < -0.05:
                    reason = f"Inverse Atmospheric Predictor (r={r_val:.3f}, MI={mi_score:.3f})"
                elif mi_score >= 0.20:
                    reason = f"Strong Non-linear & Autoregressive Signal (MI={mi_score:.3f}, r={r_val:.3f})"
                else:
                    reason = f"Informative Contributing Predictor (MI={mi_score:.3f}, |r|={abs_r:.3f})"
                self.selected_features_.append(feat)

            # Rule C: Pure uninformative noise (both MI and |r| below threshold)
            else:
                status = "DROPPED"
                reason = f"Uninformative Noise (MI={mi_score:.3f} < {self.min_mi_threshold}, |r|={abs_r:.3f} < {self.min_abs_corr_threshold})"
                self.dropped_features_.append(feat)

            summary_records.append({
                'Feature': feat,
                'MI_Score': round(mi_score, 4),
                'Pearson_r': round(r_val, 4),
                'Abs_Pearson_r': round(abs_r, 4),
                'Status': status,
                'Selection_Reason': reason
            })

        # Rank selected features by Mutual Information descending
        self.selected_features_ = sorted(self.selected_features_, key=lambda f: self.mi_scores_.get(f, 0.0), reverse=True)
        if self.top_k is not None and self.top_k < len(self.selected_features_):
            self.selected_features_ = self.selected_features_[:self.top_k]

        self.selection_summary_df_ = pd.DataFrame(summary_records).sort_values(by='MI_Score', ascending=False).reset_index(drop=True)
        self.is_fitted_ = True
        return self

    def transform(self, df):
        """
        Subset dataframe to selected features and target.
        """
        if not self.is_fitted_:
            raise RuntimeError("FeatureSelector must be fitted on training data before transform.")
        cols = [c for c in self.selected_features_ if c in df.columns]
        if self.target_col in df.columns:
            cols = cols + [self.target_col]
        if 'Date' in df.columns:
            cols = ['Date'] + [c for c in cols if c != 'Date']
        return df[cols].copy()
