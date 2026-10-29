import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

class TrainingFittedPreprocessor:
    """
    Leakage-free preprocessor that learns imputation medians, outlier thresholds,
    and feature scalers strictly from the training partition and transforms
    validation, test, and production inference data without lookahead bias.
    """
    def __init__(self, target_col='PM2_5', outlier_threshold=1.5, scaler_type='robust'):
        self.target_col = target_col
        self.outlier_threshold = outlier_threshold
        self.scaler_type = scaler_type
        
        self.train_medians_ = {}
        self.clipping_bounds_ = {}
        self.feature_scaler_ = None
        self.feature_names_ = []
        self.is_fitted_ = False

    def fit(self, train_df, feature_cols=None):
        """Fit imputation medians, outlier thresholds, and scaler on training data only."""
        df = train_df.copy()
        
        if feature_cols is None:
            num_cols = df.select_dtypes(include=[np.number]).columns
            datetime_drop = ['Date', 'date', 'Timestamp', 'timestamp', 'datetime', 'time',
                             'Month', 'month', 'Year', 'year', 'Day', 'day', 'dayofweek',
                             'quarter', 'trend', 'is_weekend', 'sin_dow', 'cos_dow', 'sin_doy', 'cos_doy']
            self.feature_names_ = [c for c in num_cols if c not in datetime_drop and c != self.target_col]
        else:
            datetime_drop = ['Date', 'date', 'Timestamp', 'timestamp', 'datetime', 'time',
                             'Month', 'month', 'Year', 'year', 'Day', 'day', 'dayofweek',
                             'quarter', 'trend', 'is_weekend', 'sin_dow', 'cos_dow', 'sin_doy', 'cos_doy']
            self.feature_names_ = [c for c in feature_cols if c not in datetime_drop and c != self.target_col]

        # 1. Fit training medians for imputation
        for c in self.feature_names_:
            self.train_medians_[c] = float(df[c].median()) if df[c].notnull().any() else 0.0

        if self.target_col in df.columns:
            self.train_medians_[self.target_col] = float(df[self.target_col].median())

        # 2. Fit IQR outlier clipping bounds on training features only
        for c in self.feature_names_:
            s = df[c].dropna()
            if len(s) > 0:
                q1 = float(s.quantile(0.25))
                q3 = float(s.quantile(0.75))
                iqr = q3 - q1
                lower = q1 - self.outlier_threshold * iqr
                upper = q3 + self.outlier_threshold * iqr
                self.clipping_bounds_[c] = (lower, upper)
            else:
                self.clipping_bounds_[c] = (-np.inf, np.inf)

        # 3. Fit scaler on training feature matrix
        imputed_features = self._impute_and_clip(df[self.feature_names_])
        if self.scaler_type == 'robust':
            self.feature_scaler_ = RobustScaler()
        else:
            self.feature_scaler_ = StandardScaler()
            
        self.feature_scaler_.fit(imputed_features)
        self.is_fitted_ = True
        return self

    def _impute_and_clip(self, feature_df):
        """Helper to impute missing values using fitted train medians and clip bounds."""
        out = feature_df.copy()
        
        # Temporal forward fill within the series
        out = out.ffill()
        
        # Fill remaining initial NaNs with training medians
        for c in out.columns:
            if c in self.train_medians_:
                out[c] = out[c].fillna(self.train_medians_[c])
            else:
                out[c] = out[c].fillna(0.0)
                
            # Apply fitted clipping bounds
            if c in self.clipping_bounds_:
                low, high = self.clipping_bounds_[c]
                out[c] = np.clip(out[c], low, high)
                
        return out

    def transform_df(self, df, feature_cols=None):
        """Impute and clip dataframe using fitted parameters."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fitted on training data before transform.")
            
        cols = feature_cols if feature_cols is not None else self.feature_names_
        out = df.copy()
        
        # Impute target if present with forward fill + train median
        if self.target_col in out.columns:
            out[self.target_col] = out[self.target_col].ffill()
            if self.target_col in self.train_medians_:
                out[self.target_col] = out[self.target_col].fillna(self.train_medians_[self.target_col])
                
        # Transform features
        out[cols] = self._impute_and_clip(out[cols])
        return out

    def transform_scaled(self, feature_df):
        """Return scaled numpy array using fitted scaler."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fitted on training data before transform.")
            
        cleaned_df = self._impute_and_clip(feature_df[self.feature_names_])
        return self.feature_scaler_.transform(cleaned_df)


def chronological_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Strict chronological partition into Train, Validation, and Test sets.
    Preserves strict temporal ordering to prevent lookahead bias.
    """
    total = len(df)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    return train_df, val_df, test_df
