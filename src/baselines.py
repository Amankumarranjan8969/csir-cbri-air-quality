import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

class PersistenceBaseline:
    """Persistence baseline: Predicted next-day target = latest observed target (t-1)."""
    def __init__(self, target_col='PM2_5'):
        self.target_col = target_col
        self.last_value_ = None

    def fit(self, train_df):
        self.last_value_ = float(train_df[self.target_col].dropna().iloc[-1])
        return self

    def predict(self, df):
        # Look for lag1 feature or shift target
        lag1_col = f"{self.target_col}_lag1"
        if lag1_col in df.columns:
            return df[lag1_col].fillna(self.last_value_).values
        elif self.target_col in df.columns:
            return df[self.target_col].shift(1).fillna(self.last_value_).values
        else:
            return np.full(len(df), self.last_value_)


class SeasonalNaiveBaseline:
    """Weekly Seasonal-Naive: Predicted target at day t = observed target at day t-7."""
    def __init__(self, target_col='PM2_5', season_lag=7):
        self.target_col = target_col
        self.season_lag = season_lag
        self.fallback_ = None

    def fit(self, train_df):
        self.fallback_ = float(train_df[self.target_col].dropna().iloc[-self.season_lag:].mean())
        return self

    def predict(self, df):
        lag_col = f"{self.target_col}_lag{self.season_lag}"
        if lag_col in df.columns:
            return df[lag_col].fillna(self.fallback_).values
        elif self.target_col in df.columns:
            return df[self.target_col].shift(self.season_lag).fillna(self.fallback_).values
        else:
            return np.full(len(df), self.fallback_)


class RollingMeanBaseline:
    """Rolling Mean Baseline: Predicted target at day t = mean of past 7 days."""
    def __init__(self, target_col='PM2_5', window=7):
        self.target_col = target_col
        self.window = window
        self.fallback_ = None

    def fit(self, train_df):
        self.fallback_ = float(train_df[self.target_col].dropna().iloc[-self.window:].mean())
        return self

    def predict(self, df):
        roll_col = f"{self.target_col}_roll_mean_{self.window}"
        if roll_col in df.columns:
            return df[roll_col].fillna(self.fallback_).values
        elif self.target_col in df.columns:
            return df[self.target_col].shift(1).rolling(self.window, min_periods=1).mean().fillna(self.fallback_).values
        else:
            return np.full(len(df), self.fallback_)


class LinearRegressionBaseline:
    """Ridge / OLS Linear Baseline."""
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)
