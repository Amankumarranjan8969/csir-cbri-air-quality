import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

class ConformalPredictor:
    """
    Split Conformal Prediction for calibrated prediction intervals.
    Computes nonconformity scores (absolute residuals) on an independent validation set
    to guarantee finite-sample coverage on unseen test observations.
    """
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.q_val_ = None

    def fit(self, y_val, y_val_pred):
        """Fit empirical quantile threshold on calibration validation set."""
        residuals = np.abs(np.asarray(y_val) - np.asarray(y_val_pred))
        n = len(residuals)
        # Standard finite-sample correction
        k = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        k = min(max(k, 1), n)
        self.q_val_ = float(np.sort(residuals)[k - 1])
        return self

    def predict_interval(self, y_pred, min_val=0.0):
        """Generate lower and upper prediction intervals for point predictions."""
        if self.q_val_ is None:
            raise RuntimeError("ConformalPredictor must be fitted on validation residuals.")
        preds = np.asarray(y_pred)
        lower = np.maximum(preds - self.q_val_, min_val)
        upper = preds + self.q_val_
        return lower, upper


def recursive_multistep_forecast(model, initial_row, feature_selector, steps=7, target_col='PM2_5'):
    """
    Perform recursive multi-step forecasting for short-to-medium horizons (e.g. 7, 30 days).
    Updates autoregressive lag features iteratively using predicted values.
    
    Args:
        model: Fitted estimator with .predict(X) method.
        initial_row (pd.DataFrame): Latest observation at forecast origin (single row).
        feature_selector: Fitted TrainingFittedFeatureSelector.
        steps (int): Forecast horizon in days.
        target_col (str): Target column name.
        
    Returns:
        forecast_df (pd.DataFrame): Forecasted timeline with step, date, and predicted values.
    """
    selected_cols = feature_selector.selected_features_
    current_state = initial_row.copy()
    predictions = []
    future_dates = pd.date_range(start=current_state['Date'].iloc[0] + pd.Timedelta(days=1), periods=steps, freq='D')

    # Historical lag buffer
    lag_buffer = [current_state[target_col].iloc[0]]
    if f"{target_col}_lag1" in current_state:
        lag_buffer.append(current_state[f"{target_col}_lag1"].iloc[0])
    if f"{target_col}_lag2" in current_state:
        lag_buffer.append(current_state[f"{target_col}_lag2"].iloc[0])
    if f"{target_col}_lag3" in current_state:
        lag_buffer.append(current_state[f"{target_col}_lag3"].iloc[0])

    for step, f_date in enumerate(future_dates, start=1):
        # Update calendar features for step
        doy = f_date.dayofyear
        if 'sin_doy' in current_state.columns:
            current_state['sin_doy'] = np.sin(2 * np.pi * doy / 365.25)
        if 'cos_doy' in current_state.columns:
            current_state['cos_doy'] = np.cos(2 * np.pi * doy / 365.25)
        if 'dayofweek' in current_state.columns:
            current_state['dayofweek'] = f_date.dayofweek
        if 'month' in current_state.columns:
            current_state['month'] = f_date.month

        # Update lag features from buffer
        if f"{target_col}_lag1" in current_state.columns and len(lag_buffer) >= 1:
            current_state[f"{target_col}_lag1"] = lag_buffer[0]
        if f"{target_col}_lag2" in current_state.columns and len(lag_buffer) >= 2:
            current_state[f"{target_col}_lag2"] = lag_buffer[1]
        if f"{target_col}_lag3" in current_state.columns and len(lag_buffer) >= 3:
            current_state[f"{target_col}_lag3"] = lag_buffer[2]

        # Extract features and predict
        X_in = current_state[selected_cols]
        step_pred = float(model.predict(X_in)[0])
        step_pred = max(step_pred, 1.0) # Physical positivity bound
        predictions.append(step_pred)

        # Update buffer with new prediction
        lag_buffer = [step_pred] + lag_buffer[:-1] if len(lag_buffer) > 1 else [step_pred]

    return pd.DataFrame({
        'Step': list(range(1, steps + 1)),
        'Forecast_Date': future_dates,
        'Predicted_PM2_5': np.round(predictions, 2)
    })


def generate_long_term_projection(historical_df, forecast_days=1825, target_col='PM2_5'):
    """
    Generate transparent 5-Year statistical scenario projection based on
    circular day-of-year seasonal harmonics and secular trend decomposition.
    Clearly labeled as scenario-based extrapolation.
    """
    last_date = historical_df['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')

    df = historical_df.copy()
    valid_mask = ~df[target_col].isna()
    df_valid = df[valid_mask].copy()

    df_valid['doy'] = df_valid['Date'].dt.dayofyear
    doy_seasonal = df_valid.groupby('doy')[target_col].agg(['mean', 'std']).reset_index()

    # Wrap around for circular boundary continuity
    doy_extended = pd.concat([doy_seasonal, doy_seasonal, doy_seasonal]).reset_index(drop=True)
    doy_extended['smooth_mean'] = doy_extended['mean'].rolling(15, center=True, min_periods=1).mean()
    doy_extended['smooth_std'] = doy_extended['std'].rolling(15, center=True, min_periods=1).mean()
    smooth_map = doy_extended.groupby('doy')[['smooth_mean', 'smooth_std']].mean()

    future_doy = future_dates.dayofyear
    future_seasonal_mean = np.array([
        smooth_map.loc[d, 'smooth_mean'] if d in smooth_map.index else smooth_map['smooth_mean'].mean()
        for d in future_doy
    ])
    future_seasonal_std = np.array([
        smooth_map.loc[d, 'smooth_std'] if d in smooth_map.index else smooth_map['smooth_std'].mean()
        for d in future_doy
    ])

    # Secular trend fit on valid historical points
    days_hist = (df_valid['Date'] - df['Date'].min()).dt.days.values
    slope, intercept = np.polyfit(days_hist, df_valid[target_col].values, 1)

    days_future = (future_dates - df['Date'].min()).days.values
    trend_proj = intercept + slope * days_future
    historical_mean = df_valid[target_col].mean()

    # Composite projection mean
    projection_mean = np.maximum(future_seasonal_mean + (trend_proj - historical_mean) * 0.5, 5.0)

    # Expanding uncertainty envelope with horizon
    horizon_scale = 1.0 + 0.15 * np.sqrt(np.arange(len(future_dates)) / 365.25)
    ci_upper = projection_mean + 1.96 * future_seasonal_std * horizon_scale
    ci_lower = np.maximum(projection_mean - 1.96 * future_seasonal_std * horizon_scale, 0.0)

    return pd.DataFrame({
        'Date': future_dates,
        'Projected_PM2_5': np.round(projection_mean, 2),
        'Lower_Bound_95': np.round(ci_lower, 2),
        'Upper_Bound_95': np.round(ci_upper, 2),
        'Secular_Trend': np.round(trend_proj, 2),
        'Year': future_dates.year,
        'Month': future_dates.month
    })


def evaluate_multi_horizon_forecasting(models_dict, X_train, y_train, X_test, y_test, horizons=(1, 3, 7), dates_test=None):
    """
    Evaluate multi-horizon direct forecasting across multiple models.
    Horizons: t+1 (1-day ahead), t+3 (3-day ahead), t+7 (7-day ahead).
    Returns results dataframe and prediction dictionary for plotting.
    """
    from sklearn.base import clone

    records = []
    preds_by_horizon = {}

    for h in horizons:
        shift_k = h - 1
        if shift_k == 0:
            X_tr_h = X_train.values if hasattr(X_train, 'values') else np.asarray(X_train)
            y_tr_h = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
            X_te_h = X_test.values if hasattr(X_test, 'values') else np.asarray(X_test)
            y_te_h = y_test.values if hasattr(y_test, 'values') else np.asarray(y_test)
            h_dates = dates_test
        else:
            X_tr_h = X_train.iloc[:-shift_k].values if hasattr(X_train, 'iloc') else np.asarray(X_train)[:-shift_k]
            y_tr_h = y_train[shift_k:] if hasattr(y_train, '__getitem__') else np.asarray(y_train)[shift_k:]
            X_te_h = X_test.iloc[:-shift_k].values if hasattr(X_test, 'iloc') else np.asarray(X_test)[:-shift_k]
            y_te_h = y_test[shift_k:] if hasattr(y_test, '__getitem__') else np.asarray(y_test)[shift_k:]
            h_dates = dates_test.iloc[shift_k:].reset_index(drop=True) if dates_test is not None else None

        preds_by_horizon[h] = {
            'y_true': y_te_h,
            'dates': h_dates,
            'models': {}
        }

        for name, model in models_dict.items():
            try:
                m_h = clone(model)
                m_h.fit(X_tr_h, y_tr_h)
                preds = m_h.predict(X_te_h)
            except Exception:
                preds = model.predict(X_te_h)

            r2 = r2_score(y_te_h, preds)
            rmse = root_mean_squared_error(y_te_h, preds)
            mae = mean_absolute_error(y_te_h, preds)

            preds_by_horizon[h]['models'][name] = preds

            records.append({
                'Model': name,
                'Horizon_Days': h,
                'Horizon_Label': f"t+{h} (Day {h})",
                'R2': round(float(r2), 4),
                'RMSE': round(float(rmse), 4),
                'MAE': round(float(mae), 4),
                'Test_Samples': len(y_te_h)
            })

    return pd.DataFrame(records), preds_by_horizon


def plot_multi_horizon_degradation(multi_horizon_df, save_path=None):
    """
    Plot degradation curves of R2 and RMSE across forecast horizons.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    models = multi_horizon_df['Model'].unique()
    palette = {
        'RandomForest': '#e53e3e',
        'ExtraTrees': '#3182ce',
        'XGBoost': '#38a169',
        'LightGBM': '#805ad5',
        'CatBoost': '#dd6b20'
    }
    markers = {
        'RandomForest': 'o',
        'ExtraTrees': 's',
        'XGBoost': '^',
        'LightGBM': 'D',
        'CatBoost': 'v'
    }

    for m in models:
        sub = multi_horizon_df[multi_horizon_df['Model'] == m].sort_values('Horizon_Days')
        col = palette.get(m, '#4a5568')
        mark = markers.get(m, 'o')
        ax1.plot(sub['Horizon_Days'], sub['R2'], color=col, marker=mark, linewidth=2.0, markersize=7, label=m)
        ax2.plot(sub['Horizon_Days'], sub['RMSE'], color=col, marker=mark, linewidth=2.0, markersize=7, label=m)

    ax1.set_xlabel('Forecast Horizon (Days Ahead)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Out-of-Sample R² Score', fontsize=11, fontweight='bold')
    ax1.set_title('Predictive R² Degradation over Forecast Horizon', fontsize=12, fontweight='bold')
    ax1.set_xticks([1, 3, 7])
    ax1.set_xticklabels(['1-Day Ahead (t+1)', '3-Day Ahead (t+3)', '7-Day Ahead (t+7)'])
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='lower left', frameon=True, fontsize=9.5)

    ax2.set_xlabel('Forecast Horizon (Days Ahead)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Test RMSE (µg/m³)', fontsize=11, fontweight='bold')
    ax2.set_title('Forecast RMSE Accumulation over Time', fontsize=12, fontweight='bold')
    ax2.set_xticks([1, 3, 7])
    ax2.set_xticklabels(['1-Day Ahead (t+1)', '3-Day Ahead (t+3)', '7-Day Ahead (t+7)'])
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='upper left', frameon=True, fontsize=9.5)

    plt.suptitle('Multi-Horizon PM2.5 Forecast Performance on Held-Out Test Set (2023–2024)', fontsize=13.5, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig

