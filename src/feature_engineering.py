import numpy as np
import pandas as pd

def build_features(df, target_col='PM2_5', lag_days=(1, 2, 3, 7, 14, 21), rolling_windows=(3, 7, 14, 30), ref_min_date=None):
    """
    Construct research-grade, strictly causal feature matrix for air quality modeling.
    All rolling features, EWMA, and lag features use strictly past observations (shifted by at least 1 step)
    to guarantee zero lookahead target leakage.
    
    Args:
        df (pd.DataFrame): Input dataframe containing 'Date' and raw pollutants/meteorology.
        target_col (str): Prediction target column.
        lag_days (tuple): Autoregressive lag offsets in days.
        rolling_windows (tuple): Rolling aggregation window lengths in days.
        ref_min_date (pd.Timestamp, optional): Reference date for calculating continuous trend.
        
    Returns:
        out (pd.DataFrame): Transformed feature dataframe.
    """
    out = df.copy()

    # Drop any raw calendar/datetime columns that might exist from EDA
    calendar_cols = ['month', 'Month', 'year', 'Year', 'day', 'Day', 'dayofweek', 'quarter', 'trend', 'is_weekend']
    out = out.drop(columns=[c for c in calendar_cols if c in out.columns])

    # 1. Chemical Precursors & Atmospheric Thermodynamics Proxies
    if 'WD' in out.columns and 'WS' in out.columns:
        wd_rad = np.radians(out['WD'])
        out['wind_u'] = -out['WS'] * np.sin(wd_rad)  # Zonal wind component (m/s)
        out['wind_v'] = -out['WS'] * np.cos(wd_rad)  # Meridional wind component (m/s)
        out['sin_wd'] = np.sin(wd_rad)
        out['cos_wd'] = np.cos(wd_rad)

    if 'NO2' in out.columns and 'NO' in out.columns:
        out['no_ratio'] = out['NO2'] / (out['NO'].clip(lower=0.0) + 1e-4)
    if 'NO2' in out.columns and 'Ozone' in out.columns:
        out['photochem_index'] = out['NO2'] * out['Ozone']
    if 'WS' in out.columns and 'AT' in out.columns:
        out['ventilation_proxy'] = out['WS'] * (out['AT'] + 273.15) / 100.0
    if 'AT' in out.columns and 'RH' in out.columns:
        out['temp_rh_index'] = out['AT'] * out['RH'] / 100.0
    if all(c in out.columns for c in ['SO2', 'NO2', 'NH3']):
        out['precursor_sum'] = out['SO2'] + out['NO2'] + out['NH3']

    # 4. Strictly Causal Autoregressive Lags (shift >= 1)
    for l in lag_days:
        if target_col in out.columns:
            out[f"{target_col}_lag{l}"] = out[target_col].shift(l)
        if 'PM10' in out.columns and l in [1, 2, 3, 7]:
            out[f"PM10_lag{l}"] = out['PM10'].shift(l)
        if 'NO2' in out.columns and l in [1, 2, 7]:
            out[f"NO2_lag{l}"] = out['NO2'].shift(l)
        if 'CO' in out.columns and l == 1:
            out["CO_lag1"] = out['CO'].shift(1)
        if 'Ozone' in out.columns and l == 1:
            out["Ozone_lag1"] = out['Ozone'].shift(1)

    # 5. Fine-to-Coarse Aerosol Ratio (Lagged)
    if f"{target_col}_lag1" in out.columns and 'PM10_lag1' in out.columns:
        out['pm_ratio_lag1'] = out[f"{target_col}_lag1"] / (out['PM10_lag1'].clip(lower=1.0) + 1e-4)

    # 6. Autoregressive Differences (Momentum & Acceleration)
    if f"{target_col}_lag1" in out.columns and f"{target_col}_lag2" in out.columns:
        out[f"{target_col}_diff1"] = out[f"{target_col}_lag1"] - out[f"{target_col}_lag2"]
    if f"{target_col}_lag2" in out.columns and f"{target_col}_lag3" in out.columns:
        out[f"{target_col}_diff2"] = out[f"{target_col}_lag2"] - out[f"{target_col}_lag3"]
    if f"{target_col}_diff1" in out.columns and f"{target_col}_diff2" in out.columns:
        out[f"{target_col}_accel"] = out[f"{target_col}_diff1"] - out[f"{target_col}_diff2"]

    # 7. Strictly Shifted Rolling Window Statistics
    # Uses .shift(1) before rolling to guarantee current day observation is NEVER in the window!
    for r in rolling_windows:
        if target_col in out.columns:
            shifted = out[target_col].shift(1)
            out[f"{target_col}_roll_mean_{r}"] = shifted.rolling(r, min_periods=1).mean()
            out[f"{target_col}_roll_std_{r}"] = shifted.rolling(r, min_periods=1).std().fillna(0.0)
            out[f"{target_col}_roll_min_{r}"] = shifted.rolling(r, min_periods=1).min()
            out[f"{target_col}_roll_max_{r}"] = shifted.rolling(r, min_periods=1).max()
        if 'PM10' in out.columns and r in [7, 14]:
            shifted_pm10 = out['PM10'].shift(1)
            out[f"PM10_roll_mean_{r}"] = shifted_pm10.rolling(r, min_periods=1).mean()

    # 8. Exponentially Weighted Moving Averages (EWMA)
    if target_col in out.columns:
        shifted = out[target_col].shift(1)
        out[f"{target_col}_ewma_span3"] = shifted.ewm(span=3).mean()
        out[f"{target_col}_ewma_span7"] = shifted.ewm(span=7).mean()
        out[f"{target_col}_ewma_span14"] = shifted.ewm(span=14).mean()

    return out
