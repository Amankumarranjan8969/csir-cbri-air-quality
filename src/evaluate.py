import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_smape(y_true, y_pred, eps=1e-6):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) bounded between 0% and 200%.
    Handles zero and near-zero values safely.
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    diff = np.abs(y_pred - y_true) / denominator
    return float(np.mean(diff) * 100.0)

def calculate_mase(y_true, y_pred, y_train_naive):
    """
    Mean Absolute Scaled Error (MASE) relative to in-sample naive persistence.
    MASE < 1 means the model is better than one-step naive persistence.
    """
    mae_model = mean_absolute_error(y_true, y_pred)
    mae_naive = np.mean(np.abs(np.diff(y_train_naive)))
    if mae_naive == 0:
        return np.nan
    return float(mae_model / mae_naive)

def calculate_directional_accuracy(y_true, y_pred):
    """
    Mean Directional Accuracy (MDA %) measuring whether the predicted change
    in air quality (rise or drop) matches the observed change.
    """
    if len(y_true) < 2:
        return 100.0
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    matches = np.sign(true_diff) == np.sign(pred_diff)
    return float(np.mean(matches) * 100.0)

def evaluate_predictions(y_true, y_pred, model_name="Model", split_name="Test", y_train_history=None, extreme_threshold=60.0):
    """
    Comprehensive, scientifically defensible evaluation suite for air quality forecasting.
    Includes 8 distinct metrics: R2, RMSE, MAE, MBE, sMAPE, MASE, Extreme_MAE, Directional_Accuracy.
    
    Args:
        y_true (np.ndarray): Ground truth observed concentrations.
        y_pred (np.ndarray): Predicted concentrations.
        model_name (str): Identifier for model architecture.
        split_name (str): Split evaluated ('Validation', 'Test', etc.).
        y_train_history (np.ndarray, optional): Historical training series for MASE.
        extreme_threshold (float): Standard threshold for NAAQS 24-hr exceedance (default: 60 ug/m3).
        
    Returns:
        metrics_dict (dict): Complete scalar evaluation metrics.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()

    # Base regression metrics
    r2 = float(r2_score(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae = float(mean_absolute_error(y_t, y_p))
    mbe = float(np.mean(y_p - y_t))  # Mean Bias Error: positive = overprediction, negative = underprediction
    smape = calculate_smape(y_t, y_p)
    mase = calculate_mase(y_t, y_p, y_train_history) if y_train_history is not None else np.nan
    dir_acc = calculate_directional_accuracy(y_t, y_p)

    # High-Pollution Episode Evaluation (NAAQS exceedances >= 60 ug/m3)
    extreme_mask = y_t >= extreme_threshold
    if np.any(extreme_mask):
        extreme_mae = float(mean_absolute_error(y_t[extreme_mask], y_p[extreme_mask]))
        extreme_rmse = float(np.sqrt(mean_squared_error(y_t[extreme_mask], y_p[extreme_mask])))
        extreme_mbe = float(np.mean(y_p[extreme_mask] - y_t[extreme_mask]))
        extreme_count = int(np.sum(extreme_mask))
    else:
        extreme_mae, extreme_rmse, extreme_mbe, extreme_count = np.nan, np.nan, np.nan, 0

    return {
        'Model': model_name,
        'Split': split_name,
        'R2': round(r2, 4),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'MBE': round(mbe, 4),
        'sMAPE_%': round(smape, 2),
        'MASE': round(mase, 4) if not np.isnan(mase) else None,
        'Directional_Accuracy_%': round(dir_acc, 2),
        'Extreme_MAE': round(extreme_mae, 4) if not np.isnan(extreme_mae) else None,
        'Extreme_MBE': round(extreme_mbe, 4) if not np.isnan(extreme_mbe) else None,
        'Extreme_Count': extreme_count
    }

def evaluate_interval_coverage(y_true, lower_bounds, upper_bounds):
    """
    Calculate empirical coverage and average width of prediction intervals.
    """
    y_t = np.asarray(y_true).ravel()
    low = np.asarray(lower_bounds).ravel()
    high = np.asarray(upper_bounds).ravel()

    covered = (y_t >= low) & (y_t <= high)
    empirical_coverage = float(np.mean(covered) * 100.0)
    avg_width = float(np.mean(high - low))

    return {
        'Empirical_Coverage_%': round(empirical_coverage, 2),
        'Average_Interval_Width': round(avg_width, 2)
    }
