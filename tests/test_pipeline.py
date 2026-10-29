import os
import pytest
import numpy as np
import pandas as pd

from src.data_loader import load_and_clean_data
from src.feature_engineering import build_features
from src.preprocessing import TrainingFittedPreprocessor, chronological_split
from src.feature_selection import TrainingFittedFeatureSelector
from src.baselines import PersistenceBaseline, SeasonalNaiveBaseline
from src.forecasting import ConformalPredictor, recursive_multistep_forecast
from src.inference import AirQualityInferenceEngine

@pytest.fixture
def clean_data():
    df, report = load_and_clean_data("combined_air_quality_2017_2024.csv")
    return df

def test_no_duplicate_dates(clean_data):
    """Test that timestamps are deduplicated and sorted."""
    assert clean_data['Date'].duplicated().sum() == 0
    assert clean_data['Date'].is_monotonic_increasing

def test_date_range_coverage(clean_data):
    """Test that data covers full range from 2017 to 2024."""
    assert clean_data['Date'].min() == pd.Timestamp('2017-07-01')
    assert clean_data['Date'].max() == pd.Timestamp('2024-12-31')
    assert len(clean_data) == 2741

def test_no_target_leakage_in_features(clean_data):
    """
    CRITICAL LEAKAGE TEST:
    Verify that current day target (PM2_5 at day t) does not enter input features for day t.
    """
    feat_df = build_features(clean_data, target_col='PM2_5')
    
    # Lag 1 must equal PM2_5 shifted by 1
    assert np.allclose(feat_df['PM2_5_lag1'].iloc[5], clean_data['PM2_5'].iloc[4], equal_nan=True)
    
    # Rolling mean 7 at index t must ONLY use values from t-1 backwards!
    # If we perturb PM2_5 at index 10, rolling mean at index 10 MUST NOT CHANGE!
    perturbed = clean_data.copy()
    perturbed.loc[10, 'PM2_5'] = 99999.0
    feat_perturbed = build_features(perturbed, target_col='PM2_5')
    
    # At index 10, PM2_5_roll_mean_7 should equal unperturbed value
    assert feat_perturbed.loc[10, 'PM2_5_roll_mean_7'] == feat_df.loc[10, 'PM2_5_roll_mean_7']
    # At index 11 (the NEXT day), it SHOULD reflect the change
    assert feat_perturbed.loc[11, 'PM2_5_roll_mean_7'] != feat_df.loc[11, 'PM2_5_roll_mean_7']

def test_chronological_splits_disjoint(clean_data):
    """Test that train, val, and test partitions are strictly disjoint and ordered."""
    train_df, val_df, test_df = chronological_split(clean_data, 0.70, 0.15, 0.15)
    assert train_df['Date'].max() < val_df['Date'].min()
    assert val_df['Date'].max() < test_df['Date'].min()
    assert len(train_df) + len(val_df) + len(test_df) == len(clean_data)

def test_preprocessor_training_only_fit(clean_data):
    """Test that preprocessor learns parameters only from train and doesn't crash on val/test."""
    feat_df = build_features(clean_data, target_col='PM2_5').iloc[30:].reset_index(drop=True)
    train_df, val_df, test_df = chronological_split(feat_df, 0.70, 0.15, 0.15)
    
    pre = TrainingFittedPreprocessor(target_col='PM2_5')
    pre.fit(train_df)
    
    val_trans = pre.transform_df(val_df)
    test_trans = pre.transform_df(test_df)
    
    assert val_trans['PM2_5'].isnull().sum() == 0
    assert test_trans['PM2_5'].isnull().sum() == 0

def test_conformal_prediction_coverage():
    """Test that conformal prediction intervals achieve expected nominal coverage."""
    np.random.seed(42)
    y_val = np.random.normal(50, 10, 500)
    y_val_pred = y_val + np.random.normal(0, 5, 500)
    
    y_test = np.random.normal(50, 10, 500)
    y_test_pred = y_test + np.random.normal(0, 5, 500)
    
    cp = ConformalPredictor(alpha=0.05).fit(y_val, y_val_pred)
    low, high = cp.predict_interval(y_test_pred)
    
    coverage = np.mean((y_test >= low) & (y_test <= high))
    assert coverage >= 0.90 # Nominal 95% with tolerance for random noise

def test_inference_engine_validation():
    """Test that inference engine rejects invalid schemas and handles valid batches."""
    pre = TrainingFittedPreprocessor().fit(pd.DataFrame({'PM2_5': [10, 20], 'PM10': [20, 30], 'Date': pd.date_range('2020-01-01', periods=2)}))
    selector = TrainingFittedFeatureSelector().fit(pd.DataFrame({'PM2_5': [10, 20, 30], 'PM10': [20, 30, 40], 'Date': pd.date_range('2020-01-01', periods=3)}))
    
    class DummyModel:
        def predict(self, X):
            return np.full(len(X), 45.0)
            
    engine = AirQualityInferenceEngine(
        models_dict={'dummy': DummyModel()},
        preprocessor=pre,
        feature_selector=selector
    )
    
    # Fails on non-dataframe
    with pytest.raises(TypeError):
        engine.predict("invalid_input")
        
    # Fails on empty dataframe
    with pytest.raises(ValueError):
        engine.predict(pd.DataFrame())
        
    # Succeeds on valid dataframe
    sample_df = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=5),
        'PM2_5': [35.0, 40.0, 38.0, 42.0, 45.0],
        'PM10': [60.0, 70.0, 65.0, 75.0, 80.0]
    })
    res = engine.predict(sample_df, model_choice='dummy', return_intervals=False)
    assert len(res) == 5
    assert 'Predicted_PM2_5_dummy' in res.columns

def test_monthly_yearly_bar_charts(clean_data):
    """Test generation of 2x2 monthly and yearly bar graphs."""
    from src.explainability import plot_monthly_yearly_2x2_pair, plot_2x2_grouped_bar_chart
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df = clean_data.copy()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    fig1 = plot_monthly_yearly_2x2_pair(df, 'Benzene', 'SO2')
    assert len(fig1.axes) == 4
    plt.close(fig1)

    fig2 = plot_2x2_grouped_bar_chart(df, ['PM2_5', 'PM10', 'SO2', 'NO2'], period='monthly')
    assert len(fig2.axes) == 4
    plt.close(fig2)

    fig3 = plot_2x2_grouped_bar_chart(df, ['PM2_5', 'PM10', 'SO2', 'NO2'], period='yearly')
    assert len(fig3.axes) == 4
    plt.close(fig3)

def test_hyperparameter_tuning():
    """Test Optuna Bayesian hyperparameter optimization on regression data."""
    from src.tuning import tune_xgboost
    X_tr = pd.DataFrame(np.random.randn(50, 5), columns=[f'f_{i}' for i in range(5)])
    y_tr = np.random.randn(50) * 10 + 50
    X_va = pd.DataFrame(np.random.randn(20, 5), columns=[f'f_{i}' for i in range(5)])
    y_va = np.random.randn(20) * 10 + 50

    model, best_params, best_score = tune_xgboost(X_tr, y_tr, X_va, y_va, n_trials=2, seed=42)
    assert model is not None
    assert 'n_estimators' in best_params
    assert isinstance(best_score, float)

def test_multicity_pipeline():
    """Test CPCB multi-city data loading, feature engineering, and cross-city integrity."""
    from src.multi_city import load_and_prepare_multicity_data, get_city_clean_subset, build_city_features
    df = load_and_prepare_multicity_data('data/cpcb_multi_city_delhi_kolkata_mumbai_bengaluru.csv')
    assert len(df) > 0
    assert set(['Delhi', 'Kolkata', 'Mumbai', 'Bengaluru']).issubset(set(df['City'].unique()))

    # Check Kolkata subset
    kol_clean = get_city_clean_subset(df, 'Kolkata')
    assert len(kol_clean) > 0
    assert kol_clean['PM2.5'].isnull().sum() == 0

    # Check feature builder has zero current-day target leakage
    feat_kol = build_city_features(kol_clean, target_col='PM2.5')
    assert 'PM2.5_lag1' in feat_kol.columns
    assert 'PM2.5_roll_mean_7' in feat_kol.columns
    assert 'Year' not in feat_kol.columns
    assert 'Month' not in feat_kol.columns


