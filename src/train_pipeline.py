import os
import time
import joblib
import numpy as np
import pandas as pd
import torch

from src.data_loader import load_and_clean_data
from src.feature_engineering import build_features
from src.preprocessing import TrainingFittedPreprocessor, chronological_split
from src.feature_selection import TrainingFittedFeatureSelector
from src.baselines import PersistenceBaseline, SeasonalNaiveBaseline, RollingMeanBaseline, LinearRegressionBaseline
from src.models import (
    train_tree_models, train_deep_air_resnet, predict_deep_air_resnet,
    train_lstm_model, predict_lstm_model, train_ann_model, predict_ann_model, DEVICE
)
from src.ensemble import ValidatedSuperBlend, TimeSeriesStackingEnsemble
from src.evaluate import evaluate_predictions, evaluate_interval_coverage
from src.forecasting import ConformalPredictor, generate_long_term_projection
from src.inference import AirQualityInferenceEngine

def run_full_pipeline(csv_path="combined_air_quality_2017_2024.csv", output_dir="models/saved_models_v2"):
    """
    Execute end-to-end research-grade, leakage-free forecasting pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    print("=" * 80)
    print("    CSIR-CBRI AIR QUALITY FORECASTING PIPELINE (V2.0 - LEAKAGE FREE)")
    print("=" * 80)

    # 1. Load and clean data
    print("\n[Step 1/8] Ingesting and validating CAAQMS observations...")
    df_clean, audit_rep = load_and_clean_data(csv_path)
    print(f"  -> Successfully deduplicated: {audit_rep['initial_rows']} -> {audit_rep['final_clean_rows']} continuous days.")
    print(f"  -> Timeline: {audit_rep['min_date']} to {audit_rep['max_date']} (Missing calendar days: {audit_rep['missing_calendar_days']})")

    # 2. Build strictly causal features
    print("\n[Step 2/8] Engineering causal features (Shifted rolling windows, lags, harmonics, trend)...")
    df_feat = build_features(df_clean, target_col='PM2_5')
    # Drop warm-up rows due to 30-day lag/rolling
    df_feat = df_feat.iloc[30:].reset_index(drop=True)
    print(f"  -> Feature matrix ready: {df_feat.shape[0]} days x {df_feat.shape[1]} raw/engineered columns.")

    # 3. Chronological Train / Validation / Test Partition (70% / 15% / 15%)
    print("\n[Step 3/8] Chronological partitioning (70% Train | 15% Val | 15% Test)...")
    train_df, val_df, test_df = chronological_split(df_feat, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    print(f"  -> Train split: {len(train_df)} days ({train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"  -> Val split:   {len(val_df)} days ({val_df['Date'].min().strftime('%Y-%m-%d')} to {val_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"  -> Test split:  {len(test_df)} days ({test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')})")

    # 4. Feature Selection on TRAINING set only! (Mutual Information + Absolute Pearson Correlation)
    print("\n[Step 4/8] Selecting predictors strictly on Training set (Dual MI + |Corr| criteria)...")
    selector = TrainingFittedFeatureSelector(target_col='PM2_5', min_mi_threshold=0.05, min_abs_corr_threshold=0.05)
    selector.fit(train_df)
    selected_features = selector.selected_features_
    print(f"  -> Selected {len(selected_features)} features (dropped {len(selector.dropped_features_)} uninformative/constant noise features).")

    # 5. Preprocessing fitted on TRAINING set only!
    print("\n[Step 5/8] Fitting imputation, outlier capping, and scalers strictly on Training set...")
    preprocessor = TrainingFittedPreprocessor(target_col='PM2_5', outlier_threshold=1.5, scaler_type='robust')
    preprocessor.fit(train_df, feature_cols=selected_features)

    train_trans = preprocessor.transform_df(train_df)
    val_trans = preprocessor.transform_df(val_df)
    test_trans = preprocessor.transform_df(test_df)

    X_train, y_train = train_trans[selected_features], train_trans['PM2_5'].values
    X_val, y_val = val_trans[selected_features], val_trans['PM2_5'].values
    X_test, y_test = test_trans[selected_features], test_trans['PM2_5'].values

    # Scaled matrices for Deep Learning
    X_tr_sc = preprocessor.transform_scaled(X_train)
    X_val_sc = preprocessor.transform_scaled(X_val)
    X_te_sc = preprocessor.transform_scaled(X_test)

    # 6. Fit Baselines
    print("\n[Step 6/8] Benchmarking baseline reference models...")
    baseline_models = {
        'Persistence (t-1)': PersistenceBaseline().fit(train_trans),
        'Weekly Seasonal-Naive (t-7)': SeasonalNaiveBaseline(season_lag=7).fit(train_trans),
        'Rolling Mean (7-day)': RollingMeanBaseline(window=7).fit(train_trans),
        'Ridge Linear Baseline': LinearRegressionBaseline().fit(X_train, y_train)
    }

    metrics_records = []
    val_predictions = {}
    test_predictions = {}

    for name, b_model in baseline_models.items():
        if isinstance(b_model, LinearRegressionBaseline):
            p_val = b_model.predict(X_val)
            p_test = b_model.predict(X_test)
        else:
            p_val = b_model.predict(val_trans)
            p_test = b_model.predict(test_trans)
        val_predictions[name] = p_val
        test_predictions[name] = p_test
        m_val = evaluate_predictions(y_val, p_val, model_name=name, split_name='Validation', y_train_history=y_train)
        m_test = evaluate_predictions(y_test, p_test, model_name=name, split_name='Test', y_train_history=y_train)
        metrics_records.extend([m_val, m_test])
        print(f"  [{name:<26}] -> Val R2: {m_val['R2']:7.4f} | Test R2: {m_test['R2']:7.4f} | Test RMSE: {m_test['RMSE']:7.4f} µg/m³")

    # 7. Train ML & Deep Learning Models
    print(f"\n[Step 7/8] Training ML & Deep Learning architectures (GPU: {DEVICE})...")
    tree_models, tree_times = train_tree_models(X_train, y_train, X_val, y_val)
    
    for name, model in tree_models.items():
        p_val = model.predict(X_val)
        p_test = model.predict(X_test)
        val_predictions[name] = p_val
        test_predictions[name] = p_test
        m_val = evaluate_predictions(y_val, p_val, model_name=name, split_name='Validation', y_train_history=y_train)
        m_test = evaluate_predictions(y_test, p_test, model_name=name, split_name='Test', y_train_history=y_train)
        m_test['Training_Time_s'] = round(tree_times.get(name, 0.0), 2)
        metrics_records.extend([m_val, m_test])
        print(f"  [{name:<26}] -> Val R2: {m_val['R2']:7.4f} | Test R2: {m_test['R2']:7.4f} | Test RMSE: {m_test['RMSE']:7.4f} µg/m³ (Time: {tree_times.get(name, 0.0):.1f}s)")

    # 2. DeepAirResNet
    print("  Training PyTorch DeepAirResNet...")
    dl_model, dl_time = train_deep_air_resnet(X_tr_sc, y_train, X_val_sc, y_val, epochs=80, lr=1e-3, device=DEVICE)
    p_val_dl = predict_deep_air_resnet(dl_model, X_val_sc, device=DEVICE)
    p_test_dl = predict_deep_air_resnet(dl_model, X_te_sc, device=DEVICE)
    val_predictions['DeepAirResNet'] = p_val_dl
    test_predictions['DeepAirResNet'] = p_test_dl
    m_val_dl = evaluate_predictions(y_val, p_val_dl, model_name='DeepAirResNet', split_name='Validation', y_train_history=y_train)
    m_test_dl = evaluate_predictions(y_test, p_test_dl, model_name='DeepAirResNet', split_name='Test', y_train_history=y_train)
    m_test_dl['Training_Time_s'] = round(dl_time, 2)
    metrics_records.extend([m_val_dl, m_test_dl])
    print(f"  [{'DeepAirResNet':<26}] -> Val R2: {m_val_dl['R2']:7.4f} | Test R2: {m_test_dl['R2']:7.4f} | Test RMSE: {m_test_dl['RMSE']:7.4f} µg/m³ (Time: {dl_time:.1f}s)")

    # 3. PyTorch LSTM Model
    print("  Training PyTorch LSTM Regressor...")
    lstm_model, lstm_time, lstm_seq = train_lstm_model(X_tr_sc, y_train, X_val_sc, y_val, seq_len=7, epochs=80, device=DEVICE)
    p_val_lstm = predict_lstm_model(lstm_model, X_val_sc, pad_estimator=tree_models.get('RandomForest'), seq_len=lstm_seq, device=DEVICE)
    p_test_lstm = predict_lstm_model(lstm_model, X_te_sc, pad_estimator=tree_models.get('RandomForest'), seq_len=lstm_seq, device=DEVICE)
    val_predictions['LSTM'] = p_val_lstm
    test_predictions['LSTM'] = p_test_lstm
    m_val_lstm = evaluate_predictions(y_val, p_val_lstm, model_name='LSTM', split_name='Validation', y_train_history=y_train)
    m_test_lstm = evaluate_predictions(y_test, p_test_lstm, model_name='LSTM', split_name='Test', y_train_history=y_train)
    m_test_lstm['Training_Time_s'] = round(lstm_time, 2)
    metrics_records.extend([m_val_lstm, m_test_lstm])
    print(f"  [{'LSTM':<26}] -> Val R2: {m_val_lstm['R2']:7.4f} | Test R2: {m_test_lstm['R2']:7.4f} | Test RMSE: {m_test_lstm['RMSE']:7.4f} µg/m³ (Time: {lstm_time:.1f}s)")

    # 4. PyTorch Artificial Neural Network (ANN)
    print("  Training PyTorch ANN Regressor...")
    ann_model, ann_time = train_ann_model(X_tr_sc, y_train, X_val_sc, y_val, epochs=80, lr=0.002, device=DEVICE)
    p_val_ann = predict_ann_model(ann_model, X_val_sc, device=DEVICE)
    p_test_ann = predict_ann_model(ann_model, X_te_sc, device=DEVICE)
    val_predictions['ANN'] = p_val_ann
    test_predictions['ANN'] = p_test_ann
    m_val_ann = evaluate_predictions(y_val, p_val_ann, model_name='ANN', split_name='Validation', y_train_history=y_train)
    m_test_ann = evaluate_predictions(y_test, p_test_ann, model_name='ANN', split_name='Test', y_train_history=y_train)
    m_test_ann['Training_Time_s'] = round(ann_time, 2)
    metrics_records.extend([m_val_ann, m_test_ann])
    print(f"  [{'ANN':<26}] -> Val R2: {m_val_ann['R2']:7.4f} | Test R2: {m_test_ann['R2']:7.4f} | Test RMSE: {m_test_ann['RMSE']:7.4f} µg/m³ (Time: {ann_time:.1f}s)")

    # 8. Train Ensembles & Uncertainty
    print("\n[Step 8/8] Building Validated Ensembles & Conformal Uncertainty...")
    # SuperBlend: fit strictly on Validation predictions across all models!
    blend_candidates = ['ExtraTrees', 'RandomForest', 'LightGBM', 'CatBoost', 'XGBoost', 'LSTM', 'ANN', 'DeepAirResNet']
    superblend = ValidatedSuperBlend()
    superblend.fit({m: val_predictions[m] for m in blend_candidates}, y_val)
    p_val_blend = superblend.predict({m: val_predictions[m] for m in blend_candidates})
    p_test_blend = superblend.predict({m: test_predictions[m] for m in blend_candidates})
    val_predictions['Hybrid_SuperBlend'] = p_val_blend
    test_predictions['Hybrid_SuperBlend'] = p_test_blend
    
    m_val_blend = evaluate_predictions(y_val, p_val_blend, model_name='Hybrid_SuperBlend', split_name='Validation', y_train_history=y_train)
    m_test_blend = evaluate_predictions(y_test, p_test_blend, model_name='Hybrid_SuperBlend', split_name='Test', y_train_history=y_train)
    metrics_records.extend([m_val_blend, m_test_blend])
    print(f"  [Hybrid_SuperBlend]        -> Val R2: {m_val_blend['R2']:7.4f} | Test R2: {m_test_blend['R2']:7.4f} | Test RMSE: {m_test_blend['RMSE']:7.4f} µg/m³")
    print(f"     Blend Weights: {superblend.get_weights_summary()}")

    # Conformal Prediction: calibrated on validation residuals
    conformal = ConformalPredictor(alpha=0.05).fit(y_val, p_val_blend)
    p_low_test, p_high_test = conformal.predict_interval(p_test_blend)
    cov_test = evaluate_interval_coverage(y_test, p_low_test, p_high_test)
    print(f"  [Conformal 95% Intervals]  -> Test Empirical Coverage: {cov_test['Empirical_Coverage_%']}% | Avg Width: {cov_test['Average_Interval_Width']} µg/m³")

    # Save benchmark table
    benchmark_df = pd.DataFrame(metrics_records)
    benchmark_df.to_csv("reports/benchmark_report.csv", index=False)
    print("\nBenchmark report successfully saved to reports/benchmark_report.csv")

    # Serialize artifacts
    all_models = {**tree_models, 'DeepAirResNet': dl_model, 'LSTM': lstm_model, 'ANN': ann_model}
    engine = AirQualityInferenceEngine(
        models_dict=all_models,
        preprocessor=preprocessor,
        feature_selector=selector,
        conformal_predictor=conformal,
        superblend=superblend
    )
    joblib.dump(engine, os.path.join(output_dir, "air_quality_inference_engine_v2.joblib"))
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
    joblib.dump(selector, os.path.join(output_dir, "feature_selector.joblib"))
    joblib.dump(superblend, os.path.join(output_dir, "superblend.joblib"))
    joblib.dump(conformal, os.path.join(output_dir, "conformal_predictor.joblib"))
    torch.save(dl_model.state_dict(), os.path.join(output_dir, "deep_air_resnet.pth"))
    torch.save(lstm_model.state_dict(), os.path.join(output_dir, "lstm_model.pth"))
    torch.save(ann_model.state_dict(), os.path.join(output_dir, "ann_model.pth"))
    print(f"All production artifacts serialized in {output_dir}/")

    return engine, benchmark_df, audit_rep
