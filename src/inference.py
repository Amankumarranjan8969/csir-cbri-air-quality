import os
import joblib
import numpy as np
import pandas as pd
import torch

from src.feature_engineering import build_features
from src.models import (
    DeepAirResNet, predict_deep_air_resnet,
    PyTorchLSTMRegressor, predict_lstm_model,
    PyTorchANNRegressor, predict_ann_model,
    DEVICE
)

class AirQualityInferenceEngine:
    """
    Production-grade, validated inference engine for real-time and batch PM2.5 forecasting.
    Reuses fitted preprocessing, feature selection, and model artifacts without leakage.
    """
    def __init__(self, models_dict=None, preprocessor=None, feature_selector=None, conformal_predictor=None, superblend=None):
        self.models_dict = models_dict or {}
        self.preprocessor = preprocessor
        self.feature_selector = feature_selector
        self.conformal_predictor = conformal_predictor
        self.superblend = superblend

    def validate_input(self, df):
        """Strict input validation for inference dataframe."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        # Ensure Date column exists
        if 'Date' not in df.columns and 'Timestamp' in df.columns:
            df['Date'] = pd.to_datetime(df['Timestamp'], format='mixed', errors='coerce')

        if 'Date' not in df.columns or df['Date'].isnull().all():
            raise ValueError("Input DataFrame must contain a valid 'Date' or 'Timestamp' column.")

        return df.copy()

    def predict(self, df, model_choice='superblend', return_intervals=True):
        """
        Execute reliable predictions on input observations.
        
        Args:
            df (pd.DataFrame): Raw or preprocessed input data.
            model_choice (str): One of 'superblend', 'catboost', 'xgboost', 'deepairresnet'.
            return_intervals (bool): Whether to include conformal prediction intervals.
            
        Returns:
            results_df (pd.DataFrame): Predictions with Date, point estimates, and metadata.
        """
        validated_df = self.validate_input(df)

        # 1. Feature Engineering
        ref_min = validated_df['Date'].min()
        feat_df = build_features(validated_df, target_col='PM2_5', ref_min_date=ref_min)

        # 2. Re-apply fitted preprocessing (imputation & clipping)
        preprocessed_df = self.preprocessor.transform_df(feat_df)

        # 3. Align selected features
        selected_cols = self.feature_selector.selected_features_
        for col in selected_cols:
            if col not in preprocessed_df.columns:
                # Use fitted train median rather than blind zero!
                fallback_val = self.preprocessor.train_medians_.get(col, 0.0)
                preprocessed_df[col] = fallback_val

        X_in = preprocessed_df[selected_cols]

        # 4. Model Prediction
        choice = model_choice.lower().replace('-', '_')
        if choice in ['superblend', 'ensemble'] and self.superblend is not None:
            # Generate predictions for all models in blend
            blend_preds = {}
            for m in self.superblend.model_names:
                if m == 'DeepAirResNet':
                    X_sc = self.preprocessor.transform_scaled(X_in)
                    blend_preds[m] = predict_deep_air_resnet(self.models_dict[m], X_sc)
                elif m == 'LSTM':
                    X_sc = self.preprocessor.transform_scaled(X_in)
                    blend_preds[m] = predict_lstm_model(self.models_dict[m], X_sc, pad_estimator=self.models_dict.get('RandomForest'))
                elif m == 'ANN':
                    X_sc = self.preprocessor.transform_scaled(X_in)
                    blend_preds[m] = predict_ann_model(self.models_dict[m], X_sc)
                else:
                    blend_preds[m] = self.models_dict[m].predict(X_in)
            preds = self.superblend.predict(blend_preds)
            pred_col_name = "Predicted_PM2_5_SuperBlend"
        elif choice in ['deepairresnet', 'resnet']:
            X_sc = self.preprocessor.transform_scaled(X_in)
            preds = predict_deep_air_resnet(self.models_dict['DeepAirResNet'], X_sc)
            pred_col_name = "Predicted_PM2_5_DeepAirResNet"
        elif choice == 'lstm':
            X_sc = self.preprocessor.transform_scaled(X_in)
            preds = predict_lstm_model(self.models_dict['LSTM'], X_sc, pad_estimator=self.models_dict.get('RandomForest'))
            pred_col_name = "Predicted_PM2_5_LSTM"
        elif choice == 'ann':
            X_sc = self.preprocessor.transform_scaled(X_in)
            preds = predict_ann_model(self.models_dict['ANN'], X_sc)
            pred_col_name = "Predicted_PM2_5_ANN"
        elif choice == 'catboost':
            preds = self.models_dict['CatBoost'].predict(X_in)
            pred_col_name = "Predicted_PM2_5_CatBoost"
        elif choice == 'xgboost':
            preds = self.models_dict['XGBoost'].predict(X_in)
            pred_col_name = "Predicted_PM2_5_XGBoost"
        elif choice == 'lightgbm':
            preds = self.models_dict['LightGBM'].predict(X_in)
            pred_col_name = "Predicted_PM2_5_LightGBM"
        elif choice in ['randomforest', 'rf']:
            preds = self.models_dict['RandomForest'].predict(X_in)
            pred_col_name = "Predicted_PM2_5_RandomForest"
        elif choice in ['extratrees', 'et']:
            preds = self.models_dict['ExtraTrees'].predict(X_in)
            pred_col_name = "Predicted_PM2_5_ExtraTrees"
        else:
            first_model = list(self.models_dict.keys())[0]
            if first_model == 'DeepAirResNet':
                X_sc = self.preprocessor.transform_scaled(X_in)
                preds = predict_deep_air_resnet(self.models_dict[first_model], X_sc)
            elif first_model == 'LSTM':
                X_sc = self.preprocessor.transform_scaled(X_in)
                preds = predict_lstm_model(self.models_dict[first_model], X_sc, pad_estimator=self.models_dict.get('RandomForest'))
            elif first_model == 'ANN':
                X_sc = self.preprocessor.transform_scaled(X_in)
                preds = predict_ann_model(self.models_dict[first_model], X_sc)
            else:
                preds = self.models_dict[first_model].predict(X_in)
            pred_col_name = f"Predicted_PM2_5_{first_model}"

        preds = np.maximum(preds, 0.0)  # Physical concentration bound

        out_df = pd.DataFrame({
            'Date': preprocessed_df['Date'],
            pred_col_name: np.round(preds, 2),
            'Model_Version': 'v2.0-research',
            'Target': 'PM2.5 (µg/m³)'
        })

        if 'PM2_5' in validated_df.columns:
            out_df['Actual_PM2_5'] = validated_df['PM2_5'].values

        # 5. Conformal Prediction Intervals
        if return_intervals and self.conformal_predictor is not None:
            low, high = self.conformal_predictor.predict_interval(preds)
            out_df['Lower_PI_95'] = np.round(low, 2)
            out_df['Upper_PI_95'] = np.round(high, 2)

        return out_df
