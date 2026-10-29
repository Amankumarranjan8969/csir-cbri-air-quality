import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from src.data_loader import load_and_clean_data
from src.forecasting import generate_long_term_projection, recursive_multistep_forecast

st.set_page_config(page_title="CSIR-CBRI Air Quality Forecasting System", layout="wide", page_icon="🌤️")

st.title("🌤️ CSIR-CBRI: Ambient Air Quality Forecasting System (v2.0)")
st.markdown("""
**Author**: Aman Kumar | CSIR - Central Building Research Institute (CBRI), Roorkee  
**Platform**: Leakage-Free Machine Learning, Deep Learning & Conformal Prediction System  
*Note: This interface operates on validated historical CAAQMS station observations (2017–2024).*
""")

@st.cache_data
def get_clean_data():
    df, report = load_and_clean_data("combined_air_quality_2017_2024.csv")
    return df, report

@st.cache_resource
def load_engine():
    path = "models/saved_models_v2/air_quality_inference_engine_v2.joblib"
    if os.path.exists(path):
        return joblib.load(path)
    return None

df, report = get_clean_data()
engine = load_engine()

# Sidebar controls
st.sidebar.header("⚙️ Forecasting Configuration")
model_choice = st.sidebar.selectbox(
    "Select Model Architecture:",
    ["SuperBlend (Ensemble)", "CatBoost", "XGBoost", "DeepAirResNet (PyTorch)", "LightGBM"]
)

task = st.sidebar.radio(
    "Choose Forecasting Task:",
    ["Historical Evaluation & Actual vs Predicted", "Recursive Multi-Step Forecast (7/30 Days)", "5-Year Statistical Scenario Projection (2025–2030)"]
)

# Tab layout
tab1, tab2, tab3 = st.tabs(["📊 Historical Trends & Data Quality", "🔮 Predictions & Forecasting", "📋 Benchmark & Explainability"])

with tab1:
    st.subheader("CAAQMS Station Data Quality Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Timeline Span", f"{report['min_date']} to {report['max_date']}")
    col2.metric("Total Clean Days", f"{report['final_clean_rows']} Days")
    col3.metric("Duplicate Rows Removed", f"{report['deduplicated_dropped_rows']} Rows")
    col4.metric("Missing Calendar Days", f"{report['missing_calendar_days']} Days")

    st.subheader("Historical PM2.5 & PM10 Trajectories")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df['Date'], df['PM2_5'], label="PM2.5 (Daily Mean)", color="#e53e3e", alpha=0.8, linewidth=1.0)
    if 'PM10' in df.columns:
        ax.plot(df['Date'], df['PM10'], label="PM10 (Daily Mean)", color="#3182ce", alpha=0.5, linewidth=0.8)
    ax.axhline(60, color="#dd6b20", linestyle="--", label="India NAAQS 24-hr PM2.5 Limit (60 µg/m³)")
    ax.axhline(15, color="#38a169", linestyle=":", label="WHO PM2.5 Guideline (15 µg/m³)")
    ax.set_ylabel("Concentration (µg/m³)")
    ax.legend(loc="upper right")
    st.pyplot(fig)

with tab2:
    if task == "Historical Evaluation & Actual vs Predicted":
        st.subheader("Test-Set Evaluation (Actual vs. Predicted with Conformal 95% Intervals)")
        if engine is not None:
            # Predict on the last 100 days of the dataset
            sample_df = df.iloc[-120:].reset_index(drop=True)
            res = engine.predict(sample_df, model_choice=model_choice.split()[0].lower())
            
            fig, ax = plt.subplots(figsize=(14, 5))
            pred_col = [c for c in res.columns if 'Predicted' in c][0]
            ax.plot(res['Date'], res['Actual_PM2_5'], label="Observed Ground Truth", color="black", linewidth=1.5)
            ax.plot(res['Date'], res[pred_col], label=f"{model_choice} Prediction", color="#e53e3e", linewidth=1.5)
            if 'Lower_PI_95' in res.columns:
                ax.fill_between(res['Date'], res['Lower_PI_95'], res['Upper_PI_95'], color="#fed7d7", alpha=0.5, label="Conformal 95% Prediction Interval")
            ax.axhline(60, color="#dd6b20", linestyle="--", label="NAAQS Limit (60 µg/m³)")
            ax.set_ylabel("PM2.5 (µg/m³)")
            ax.legend(loc="upper right")
            st.pyplot(fig)
            st.dataframe(res.tail(15))
        else:
            st.warning("Model engine artifact not found. Please train models first.")

    elif task == "Recursive Multi-Step Forecast (7/30 Days)":
        st.subheader("Recursive Multi-Step Forecasting (Iterative Lag Updates)")
        horizon = st.slider("Select Forecast Horizon (Days):", min_value=1, max_value=30, value=7)
        if engine is not None and 'CatBoost' in engine.models_dict:
            last_row = df.iloc[-1:].copy()
            forecast_res = recursive_multistep_forecast(
                engine.models_dict['CatBoost'],
                last_row,
                engine.feature_selector,
                steps=horizon
            )
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(forecast_res['Forecast_Date'], forecast_res['Predicted_PM2_5'], marker='o', color="#805ad5", linewidth=2.0)
            ax.axhline(60, color="#dd6b20", linestyle="--", label="NAAQS 24-hr Limit (60 µg/m³)")
            ax.set_ylabel("Predicted PM2.5 (µg/m³)")
            ax.set_title(f"Recursive Next-{horizon}-Days PM2.5 Forecast (Starting {forecast_res['Forecast_Date'].min().strftime('%Y-%m-%d')})")
            ax.legend()
            st.pyplot(fig)
            st.dataframe(forecast_res)
            
    else: # 5-Year Statistical Scenario Projection
        st.subheader("5-Year Statistical Scenario Projection (2025–2030)")
        st.info("⚠️ Transparency Notice: This 5-year projection is a statistical scenario based on circular annual harmonics and secular trend decomposition, with an expanding uncertainty envelope. It does not claim deterministic day-by-day ML precision.")
        proj_df = generate_long_term_projection(df, forecast_days=1825)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df['Date'], df['PM2_5'], color="#718096", alpha=0.5, label="Historical Observations (2017–2024)")
        ax.plot(proj_df['Date'], proj_df['Projected_PM2_5'], color="#e53e3e", linewidth=1.5, label="5-Year Projection (2025–2030)")
        ax.fill_between(proj_df['Date'], proj_df['Lower_Bound_95'], proj_df['Upper_Bound_95'], color="#fed7d7", alpha=0.4, label="Expanding 95% Uncertainty Envelope")
        ax.axhline(60, color="#dd6b20", linestyle="--", label="NAAQS Standard (60 µg/m³)")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.legend()
        st.pyplot(fig)

with tab3:
    st.subheader("Scientific Model Benchmark (Leakage-Free Chronological Test)")
    if os.path.exists("reports/benchmark_report.csv"):
        bench_df = pd.read_csv("reports/benchmark_report.csv")
        st.dataframe(bench_df[bench_df['Split'] == 'Test'].sort_values('R2', ascending=False))
    else:
        st.info("Run `python -m src.train_pipeline` to generate reports/benchmark_report.csv.")
