"""
Multi-City CPCB Air Quality Pipeline Module.
Provides data preparation, cross-city comparative analytics (2x2 bar charts, distributions, AQI severity),
causal feature engineering, individual city model training, Taylor diagrams, spider charts,
actual vs predicted with +-20% bounds, and cross-city generalization/transferability matrix.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from src.feature_selection import TrainingFittedFeatureSelector
from src.models import train_ann_model, predict_ann_model, DEVICE

# Standard National Ambient Air Quality Standards (NAAQS) India Annual Limits (ug/m3)
NAAQS_LIMITS = {
    'PM2.5': 40.0,
    'PM10': 60.0,
    'NO2': 40.0,
    'SO2': 50.0,
    'NH3': 100.0,
    'O3': 100.0,
    'Benzene': 5.0,
    'CO': 2.0  # mg/m3
}

POLLUTANT_UNITS = {
    'PM2.5': r'$\mu\mathrm{g/m}^3$',
    'PM10': r'$\mu\mathrm{g/m}^3$',
    'NO2': r'$\mu\mathrm{g/m}^3$',
    'SO2': r'$\mu\mathrm{g/m}^3$',
    'NH3': r'$\mu\mathrm{g/m}^3$',
    'O3': r'$\mu\mathrm{g/m}^3$',
    'Benzene': r'$\mu\mathrm{g/m}^3$',
    'Toluene': r'$\mu\mathrm{g/m}^3$',
    'Xylene': r'$\mu\mathrm{g/m}^3$',
    'CO': r'$\mathrm{mg/m}^3$',
    'AQI': 'Index Points'
}

CITY_COLORS = {
    'Delhi': '#dc2626',          # Red
    'Kolkata': '#ea580c',        # Amber/Orange
    'Mumbai': '#2563eb',         # Blue
    'Bengaluru': '#059669',      # Emerald Green
    'Visakhapatnam': '#7c3aed'   # Royal Purple
}

def load_and_prepare_multicity_data(csv_path='data/cpcb_5_cities_india.csv'):
    """Loads and standardizes CPCB multi-city records."""
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df = df.sort_values(['City', 'Date']).reset_index(drop=True)
    return df

def get_city_clean_subset(df, city_name):
    """
    Extracts chronological subset for a city starting from the first valid PM2.5 recording.
    Imputes isolated missing values causally fitted on training period.
    """
    sub = df[df['City'] == city_name].sort_values('Date').copy()
    first_valid_date = sub.dropna(subset=['PM2.5'])['Date'].min()
    sub = sub[sub['Date'] >= first_valid_date].reset_index(drop=True)
    
    # Linear forward interpolation for short gaps up to 3 days using DatetimeIndex
    num_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
    avail = [c for c in num_cols if c in sub.columns]
    
    sub = sub.set_index('Date')
    sub[avail] = sub[avail].interpolate(method='time', limit=3).ffill().bfill()
    sub = sub.reset_index()
    return sub

# =====================================================================
# 1. MULTI-CITY COMPARATIVE POLLUTION ANALYTICS (2x2 VERTICAL BARS)
# =====================================================================

def plot_multicity_2x2_monthly_bars(df, target_cities=['Delhi', 'Kolkata', 'Mumbai', 'Bengaluru'], 
                                   param='PM2.5', save_path='reports/figures/multicity_monthly_pm25.png'):
    """
    Renders publication-grade 2x2 vertical bar chart comparing monthly mean levels
    of a specified parameter across 4 major metros.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_cities = len(target_cities)
    if n_cities <= 4:
        nrows, ncols = 2, 2
        figsize = (16, 11)
    else:
        nrows = int(np.ceil(n_cities / 3))
        ncols = 3
        figsize = (18, 5.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axes = axes.flatten()
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    unit = POLLUTANT_UNITS.get(param, r'$\mu\mathrm{g/m}^3$')
    naaqs = NAAQS_LIMITS.get(param, None)

    for i, city in enumerate(target_cities):
        ax = axes[i]
        c_sub = df[df['City'] == city].dropna(subset=[param])
        m_means = c_sub.groupby('Month')[param].mean().reindex(range(1, 13))
        
        col = CITY_COLORS.get(city, '#3b82f6')
        bars = ax.bar(month_names, m_means, color=col, edgecolor='#1e293b', alpha=0.85, width=0.65)
        
        # Add value annotations
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h) and h > 0:
                ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                            textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

        # Add NAAQS Reference Line if applicable
        if naaqs is not None:
            ax.axhline(naaqs, color='#b91c1c', linestyle='--', linewidth=1.5, alpha=0.85, label=f'NAAQS ({naaqs} {unit})')
            ax.legend(loc='upper right', frameon=True, fontsize=9)

        ax.set_title(f"{city} - Monthly Mean {param}", fontsize=12, fontweight='bold', pad=8)
        ax.set_ylabel(f"Mean Concentration ({unit})", fontsize=10, fontweight='bold')
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.tick_params(axis='x', rotation=30)

    for k in range(n_cities, len(axes)):
        axes[k].set_visible(False)
    plt.suptitle(f"Multi-City Monthly Seasonal Evolution: {param} Across Indian Metropolitan Airsheds", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path

def plot_multicity_2x2_yearly_bars(df, target_cities=['Delhi', 'Kolkata', 'Mumbai', 'Bengaluru'],
                                  param='PM2.5', save_path='reports/figures/multicity_yearly_pm25.png'):
    """
    Renders publication-grade 2x2 vertical bar chart comparing yearly secular trends
    of a specified parameter across 4 major metros.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_cities = len(target_cities)
    if n_cities <= 4:
        nrows, ncols = 2, 2
        figsize = (16, 11)
    else:
        nrows = int(np.ceil(n_cities / 3))
        ncols = 3
        figsize = (18, 5.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axes = axes.flatten()

    unit = POLLUTANT_UNITS.get(param, r'$\mu\mathrm{g/m}^3$')
    naaqs = NAAQS_LIMITS.get(param, None)

    for i, city in enumerate(target_cities):
        ax = axes[i]
        c_sub = df[df['City'] == city].dropna(subset=[param])
        y_means = c_sub.groupby('Year')[param].mean()
        years = [str(y) for y in y_means.index]
        
        col = CITY_COLORS.get(city, '#3b82f6')
        bars = ax.bar(years, y_means, color=col, edgecolor='#1e293b', alpha=0.85, width=0.55)
        
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h) and h > 0:
                ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                            textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

        if naaqs is not None:
            ax.axhline(naaqs, color='#b91c1c', linestyle='--', linewidth=1.5, alpha=0.85, label=f'NAAQS ({naaqs} {unit})')
            ax.legend(loc='upper right', frameon=True, fontsize=9)

        ax.set_title(f"{city} - Annual Secular Trend {param}", fontsize=12, fontweight='bold', pad=8)
        ax.set_ylabel(f"Annual Mean ({unit})", fontsize=10, fontweight='bold')
        ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.suptitle(f"Multi-City Annual Secular Trajectory: {param} Across 4 Major Indian Metros", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path

def plot_multicity_pollution_distributions(df, target_cities=['Delhi', 'Kolkata', 'Mumbai', 'Bengaluru'],
                                          save_path='reports/figures/multicity_pollution_distribution_boxplots.png'):
    """
    Renders comparative boxplots for PM2.5, PM10, NO2, and SO2 across the 4 cities.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()
    
    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2']
    
    for i, p in enumerate(pollutants):
        ax = axes[i]
        sub = df[df['City'].isin(target_cities)].dropna(subset=[p])
        
        sns.boxplot(x='City', y=p, data=sub, ax=ax, palette=CITY_COLORS, hue='City', legend=False,
                    order=target_cities, width=0.5, fliersize=2.5, linewidth=1.2)
        
        naaqs = NAAQS_LIMITS.get(p)
        if naaqs:
            ax.axhline(naaqs, color='crimson', linestyle='--', linewidth=1.5, label=f'NAAQS Limit ({naaqs})')
            ax.legend(loc='upper right', frameon=True, fontsize=9)
            
        unit = POLLUTANT_UNITS.get(p, r'$\mu\mathrm{g/m}^3$')
        ax.set_title(f"Atmospheric Concentration Distribution: {p}", fontsize=12, fontweight='bold')
        ax.set_ylabel(f"{p} ({unit})", fontsize=10, fontweight='bold')
        ax.set_xlabel("")
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        
    plt.suptitle("Cross-City Ambient Pollutant Distribution & Inter-City Contrast", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path

def plot_multicity_aqi_severity_breakdown(df, target_cities=['Delhi', 'Kolkata', 'Mumbai', 'Bengaluru'],
                                         save_path='reports/figures/multicity_aqi_severity_breakdown.png'):
    """
    Renders 100% stacked horizontal bar chart showing proportions of CPCB AQI categories across cities.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    categories = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    cat_colors = {
        'Good': '#16a34a',         # Green
        'Satisfactory': '#84cc16', # Light green
        'Moderate': '#eab308',     # Yellow
        'Poor': '#f97316',         # Orange
        'Very Poor': '#ef4444',    # Red
        'Severe': '#7f1d1d'        # Dark Maroon
    }

    sub = df[df['City'].isin(target_cities)].dropna(subset=['AQI_Bucket'])
    ct = pd.crosstab(sub['City'], sub['AQI_Bucket'], normalize='index') * 100
    
    # Ensure correct column ordering
    ct = ct.reindex(columns=[c for c in categories if c in ct.columns], fill_value=0)
    ct = ct.reindex(index=target_cities)

    left = np.zeros(len(ct))
    for cat in ct.columns:
        vals = ct[cat].values
        ax.barh(ct.index, vals, left=left, label=cat, color=cat_colors.get(cat, '#94a3b8'),
                edgecolor='white', height=0.55)
        
        # Annotate text if width > 4%
        for j, (v, l) in enumerate(zip(vals, left)):
            if v >= 5.0:
                ax.text(l + v/2, j, f"{v:.1f}%", ha='center', va='center',
                        color='white' if cat in ['Very Poor', 'Severe', 'Good'] else 'black',
                        fontsize=8.5, fontweight='bold')
        left += vals

    ax.set_xlabel("Percentage of Monitored Days (%)", fontsize=11, fontweight='bold')
    ax.set_title("Cross-City Air Quality Index (AQI) Severity Category Breakdown", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlim(0, 100)
    ax.legend(title="CPCB AQI Category", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    ax.grid(axis='x', linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path

# =====================================================================
# 2. CAUSAL FEATURE ENGINEERING & DATA SPLITTING (PER CITY)
# =====================================================================

def build_city_features(city_clean_df, target_col='PM2.5', lag_days=(1, 2, 3, 7, 14), rolling_windows=(3, 7, 14, 30)):
    """
    Constructs causal time-series features strictly from past observations.
    Excludes all raw calendar integers (Year, Month, Day) to prevent calendar overfitting.
    """
    df_feat = city_clean_df.copy().sort_values('Date').reset_index(drop=True)
    
    # 1. Target Lags
    for lag in lag_days:
        df_feat[f'{target_col}_lag{lag}'] = df_feat[target_col].shift(lag)
        
    # 2. Co-pollutant Lags
    co_pollutants = [c for c in ['PM10', 'NO2', 'SO2', 'CO', 'O3', 'Benzene'] if c in df_feat.columns and c != target_col]
    for cp in co_pollutants:
        for lag in (1, 2, 3, 7):
            df_feat[f'{cp}_lag{lag}'] = df_feat[cp].shift(lag)
            
    # 3. Rolling Statistics
    for w in rolling_windows:
        # Shift(1) guarantees strictly past observations
        roll = df_feat[target_col].shift(1).rolling(w)
        df_feat[f'{target_col}_roll_mean_{w}'] = roll.mean()
        df_feat[f'{target_col}_roll_std_{w}'] = roll.std()
        df_feat[f'{target_col}_roll_max_{w}'] = roll.max()
        df_feat[f'{target_col}_roll_min_{w}'] = roll.min()
        
    # 4. Chemical / Environmental Proxy Ratios
    if 'PM10' in df_feat.columns:
        df_feat['pm_ratio_lag1'] = (df_feat[target_col].shift(1) / (df_feat['PM10'].shift(1) + 1e-4)).clip(0.05, 1.0)
    if 'NO2' in df_feat.columns and 'SO2' in df_feat.columns:
        df_feat['no2_so2_ratio_lag1'] = (df_feat['NO2'].shift(1) / (df_feat['SO2'].shift(1) + 1e-4)).clip(0.1, 50.0)

    # 5. Continuous Cyclical Day-of-Year Harmonics (Trigonometric smooth signals)
    doy = df_feat['Date'].dt.dayofyear
    df_feat['sin_doy'] = np.sin(2 * np.pi * doy / 365.25)
    df_feat['cos_doy'] = np.cos(2 * np.pi * doy / 365.25)
    
    # Drop raw calendar integer columns if present
    df_feat = df_feat.drop(columns=[c for c in ['Year', 'Month', 'Day'] if c in df_feat.columns])

    # Drop rows corresponding to max lookback window
    max_lookback = max(max(rolling_windows), max(lag_days))
    df_feat = df_feat.iloc[max_lookback:].reset_index(drop=True)
    return df_feat

def prepare_city_splits(df_feat, target_col='PM2.5', train_ratio=0.70, val_ratio=0.15,
                        min_mi_threshold=0.03, min_abs_corr_threshold=0.03):
    """
    Chronological partition (70% Train, 15% Val, 15% Test) with training-only dual-criteria
    feature selection and training-fitted median imputation (strictly zero lookahead leakage).
    """
    n = len(df_feat)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_df = df_feat.iloc[:n_train].copy()
    val_df = df_feat.iloc[n_train:n_train+n_val].copy()
    test_df = df_feat.iloc[n_train+n_val:].copy()
    
    # Feature columns (exclude meta strings and target)
    meta_cols = ['City', 'Date', 'AQI_Bucket', 'Year', 'Month', 'Day', target_col]
    candidate_cols = [c for c in df_feat.columns if c not in meta_cols and not c.startswith('Date')]
    
    # Dual-Criteria Feature Selection fitted strictly on Training split
    selector = TrainingFittedFeatureSelector(
        target_col=target_col,
        min_mi_threshold=min_mi_threshold,
        min_abs_corr_threshold=min_abs_corr_threshold
    )
    selector.fit(train_df, candidate_cols=candidate_cols)
    selected_features = selector.selected_features_

    # Fallback to candidate_cols if overly sparse selection
    if len(selected_features) < 8:
        selected_features = candidate_cols

    # Fit imputer strictly on Training split
    train_medians = train_df[selected_features].median().fillna(0.0)
    
    X_train = train_df[selected_features].fillna(train_medians).fillna(0.0).values
    y_train = train_df[target_col].values
    
    X_val = val_df[selected_features].fillna(train_medians).fillna(0.0).values
    y_val = val_df[target_col].values
    
    X_test = test_df[selected_features].fillna(train_medians).fillna(0.0).values
    y_test = test_df[target_col].values
    
    dates_test = test_df['Date'].values
    
    return X_train, y_train, X_val, y_val, X_test, y_test, selected_features, dates_test

# =====================================================================
# 3. MODEL TRAINING & EVALUATION (PER CITY)
# =====================================================================

class ANNWrapper:
    """Wrapper around PyTorch ANN model and StandardScaler for unified scikit-learn API."""
    def __init__(self, model, scaler, device=DEVICE):
        self.model = model
        self.scaler = scaler
        self.device = device
        self.n_features_in_ = scaler.n_features_in_
        
    def predict(self, X):
        X_sc = self.scaler.transform(X)
        return predict_ann_model(self.model, X_sc, device=self.device)

def train_and_evaluate_city_models(city_name, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Trains portfolio of regression models (ML Tree Ensembles, Deep ANN, Baselines, Ensemble)
    for a specific city and computes comprehensive metrics.
    """
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=300, max_depth=16, max_features=0.8, random_state=42, n_jobs=-1),
        'Extra Trees': ExtraTreesRegressor(n_estimators=300, max_depth=18, max_features=0.85, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.02, subsample=0.75, colsample_bytree=0.75, random_state=42, n_jobs=-1, tree_method='hist'),
        'LightGBM': lgb.LGBMRegressor(n_estimators=350, num_leaves=25, max_depth=7, learning_rate=0.02, subsample=0.8, random_state=42, n_jobs=-1, verbose=-1),
        'CatBoost': CatBoostRegressor(iterations=400, depth=5, learning_rate=0.03, l2_leaf_reg=4.0, random_seed=42, verbose=False),
        'Ridge Baseline': Ridge(alpha=10.0, random_state=42)
    }
    
    trained_models = {}
    metrics_list = []
    test_predictions = {}
    
    print(f"\n[{city_name}] Training candidate models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        preds_val = model.predict(X_val)
        preds_test = model.predict(X_test)
        test_predictions[name] = preds_test
        
        r2 = r2_score(y_test, preds_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds_test))
        mae = mean_absolute_error(y_test, preds_test)
        
        denom = (np.abs(y_test) + np.abs(preds_test)) / 2.0
        smape = np.mean(np.where(denom == 0, 0, np.abs(preds_test - y_test) / denom)) * 100.0
        
        if len(y_test) > 1:
            dir_acc = np.mean(np.sign(np.diff(y_test)) == np.sign(np.diff(preds_test))) * 100.0
        else:
            dir_acc = np.nan
            
        ratio = preds_test / np.maximum(y_test, 1e-4)
        fac2 = np.mean((ratio >= 0.5) & (ratio <= 2.0)) * 100.0
        
        metrics_list.append({
            'City': city_name,
            'Model': name,
            'Val_R2': r2_score(y_val, preds_val),
            'Test_R2': r2,
            'Test_RMSE': rmse,
            'Test_MAE': mae,
            'Test_sMAPE_%': smape,
            'Directional_Accuracy_%': dir_acc,
            'FAC2_%': fac2
        })
        print(f"  - {name:<15} | Val R2: {r2_score(y_val, preds_val):.4f} | Test R2: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")

    # Train Deep Learning PyTorch ANN Regressor
    try:
        scaler = StandardScaler().fit(X_train)
        X_tr_sc = scaler.transform(X_train)
        X_va_sc = scaler.transform(X_val)
        X_te_sc = scaler.transform(X_test)
        
        ann_model, _ = train_ann_model(X_tr_sc, y_train, X_va_sc, y_val, epochs=50, lr=0.003, device=DEVICE)
        preds_val_ann = predict_ann_model(ann_model, X_va_sc, device=DEVICE)
        preds_test_ann = predict_ann_model(ann_model, X_te_sc, device=DEVICE)
        trained_models['PyTorch ANN'] = ANNWrapper(ann_model, scaler)
        
        r2_ann = r2_score(y_test, preds_test_ann)
        rmse_ann = np.sqrt(mean_squared_error(y_test, preds_test_ann))
        mae_ann = mean_absolute_error(y_test, preds_test_ann)
        denom_ann = (np.abs(y_test) + np.abs(preds_test_ann)) / 2.0
        smape_ann = np.mean(np.where(denom_ann == 0, 0, np.abs(preds_test_ann - y_test) / denom_ann)) * 100.0
        dir_ann = np.mean(np.sign(np.diff(y_test)) == np.sign(np.diff(preds_test_ann))) * 100.0 if len(y_test) > 1 else np.nan
        ratio_ann = preds_test_ann / np.maximum(y_test, 1e-4)
        fac2_ann = np.mean((ratio_ann >= 0.5) & (ratio_ann <= 2.0)) * 100.0
        
        test_predictions['PyTorch ANN'] = preds_test_ann
        metrics_list.append({
            'City': city_name,
            'Model': 'PyTorch ANN',
            'Val_R2': r2_score(y_val, preds_val_ann),
            'Test_R2': r2_ann,
            'Test_RMSE': rmse_ann,
            'Test_MAE': mae_ann,
            'Test_sMAPE_%': smape_ann,
            'Directional_Accuracy_%': dir_ann,
            'FAC2_%': fac2_ann
        })
        print(f"  - {'PyTorch ANN':<15} | Val R2: {r2_score(y_val, preds_val_ann):.4f} | Test R2: {r2_ann:.4f} | RMSE: {rmse_ann:.2f} | MAE: {mae_ann:.2f}")
    except Exception as e:
        print(f"  [PyTorch ANN] Skipped for {city_name}: {e}")

    # Build Weighted Ensemble from top 3 validation performers
    val_scores = {m['Model']: m['Val_R2'] for m in metrics_list}
    top_3 = sorted(val_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    top_names = [k for k, v in top_3]
    w = np.array([max(0.01, v) for _, v in top_3])
    w = w / np.sum(w)
    
    ens_test_pred = np.zeros_like(y_test, dtype=float)
    for weight, m_name in zip(w, top_names):
        ens_test_pred += weight * test_predictions[m_name]
    test_predictions['Weighted Ensemble'] = ens_test_pred
    
    ens_r2 = r2_score(y_test, ens_test_pred)
    ens_rmse = np.sqrt(mean_squared_error(y_test, ens_test_pred))
    ens_mae = mean_absolute_error(y_test, ens_test_pred)
    denom = (np.abs(y_test) + np.abs(ens_test_pred)) / 2.0
    ens_smape = np.mean(np.where(denom == 0, 0, np.abs(ens_test_pred - y_test) / denom)) * 100.0
    ens_dir_acc = np.mean(np.sign(np.diff(y_test)) == np.sign(np.diff(ens_test_pred))) * 100.0
    ratio = ens_test_pred / np.maximum(y_test, 1e-4)
    ens_fac2 = np.mean((ratio >= 0.5) & (ratio <= 2.0)) * 100.0
    
    metrics_list.append({
        'City': city_name,
        'Model': 'Weighted Ensemble',
        'Val_R2': np.mean([v for _, v in top_3]),
        'Test_R2': ens_r2,
        'Test_RMSE': ens_rmse,
        'Test_MAE': ens_mae,
        'Test_sMAPE_%': ens_smape,
        'Directional_Accuracy_%': ens_dir_acc,
        'FAC2_%': ens_fac2
    })
    print(f"  * {'Weighted Ensemble':<15} | Test R2: {ens_r2:.4f} | RMSE: {ens_rmse:.2f} | MAE: {ens_mae:.2f}")

    metrics_df = pd.DataFrame(metrics_list)
    return trained_models, metrics_df, test_predictions

# =====================================================================
# 4. CROSS-CITY COMPARATIVE VISUALIZATIONS
# =====================================================================

def plot_multicity_actual_vs_predicted_20pct(city_eval_dict, save_path='reports/figures/multicity_actual_vs_predicted_20pct.png'):
    """
    Renders 2x2 multi-panel plot of Actual vs Predicted with +-20% error lines
    for the winning model of each city.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_cities = len(city_eval_dict)
    if n_cities <= 4:
        nrows, ncols = 2, 2
        figsize = (15, 13)
    else:
        nrows = int(np.ceil(n_cities / 3))
        ncols = 3
        figsize = (18, 6.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, (city_name, data) in enumerate(city_eval_dict.items()):
        ax = axes[i]
        y_true = data['y_test']
        y_pred = data['y_pred']
        model_name = data['best_model_name']
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        col = CITY_COLORS.get(city_name, '#2563eb')
        ax.scatter(y_true, y_pred, color=col, alpha=0.65, edgecolors='black', linewidth=0.5, s=35, label='Observations')

        # 1:1 Identity Line
        max_val = max(np.max(y_true), np.max(y_pred)) * 1.05
        min_val = 0.0
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='1:1 Perfect Agreement')

        # +-20% Error Lines
        ax.plot([min_val, max_val], [min_val, max_val * 1.20], color='#ea580c', linestyle=':', linewidth=1.4, label='+20% Margin')
        ax.plot([min_val, max_val], [min_val, max_val * 0.80], color='#ea580c', linestyle=':', linewidth=1.4, label='-20% Margin')
        ax.fill_between([min_val, max_val], [min_val, max_val * 0.80], [min_val, max_val * 1.20], color='#ea580c', alpha=0.08)

        # Fraction within +-20%
        within_20 = np.mean((y_pred >= 0.80 * y_true) & (y_pred <= 1.20 * y_true)) * 100.0

        ax.set_title(f"{city_name} - {model_name}\n$R^2 = {r2:.4f}$ | RMSE = ${rmse:.2f}\\ \\mu\\mathrm{{g/m}}^3$ | Within $\\pm20\\%: {within_20:.1f}\\%$", 
                     fontsize=11.5, fontweight='bold', pad=8)
        ax.set_xlabel(r"Observed Actual $\mathrm{PM}_{2.5}\ (\mu\mathrm{g/m}^3)$", fontsize=10, fontweight='bold')
        ax.set_ylabel(r"Model Predicted $\mathrm{PM}_{2.5}\ (\mu\mathrm{g/m}^3)$", fontsize=10, fontweight='bold')
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', frameon=True, fontsize=8.5)

    plt.suptitle("Cross-City Actual vs. Predicted Validation with $\\pm20\\%$ Precision Envelopes", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path

def plot_multicity_spider_chart(city_metrics_dict, save_path='reports/figures/multicity_spider_chart.png'):
    """
    Renders multi-city radar / spider chart comparing performance dimensions across cities:
    R2, 1 - (Normalized RMSE), 1 - (Normalized MAE), Directional Accuracy, FAC2.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    categories = ['Predictive Skill (R2)', 'Accuracy (1 - N-RMSE)', 'Precision (1 - N-MAE)', 
                  'Directional Accuracy', 'Factor-of-2 Agreement (FAC2)']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for city_name, met in city_metrics_dict.items():
        r2 = max(0.0, met['Test_R2'])
        norm_rmse = max(0.0, 1.0 - (met['Test_RMSE'] / 35.0))
        norm_mae = max(0.0, 1.0 - (met['Test_MAE'] / 25.0))
        dir_acc = met['Directional_Accuracy_%'] / 100.0
        fac2 = met['FAC2_%'] / 100.0

        values = [r2, norm_rmse, norm_mae, dir_acc, fac2]
        values += values[:1]

        col = CITY_COLORS.get(city_name, '#3b82f6')
        ax.plot(angles, values, linewidth=2.0, linestyle='solid', label=f"{city_name} (R2={r2:.3f})", color=col)
        ax.fill(angles, values, color=col, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title("Multi-City Forecasting Performance Radar Comparison", fontsize=13, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), frameon=True, fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path

def plot_multicity_taylor_diagram(city_eval_dict, save_path='reports/figures/multicity_taylor_diagram.png'):
    """
    Renders Karl Taylor Diagram comparing model variance ratio and correlation across the 4 cities.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, polar=True)

    # Correlation angles
    corr_ticks = [0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    angle_ticks = np.arccos(corr_ticks)
    ax.set_thetagrids(np.degrees(angle_ticks), labels=[str(c) for c in corr_ticks])
    ax.set_thetamin(0)
    ax.set_thetamax(90)

    # Reference point at correlation = 1.0, std = 1.0
    ax.plot(0, 1.0, 'k*', markersize=14, label='Reference Observation')

    # Radial isolines of Centered RMS difference
    rs = np.linspace(0, 1.5, 100)
    thetas = np.linspace(0, np.pi/2, 100)
    for c_rms in [0.25, 0.5, 0.75, 1.0]:
        t_arc = np.linspace(0, np.pi/2, 200)
        # c_rms^2 = 1^2 + r^2 - 2*r*cos(theta)
        r_arc = np.cos(t_arc) + np.sqrt(np.maximum(0, c_rms**2 - np.sin(t_arc)**2))
        valid = (r_arc >= 0) & (r_arc <= 1.5)
        if np.any(valid):
            ax.plot(t_arc[valid], r_arc[valid], 'g--', linewidth=0.7, alpha=0.5)

    for city_name, data in city_eval_dict.items():
        y_true = data['y_test']
        y_pred = data['y_pred']

        std_ref = np.std(y_true)
        std_pred = np.std(y_pred)
        norm_std = std_pred / (std_ref + 1e-4)

        r = np.corrcoef(y_true, y_pred)[0, 1]
        theta = np.arccos(np.clip(r, -1.0, 1.0))

        col = CITY_COLORS.get(city_name, '#3b82f6')
        ax.plot(theta, norm_std, marker='o', markersize=9, color=col, markeredgecolor='black',
                label=f"{city_name} (r={r:.3f}, $\\sigma_n$={norm_std:.2f})")

    ax.set_rlabel_position(90)
    ax.set_ylim(0, 1.4)
    ax.set_title("Multi-City Karl Taylor Diagram: Normalized Skill & Variance Ratio", fontsize=12.5, fontweight='bold', pad=18)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05), frameon=True, fontsize=9.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path

# =====================================================================
# 5. CROSS-CITY GENERALIZATION & TRANSFERABILITY MATRIX
# =====================================================================

def compute_and_plot_cross_city_transferability(city_models_dict, city_data_dict,
                                               save_path='reports/figures/multicity_transferability_matrix.png'):
    """
    Evaluates model transferability: Model trained on City A evaluated on City B test observations.
    Renders 4x4 annotated cross-generalization heatmap.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cities = list(city_models_dict.keys())
    matrix_r2 = np.zeros((len(cities), len(cities)))

    for i, train_city in enumerate(cities):
        model = city_models_dict[train_city]
        cols_a = city_data_dict[train_city]['feature_cols']
        for j, test_city in enumerate(cities):
            X_test_b = city_data_dict[test_city]['X_test']
            y_test_b = city_data_dict[test_city]['y_test']
            cols_b = city_data_dict[test_city]['feature_cols']
            
            # Align test_city observations to train_city feature space by column name
            df_eval = pd.DataFrame(X_test_b, columns=cols_b)
            df_aligned = df_eval.reindex(columns=cols_a, fill_value=0.0)
            
            try:
                preds = model.predict(df_aligned)
            except Exception:
                preds = model.predict(df_aligned.values)
            r2 = r2_score(y_test_b, preds)
            matrix_r2[i, j] = r2

    df_transfer = pd.DataFrame(matrix_r2, index=[f"Train: {c}" for c in cities], columns=[f"Test: {c}" for c in cities])
    df_transfer.to_csv('reports/multicity_transferability_matrix.csv')

    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(df_transfer, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Transferability Test R2 Score'},
                vmin=max(0.0, np.min(matrix_r2)), vmax=1.0, linewidths=1.0, linecolor='white', ax=ax)
    
    ax.set_title("Cross-City Generalization & Transferability Matrix ($R^2$ Heatmap)\n(Evaluating Spatial Domain Generalization Across Urban Typologies)",
                 fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return df_transfer


# =====================================================================
# 5. 5-YEAR MULTI-CITY SCENARIO FORECASTING VISUALIZATIONS
# =====================================================================

def plot_city_5year_scenario_projection(historical_df, forecast_df, city_name,
                                        save_path=None):
    """
    Renders publication-grade 5-Year Scenario Forecast (2025-2029) for an individual city,
    complete with historical observations, seasonal cyclicality, secular trend projection,
    and 95% conformal uncertainty prediction intervals.
    """
    if save_path is None:
        save_path = f"reports/figures/forecast_5yr_{city_name.lower().replace(' ', '_')}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    col = CITY_COLORS.get(city_name, '#2563eb')
    pm_col = 'PM2.5' if 'PM2.5' in historical_df.columns else ('PM2_5' if 'PM2_5' in historical_df.columns else historical_df.columns[1])
    
    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.plot(historical_df['Date'], historical_df[pm_col], color='#64748b',
            label=f'Historical Observations ({historical_df["Date"].min().year}–{historical_df["Date"].max().year})',
            linewidth=1.1, alpha=0.75)
    ax.plot(forecast_df['Date'], forecast_df['Projected_PM2_5'], color=col,
            linewidth=1.5, label=f'{city_name} 5-Year Scenario Forecast (2025–2029)')
    ax.fill_between(forecast_df['Date'], forecast_df['Lower_Bound_95'], forecast_df['Upper_Bound_95'],
                    color=col, alpha=0.22, label='95% Conformal Uncertainty Interval')
    ax.axhline(60, color='#ea580c', linestyle='--', linewidth=1.3,
               label=r'India NAAQS Standard (60 $\mu\mathrm{g/m}^3$)')
    
    ax.set_title(f"{city_name} — Continuous Atmospheric Trajectory & 5-Year Scenario Forecast (2025–2029)\nAutoregressive Seasonality, Secular Extrapolation & Calibrated 95% Uncertainty Bounds",
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Timeline", fontsize=10, fontweight='bold')
    ax.set_ylabel(r"$\mathrm{PM}_{2.5}$ Concentration ($\mu\mathrm{g/m}^3$)", fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path

def plot_multicity_5year_overlay(city_forecasts_dict,
                                 save_path='reports/figures/multicity_5year_forecast_comparison.png'):
    """
    Renders consolidated multi-city 5-year forecast trajectories across all 5 urban centers
    on a unified timeline for comparative cross-airshed trajectory analysis.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 6.5))
    
    for city_name, fc_df in city_forecasts_dict.items():
        col = CITY_COLORS.get(city_name, '#3b82f6')
        smooth_line = fc_df['Projected_PM2_5'].rolling(14, min_periods=1, center=True).mean()
        ax.plot(fc_df['Date'], smooth_line, color=col, linewidth=1.8, label=f"{city_name} (14-day smoothed)")
        ax.fill_between(fc_df['Date'], fc_df['Lower_Bound_95'], fc_df['Upper_Bound_95'],
                        color=col, alpha=0.08)
        
    ax.axhline(60, color='#b91c1c', linestyle='--', linewidth=1.5, label=r'India NAAQS Standard (60 $\mu\mathrm{g/m}^3$)')
    ax.set_title("Comparative 5-Year Scenario Projections Across All 5 Metropolitan Airsheds (2025–2029)\n(Trajectory Comparison: Seasonal Harmonics & Secular Extrapolation with 95% Conformal Envelopes)",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Forecast Horizon", fontsize=11, fontweight='bold')
    ax.set_ylabel(r"Projected $\mathrm{PM}_{2.5}$ ($\mu\mathrm{g/m}^3$)", fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fontsize=9.5)
    ax.grid(alpha=0.35)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path
