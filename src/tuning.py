"""
Hyperparameter Tuning Module using Optuna Bayesian Optimization.
Tunes tree-based and gradient boosting architectures strictly on historical training/validation splits,
ensuring zero test leakage while maximizing predictive fidelity and R2 scores.
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

def tune_xgboost(X_train, y_train, X_val, y_val, n_trials=25, seed=42):
    """Bayesian hyperparameter optimization for XGBoost Regressor."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 600, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.012, 0.06, log=True),
            'subsample': trial.suggest_float('subsample', 0.70, 0.90),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.70, 0.90),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 2.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 2, 6),
            'random_state': seed,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
        m = xgb.XGBRegressor(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return r2_score(y_val, m.predict(X_val))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best_p = study.best_params
    best_p.update({'random_state': seed, 'n_jobs': -1, 'tree_method': 'hist'})
    model = xgb.XGBRegressor(**best_p).fit(X_train, y_train)
    return model, best_p, study.best_value


def tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=25, seed=42):
    """Bayesian hyperparameter optimization for LightGBM Regressor."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 600, step=50),
            'num_leaves': trial.suggest_int('num_leaves', 16, 48),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.012, 0.06, log=True),
            'subsample': trial.suggest_float('subsample', 0.70, 0.90),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.70, 0.90),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 2.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 12, 40),
            'random_state': seed,
            'n_jobs': -1,
            'verbose': -1
        }
        m = lgb.LGBMRegressor(**params)
        m.fit(X_train, y_train)
        return r2_score(y_val, m.predict(X_val))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best_p = study.best_params
    best_p.update({'random_state': seed, 'n_jobs': -1, 'verbose': -1})
    model = lgb.LGBMRegressor(**best_p).fit(X_train, y_train)
    return model, best_p, study.best_value


def tune_catboost(X_train, y_train, X_val, y_val, n_trials=20, seed=42):
    """Bayesian hyperparameter optimization for CatBoost Regressor."""
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 300, 700, step=50),
            'depth': trial.suggest_int('depth', 4, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.018, 0.06, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 2.0, 10.0),
            'random_seed': seed,
            'verbose': 0
        }
        m = CatBoostRegressor(**params)
        m.fit(X_train, y_train)
        return r2_score(y_val, m.predict(X_val))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best_p = study.best_params
    best_p.update({'random_seed': seed, 'verbose': 0})
    model = CatBoostRegressor(**best_p).fit(X_train, y_train)
    return model, best_p, study.best_value


def tune_random_forest(X_train, y_train, X_val, y_val, n_trials=20, seed=42):
    """Bayesian hyperparameter optimization for Random Forest Regressor."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 450, step=50),
            'max_depth': trial.suggest_int('max_depth', 12, 18),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 5),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 3),
            'max_features': trial.suggest_float('max_features', 0.65, 0.90),
            'random_state': seed,
            'n_jobs': -1
        }
        m = RandomForestRegressor(**params)
        m.fit(X_train, y_train)
        return r2_score(y_val, m.predict(X_val))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best_p = study.best_params
    best_p.update({'random_state': seed, 'n_jobs': -1})
    model = RandomForestRegressor(**best_p).fit(X_train, y_train)
    return model, best_p, study.best_value


def tune_extra_trees(X_train, y_train, X_val, y_val, n_trials=20, seed=42):
    """Bayesian hyperparameter optimization for Extra Trees Regressor."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 450, step=50),
            'max_depth': trial.suggest_int('max_depth', 14, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 5),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 3),
            'max_features': trial.suggest_float('max_features', 0.70, 0.95),
            'random_state': seed,
            'n_jobs': -1
        }
        m = ExtraTreesRegressor(**params)
        m.fit(X_train, y_train)
        return r2_score(y_val, m.predict(X_val))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best_p = study.best_params
    best_p.update({'random_state': seed, 'n_jobs': -1})
    model = ExtraTreesRegressor(**best_p).fit(X_train, y_train)
    return model, best_p, study.best_value


def run_full_hyperparameter_optimization(X_train, y_train, X_val, y_val, X_test, y_test, default_models=None):
    """
    Executes systematic Bayesian tuning across all model architectures,
    evaluating performance gains on both validation and held-out test splits.
    """
    print("=" * 80)
    print("SYSTEMATIC BAYESIAN HYPERPARAMETER OPTIMIZATION (OPTUNA TPE)")
    print("=" * 80)
    
    records = []
    tuned_models = {}

    tuners = [
        ('RandomForest', tune_random_forest, 20),
        ('ExtraTrees', tune_extra_trees, 20),
        ('XGBoost', tune_xgboost, 25),
        ('LightGBM', tune_lightgbm, 25),
        ('CatBoost', tune_catboost, 20)
    ]

    for name, tuner_fn, n_tr in tuners:
        t0 = time.time()
        print(f"\n[Optuna] Optimizing {name} across {n_tr} trials...")
        model, best_params, best_val_r2 = tuner_fn(X_train, y_train, X_val, y_val, n_trials=n_tr)
        elapsed = time.time() - t0
        
        # Test performance
        p_te = model.predict(X_test)
        te_r2 = r2_score(y_test, p_te)
        te_rmse = np.sqrt(mean_squared_error(y_test, p_te))
        te_mae = mean_absolute_error(y_test, p_te)
        
        tuned_models[name] = model

        # Benchmark against default if available
        def_val_r2 = r2_score(y_val, default_models[name].predict(X_val)) if default_models and name in default_models else np.nan
        def_te_r2 = r2_score(y_test, default_models[name].predict(X_test)) if default_models and name in default_models else np.nan
        
        records.append({
            'Model': name,
            'Default_Val_R2': def_val_r2,
            'Tuned_Val_R2': best_val_r2,
            'Val_R2_Gain': best_val_r2 - def_val_r2 if not np.isnan(def_val_r2) else np.nan,
            'Default_Test_R2': def_te_r2,
            'Tuned_Test_R2': te_r2,
            'Test_R2_Gain': te_r2 - def_te_r2 if not np.isnan(def_te_r2) else np.nan,
            'Tuned_Test_RMSE': te_rmse,
            'Tuned_Test_MAE': te_mae,
            'Tuning_Time_s': round(elapsed, 1),
            'Best_Hyperparameters': str(best_params)
        })
        print(f"  -> Best {name} Val R2: {best_val_r2:.4f} | Test R2: {te_r2:.4f} (RMSE: {te_rmse:.2f} ug/m3, Time: {elapsed:.1f}s)")

    summary_df = pd.DataFrame(records)
    return tuned_models, summary_df


def plot_tuning_comparison(summary_df, save_path='reports/figures/hyperparameter_tuning_comparison.png'):
    """
    Renders high-DPI grouped bar chart comparing default vs. tuned R2 performance.
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(summary_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, summary_df['Default_Test_R2'], width, label='Default Parameters',
                   color='#94a3b8', edgecolor='#1e293b', linewidth=0.8, alpha=0.9)
    bars2 = ax.bar(x + width/2, summary_df['Tuned_Test_R2'], width, label='Optuna Tuned Parameters',
                   color='#2563eb', edgecolor='#1e293b', linewidth=0.8, alpha=0.9)

    for bar in bars1:
        h = bar.get_height()
        if not np.isnan(h):
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                        textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#475569')

    for bar in bars2:
        h = bar.get_height()
        if not np.isnan(h):
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                        textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1d4ed8')

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['Model'], fontsize=11, fontweight='bold')
    ax.set_ylabel('Test R2 Score (Higher is Better)', fontsize=11, fontweight='bold')
    ax.set_title('Impact of Bayesian Hyperparameter Optimization on Out-of-Sample Test R2', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(0.70, 0.95)
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
