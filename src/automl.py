"""
Automated Machine Learning (AutoML) Module.
Leverages FLAML BlendSearch and multi-algorithm competitive evaluation
to automatically select the best model family and optimal hyperparameter configuration.
"""
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flaml import AutoML
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class AirQualityAutoML:
    """
    AutoML pipeline tailored for air quality time series regression.
    Evaluates XGBoost, LightGBM, CatBoost, Random Forest, and Extra Trees
    under a unified time budget with automatic ensemble construction.
    """
    def __init__(self, time_budget=60, metric='r2', seed=42):
        self.time_budget = time_budget
        self.metric = metric
        self.seed = seed
        self.automl = None
        self.leaderboard_df = None
        self.best_estimator = None
        self.best_config = None

    def fit(self, X_train, y_train, X_val, y_val, estimator_list=None):
        """Runs the AutoML competitive search."""
        if estimator_list is None:
            estimator_list = ['xgboost', 'catboost', 'rf', 'extra_tree', 'lgbm']
            
        self.automl = AutoML()
        settings = {
            'time_budget': self.time_budget,
            'metric': self.metric,
            'estimator_list': estimator_list,
            'task': 'regression',
            'seed': self.seed,
            'verbose': 0
        }
        
        t0 = time.time()
        self.automl.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **settings)
        elapsed = time.time() - t0
        
        self.best_estimator = self.automl.best_estimator
        self.best_config = self.automl.best_config

        # Extract search history leaderboard if available
        trials_list = []
        if hasattr(self.automl, 'search_history'):
            for est, data in self.automl.search_history.items():
                loss_hist = data.get('val_loss', [])
                if len(loss_hist) > 0:
                    best_loss = min(loss_hist)
                    best_r2 = 1.0 - best_loss if self.metric == 'r2' else np.nan
                    trials_list.append({
                        'Estimator': est,
                        'Best_Val_Loss': best_loss,
                        'Best_Val_R2': best_r2,
                        'Total_Trials': len(loss_hist)
                    })
        
        if trials_list:
            self.leaderboard_df = pd.DataFrame(trials_list).sort_values('Best_Val_R2', ascending=False).reset_index(drop=True)
        else:
            self.leaderboard_df = pd.DataFrame([{
                'Estimator': self.best_estimator,
                'Best_Val_R2': 1.0 - self.automl.best_loss,
                'Best_Params': str(self.best_config)
            }])
            
        print(f"[AutoML] Finished in {elapsed:.1f}s. Winning Architecture: {self.best_estimator.upper()}")
        return self

    def predict(self, X):
        return self.automl.predict(X)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        return {
            'AutoML_Winning_Model': self.best_estimator,
            'Test_R2': r2_score(y_test, preds),
            'Test_RMSE': np.sqrt(mean_squared_error(y_test, preds)),
            'Test_MAE': mean_absolute_error(y_test, preds),
            'Predictions': preds
        }


def plot_automl_leaderboard(leaderboard_df, save_path='reports/figures/automl_model_competition_leaderboard.png'):
    """Renders competitive leaderboard bar chart across evaluated algorithms."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    estimators = leaderboard_df['Estimator'].str.upper()
    r2_scores = leaderboard_df['Best_Val_R2']

    bars = ax.barh(estimators[::-1], r2_scores[::-1], color='#0284c7', edgecolor='#0f172a', alpha=0.85, height=0.6)
    for bar in bars:
        w = bar.get_width()
        if not np.isnan(w):
            ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va='center', fontsize=9.5, fontweight='bold', color='#0f172a')

    ax.set_xlabel('Validation R2 Score (AutoML BlendSearch Optimization)', fontsize=11, fontweight='bold')
    ax.set_title('Automated Machine Learning (AutoML) Algorithm Competition Leaderboard', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlim(0.75, max(r2_scores) * 1.05)
    ax.grid(axis='x', linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_smbo_automl_comparison(comparison_df, save_path='reports/figures/smbo_automl_r2_comparison.png'):
    """
    Compares 3 tiers: Default Model vs. Sequential Model-Based (SMBO) Tuned vs. AutoML Champion.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(comparison_df))
    width = 0.28

    b1 = ax.bar(x - width, comparison_df['Default_R2'], width, label='Default Baseline', color='#94a3b8', edgecolor='#334155')
    b2 = ax.bar(x, comparison_df['SMBO_R2'], width, label='SMBO (Bayesian TPE) Tuned', color='#2563eb', edgecolor='#1e3a8a')
    b3 = ax.bar(x + width, comparison_df['AutoML_R2'], width, label='AutoML BlendSearch Winner', color='#10b981', edgecolor='#065f46')

    for bars, col in [(b1, '#475569'), (b2, '#1d4ed8'), (b3, '#047857')]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h) and h > 0:
                ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                            textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Model'], fontsize=11, fontweight='bold')
    ax.set_ylabel('Held-Out Test R2 Score', fontsize=11, fontweight='bold')
    ax.set_title('Three-Tier Model Optimization Comparison: Default vs SMBO vs AutoML', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(0.75, 0.95)
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
