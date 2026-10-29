"""
Sequential Model-Based Optimization (SMBO) Engine.
Implements Bayesian Optimization using surrogate regression and Expected Improvement (EI)
acquisition function for atmospheric time-series forecasting models.
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

class SMBOSurrogateOptimizer:
    """
    Sequential Model-Based Optimization (SMBO) with Gaussian Process / Random Forest surrogate
    and Expected Improvement (EI) acquisition function.
    """
    def __init__(self, param_space, objective_fn, n_initial=5, n_iterations=20, xi=0.01, random_state=42):
        """
        Args:
            param_space (dict): Parameter names and bounds {'name': (min_val, max_val, is_int)}.
            objective_fn (callable): Black-box function returning validation score (higher is better).
            n_initial (int): Number of initial quasi-random exploration trials.
            n_iterations (int): Sequential Bayesian optimization steps.
            xi (float): Exploration-exploitation trade-off parameter.
        """
        self.param_space = param_space
        self.objective_fn = objective_fn
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.xi = xi
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        self.trials = []
        self.best_params = None
        self.best_score = -float('inf')
        self.surrogate = None

    def _sample_random_point(self):
        point = {}
        for name, (low, high, is_int) in self.param_space.items():
            if is_int:
                point[name] = int(self.rng.randint(low, high + 1))
            else:
                point[name] = float(self.rng.uniform(low, high))
        return point

    def _params_to_vector(self, params):
        return np.array([params[name] for name in self.param_space.keys()], dtype=float)

    def _vector_to_params(self, vec):
        params = {}
        for i, (name, (low, high, is_int)) in enumerate(self.param_space.items()):
            val = np.clip(vec[i], low, high)
            params[name] = int(round(val)) if is_int else float(val)
        return params

    def _expected_improvement(self, X_candidates, y_best):
        mu, sigma = self.surrogate.predict(X_candidates, return_std=True)
        sigma = np.maximum(sigma, 1e-8)
        improvement = mu - y_best - self.xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei, mu, sigma

    def optimize(self):
        """Executes the sequential SMBO loop."""
        # 1. Initial Exploration Phase
        for i in range(self.n_initial):
            p = self._sample_random_point()
            score = self.objective_fn(p)
            self.trials.append({
                'iteration': i + 1,
                'phase': 'Initial_Design',
                'params': p,
                'score': score,
                'surrogate_mu': np.nan,
                'surrogate_sigma': np.nan,
                'acquisition_ei': np.nan
            })
            if score > self.best_score:
                self.best_score = score
                self.best_params = p

        # 2. Sequential Surrogate Optimization Phase
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)

        for step in range(self.n_iterations):
            iter_num = self.n_initial + step + 1
            X_hist = np.array([self._params_to_vector(t['params']) for t in self.trials])
            y_hist = np.array([t['score'] for t in self.trials])

            # Fit surrogate model
            self.surrogate = GaussianProcessRegressor(
                kernel=kernel, alpha=1e-4, normalize_y=True,
                n_restarts_optimizer=2, random_state=self.random_state
            )
            self.surrogate.fit(X_hist, y_hist)

            # Generate candidate pool for acquisition evaluation (Monte Carlo candidates)
            n_candidates = 250
            cand_points = [self._sample_random_point() for _ in range(n_candidates)]
            X_cand = np.array([self._params_to_vector(p) for p in cand_points])

            ei_vals, mus, sigmas = self._expected_improvement(X_cand, self.best_score)
            best_cand_idx = np.argmax(ei_vals)

            next_p = cand_points[best_cand_idx]
            next_score = self.objective_fn(next_p)

            self.trials.append({
                'iteration': iter_num,
                'phase': 'SMBO_Sequential',
                'params': next_p,
                'score': next_score,
                'surrogate_mu': float(mus[best_cand_idx]),
                'surrogate_sigma': float(sigmas[best_cand_idx]),
                'acquisition_ei': float(ei_vals[best_cand_idx])
            })

            if next_score > self.best_score:
                self.best_score = next_score
                self.best_params = next_p

        # Compile historical dataframe
        df_history = pd.DataFrame(self.trials)
        df_history['incumbent_best'] = df_history['score'].cummax()
        return self.best_params, self.best_score, df_history


def plot_smbo_trajectory(history_df, model_name="XGBoost", save_path="reports/figures/smbo_optimization_trajectory.png"):
    """
    Renders publication-grade visualization of SMBO search trajectory:
    Panel 1: Objective R2 score trajectory with incumbent best boundary.
    Panel 2: Expected Improvement acquisition value decay across sequential iterations.
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={'height_ratios': [1.3, 1]})

    iters = history_df['iteration']
    scores = history_df['score']
    incumbent = history_df['incumbent_best']

    # Panel 1: Convergence trajectory
    ax1.scatter(iters, scores, color='#3b82f6', s=45, alpha=0.75, edgecolors='black', label='Evaluated Trials f(x)')
    ax1.step(iters, incumbent, where='post', color='#dc2626', linewidth=2.0, label='Incumbent Best R2')
    ax1.set_ylabel('Validation R2 Score', fontsize=11, fontweight='bold')
    ax1.set_title(f'Sequential Model-Based Optimization (SMBO) Trajectory - {model_name}', fontsize=12.5, fontweight='bold', pad=10)
    ax1.legend(loc='lower right', frameon=True, fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Panel 2: Expected Improvement acquisition
    smbo_mask = history_df['phase'] == 'SMBO_Sequential'
    smbo_iters = iters[smbo_mask]
    eis = history_df.loc[smbo_mask, 'acquisition_ei']

    ax2.plot(smbo_iters, eis, marker='o', color='#0d9488', linewidth=1.5, markersize=5, label='Expected Improvement EI(x)')
    ax2.fill_between(smbo_iters, 0, eis, color='#0d9488', alpha=0.2)
    ax2.set_xlabel('Sequential SMBO Iteration', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Acquisition Value EI(x)', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def run_smbo_study_for_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Runs SMBO optimization for each core model family and evaluates held-out test performance.
    """
    records = []
    trained_models = {}
    trajectories = {}

    # 1. XGBoost Param Space
    xgb_space = {
        'n_estimators': (200, 500, True),
        'max_depth': (3, 6, True),
        'learning_rate': (0.015, 0.05, False),
        'subsample': (0.70, 0.90, False),
        'colsample_bytree': (0.70, 0.90, False),
        'reg_alpha': (0.01, 1.0, False),
        'reg_lambda': (1.0, 5.0, False)
    }
    def obj_xgb(p):
        m = xgb.XGBRegressor(**p, random_state=42, n_jobs=-1, tree_method='hist')
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return float(r2_score(y_val, m.predict(X_val)))

    print("[SMBO] Running sequential optimization for XGBoost...")
    opt_xgb = SMBOSurrogateOptimizer(xgb_space, obj_xgb, n_initial=5, n_iterations=15)
    best_p_xgb, best_v_xgb, hist_xgb = opt_xgb.optimize()
    m_xgb = xgb.XGBRegressor(**best_p_xgb, random_state=42, n_jobs=-1, tree_method='hist').fit(X_train, y_train)
    trained_models['XGBoost'] = m_xgb
    trajectories['XGBoost'] = hist_xgb

    p_te_xgb = m_xgb.predict(X_test)
    records.append({
        'Model': 'XGBoost',
        'SMBO_Val_R2': best_v_xgb,
        'SMBO_Test_R2': r2_score(y_test, p_te_xgb),
        'SMBO_Test_RMSE': np.sqrt(mean_squared_error(y_test, p_te_xgb)),
        'SMBO_Test_MAE': mean_absolute_error(y_test, p_te_xgb),
        'Best_Parameters': str(best_p_xgb)
    })

    # 2. Random Forest Param Space
    rf_space = {
        'n_estimators': (200, 400, True),
        'max_depth': (12, 18, True),
        'min_samples_split': (2, 5, True),
        'max_features': (0.65, 0.85, False)
    }
    def obj_rf(p):
        m = RandomForestRegressor(
            n_estimators=int(p['n_estimators']),
            max_depth=int(p['max_depth']),
            min_samples_split=int(p['min_samples_split']),
            max_features=float(p['max_features']),
            random_state=42, n_jobs=-1
        )
        m.fit(X_train, y_train)
        return float(r2_score(y_val, m.predict(X_val)))

    print("[SMBO] Running sequential optimization for Random Forest...")
    opt_rf = SMBOSurrogateOptimizer(rf_space, obj_rf, n_initial=4, n_iterations=10)
    best_p_rf, best_v_rf, hist_rf = opt_rf.optimize()
    m_rf = RandomForestRegressor(
        n_estimators=int(best_p_rf['n_estimators']),
        max_depth=int(best_p_rf['max_depth']),
        min_samples_split=int(best_p_rf['min_samples_split']),
        max_features=float(best_p_rf['max_features']),
        random_state=42, n_jobs=-1
    ).fit(X_train, y_train)
    trained_models['RandomForest'] = m_rf
    trajectories['RandomForest'] = hist_rf

    p_te_rf = m_rf.predict(X_test)
    records.append({
        'Model': 'RandomForest',
        'SMBO_Val_R2': best_v_rf,
        'SMBO_Test_R2': r2_score(y_test, p_te_rf),
        'SMBO_Test_RMSE': np.sqrt(mean_squared_error(y_test, p_te_rf)),
        'SMBO_Test_MAE': mean_absolute_error(y_test, p_te_rf),
        'Best_Parameters': str(best_p_rf)
    })

    # 3. LightGBM Param Space
    lgb_space = {
        'n_estimators': (200, 450, True),
        'num_leaves': (20, 45, True),
        'max_depth': (5, 9, True),
        'learning_rate': (0.015, 0.05, False),
        'subsample': (0.70, 0.90, False),
        'colsample_bytree': (0.70, 0.90, False),
        'reg_alpha': (0.01, 1.0, False),
        'reg_lambda': (1.0, 5.0, False)
    }
    def obj_lgb(p):
        m = lgb.LGBMRegressor(
            n_estimators=int(p['n_estimators']),
            num_leaves=int(p['num_leaves']),
            max_depth=int(p['max_depth']),
            learning_rate=float(p['learning_rate']),
            subsample=float(p['subsample']),
            colsample_bytree=float(p['colsample_bytree']),
            reg_alpha=float(p['reg_alpha']),
            reg_lambda=float(p['reg_lambda']),
            subsample_freq=1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        m.fit(X_train, y_train)
        return float(r2_score(y_val, m.predict(X_val)))

    print("[SMBO] Running sequential optimization for LightGBM...")
    opt_lgb = SMBOSurrogateOptimizer(lgb_space, obj_lgb, n_initial=4, n_iterations=10)
    best_p_lgb, best_v_lgb, hist_lgb = opt_lgb.optimize()
    m_lgb = lgb.LGBMRegressor(
        n_estimators=int(best_p_lgb['n_estimators']),
        num_leaves=int(best_p_lgb['num_leaves']),
        max_depth=int(best_p_lgb['max_depth']),
        learning_rate=float(best_p_lgb['learning_rate']),
        subsample=float(best_p_lgb['subsample']),
        colsample_bytree=float(best_p_lgb['colsample_bytree']),
        reg_alpha=float(best_p_lgb['reg_alpha']),
        reg_lambda=float(best_p_lgb['reg_lambda']),
        subsample_freq=1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ).fit(X_train, y_train)
    trained_models['LightGBM'] = m_lgb
    trajectories['LightGBM'] = hist_lgb

    p_te_lgb = m_lgb.predict(X_test)
    records.append({
        'Model': 'LightGBM',
        'SMBO_Val_R2': best_v_lgb,
        'SMBO_Test_R2': r2_score(y_test, p_te_lgb),
        'SMBO_Test_RMSE': np.sqrt(mean_squared_error(y_test, p_te_lgb)),
        'SMBO_Test_MAE': mean_absolute_error(y_test, p_te_lgb),
        'Best_Parameters': str(best_p_lgb)
    })

    # 4. Extra Trees Param Space
    et_space = {
        'n_estimators': (200, 400, True),
        'max_depth': (14, 22, True),
        'min_samples_split': (2, 5, True),
        'max_features': (0.70, 0.90, False)
    }
    def obj_et(p):
        m = ExtraTreesRegressor(
            n_estimators=int(p['n_estimators']),
            max_depth=int(p['max_depth']),
            min_samples_split=int(p['min_samples_split']),
            max_features=float(p['max_features']),
            random_state=42, n_jobs=-1
        )
        m.fit(X_train, y_train)
        return float(r2_score(y_val, m.predict(X_val)))

    print("[SMBO] Running sequential optimization for Extra Trees...")
    opt_et = SMBOSurrogateOptimizer(et_space, obj_et, n_initial=4, n_iterations=10)
    best_p_et, best_v_et, hist_et = opt_et.optimize()
    m_et = ExtraTreesRegressor(
        n_estimators=int(best_p_et['n_estimators']),
        max_depth=int(best_p_et['max_depth']),
        min_samples_split=int(best_p_et['min_samples_split']),
        max_features=float(best_p_et['max_features']),
        random_state=42, n_jobs=-1
    ).fit(X_train, y_train)
    trained_models['ExtraTrees'] = m_et
    trajectories['ExtraTrees'] = hist_et

    p_te_et = m_et.predict(X_test)
    records.append({
        'Model': 'ExtraTrees',
        'SMBO_Val_R2': best_v_et,
        'SMBO_Test_R2': r2_score(y_test, p_te_et),
        'SMBO_Test_RMSE': np.sqrt(mean_squared_error(y_test, p_te_et)),
        'SMBO_Test_MAE': mean_absolute_error(y_test, p_te_et),
        'Best_Parameters': str(best_p_et)
    })

    # 5. CatBoost Param Space
    cb_space = {
        'iterations': (250, 500, True),
        'depth': (4, 7, True),
        'learning_rate': (0.02, 0.06, False),
        'l2_leaf_reg': (2.0, 6.0, False)
    }
    def obj_cb(p):
        m = CatBoostRegressor(
            iterations=int(p['iterations']),
            depth=int(p['depth']),
            learning_rate=float(p['learning_rate']),
            l2_leaf_reg=float(p['l2_leaf_reg']),
            random_seed=42,
            verbose=False
        )
        m.fit(X_train, y_train)
        return float(r2_score(y_val, m.predict(X_val)))

    print("[SMBO] Running sequential optimization for CatBoost...")
    opt_cb = SMBOSurrogateOptimizer(cb_space, obj_cb, n_initial=4, n_iterations=10)
    best_p_cb, best_v_cb, hist_cb = opt_cb.optimize()
    m_cb = CatBoostRegressor(
        iterations=int(best_p_cb['iterations']),
        depth=int(best_p_cb['depth']),
        learning_rate=float(best_p_cb['learning_rate']),
        l2_leaf_reg=float(best_p_cb['l2_leaf_reg']),
        random_seed=42,
        verbose=False
    ).fit(X_train, y_train)
    trained_models['CatBoost'] = m_cb
    trajectories['CatBoost'] = hist_cb

    p_te_cb = m_cb.predict(X_test)
    records.append({
        'Model': 'CatBoost',
        'SMBO_Val_R2': best_v_cb,
        'SMBO_Test_R2': r2_score(y_test, p_te_cb),
        'SMBO_Test_RMSE': np.sqrt(mean_squared_error(y_test, p_te_cb)),
        'SMBO_Test_MAE': mean_absolute_error(y_test, p_te_cb),
        'Best_Parameters': str(best_p_cb)
    })

    summary_df = pd.DataFrame(records)
    return trained_models, summary_df, trajectories
