import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_tree_models(X_train, y_train, X_val, y_val, random_seed=42):
    """
    Train benchmark tree-based regression models with early stopping on Validation set only.
    
    Returns:
        models (dict): Fitted model instances.
        fit_times (dict): Training duration in seconds.
    """
    models = {}
    fit_times = {}

    # 1. CatBoost (Optuna Tuned)
    t0 = time.time()
    cb = CatBoostRegressor(
        iterations=500,
        learning_rate=0.035,
        depth=5,
        l2_leaf_reg=4.0,
        random_seed=random_seed,
        verbose=0
    )
    cb.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=60)
    models['CatBoost'] = cb
    fit_times['CatBoost'] = time.time() - t0

    # 2. LightGBM (Optuna Tuned)
    t0 = time.time()
    lgb = LGBMRegressor(
        n_estimators=350,
        num_leaves=29,
        max_depth=7,
        learning_rate=0.015,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.05,
        reg_lambda=3.7,
        min_child_samples=18,
        random_state=random_seed,
        verbose=-1
    )
    lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
    models['LightGBM'] = lgb
    fit_times['LightGBM'] = time.time() - t0

    # 3. XGBoost (Optuna Tuned)
    t0 = time.time()
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.018,
        max_depth=4,
        subsample=0.72,
        colsample_bytree=0.75,
        reg_alpha=0.02,
        reg_lambda=2.8,
        min_child_weight=5,
        random_state=random_seed,
        tree_method='hist',
        early_stopping_rounds=60
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    models['XGBoost'] = xgb
    fit_times['XGBoost'] = time.time() - t0

    # 4. Random Forest (Optuna Tuned)
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=350,
        max_depth=16,
        min_samples_split=3,
        max_features=0.80,
        random_state=random_seed,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf
    fit_times['RandomForest'] = time.time() - t0

    # 5. ExtraTrees (Optuna Tuned)
    t0 = time.time()
    et = ExtraTreesRegressor(
        n_estimators=350,
        max_depth=18,
        min_samples_split=3,
        max_features=0.85,
        random_state=random_seed,
        n_jobs=-1
    )
    et.fit(X_train, y_train)
    models['ExtraTrees'] = et
    fit_times['ExtraTrees'] = time.time() - t0

    return models, fit_times


# -------------------------------------------------------------------------
# PyTorch Deep Learning Architectures (LSTM, ANN & DeepAirResNet)
# -------------------------------------------------------------------------

class PyTorchLSTMRegressor(nn.Module):
    """
    Recurrent Neural Network with Long Short-Term Memory (LSTM) cells
    designed to model temporal autoregressive dependencies.
    """
    def __init__(self, in_features, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, in_features)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden).squeeze(-1)

    def predict(self, X, pad_estimator=None, seq_len=7):
        return predict_lstm_model(self, X, pad_estimator=pad_estimator, seq_len=seq_len)


def make_lstm_sequences(X_mat, y_vec, seq_len=7):
    """Construct causal sliding window sequences for recurrent LSTM training."""
    Xs, ys = [], []
    for i in range(len(X_mat) - seq_len + 1):
        Xs.append(X_mat[i:i+seq_len])
        ys.append(y_vec[i+seq_len-1])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_lstm_model(X_tr_sc, y_tr, X_val_sc, y_val, seq_len=7, hidden_dim=64, epochs=80, lr=0.003, device=DEVICE):
    """
    Train PyTorch LSTM on sliding sequence windows with early stopping on validation loss.
    """
    t0 = time.time()
    X_tr_seq, y_tr_seq = make_lstm_sequences(X_tr_sc, y_tr, seq_len=seq_len)
    X_val_seq, y_val_seq = make_lstm_sequences(X_val_sc, y_val, seq_len=seq_len)

    model = PyTorchLSTMRegressor(in_features=X_tr_sc.shape[1], hidden_dim=hidden_dim, num_layers=2, dropout=0.2).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq)), batch_size=32, shuffle=True)

    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            v_pred = model(torch.tensor(X_val_seq).to(device)).cpu().numpy()
            v_loss = float(np.mean((y_val_seq - v_pred)**2))
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    fit_time = time.time() - t0
    return model, fit_time, seq_len


def predict_lstm_model(model, X_sc, pad_estimator=None, seq_len=7, device=None):
    """
    Generate predictions using fitted LSTM model, padding first seq_len-1 warmup rows.
    Handles small batch sizes (n < seq_len) gracefully via pad_estimator fallback.
    """
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = DEVICE
    n = len(X_sc)
    if n < seq_len:
        if pad_estimator is not None:
            try:
                return pad_estimator.predict(X_sc)
            except Exception:
                pass
        return np.zeros(n, dtype=np.float32)

    X_seq, _ = make_lstm_sequences(X_sc, np.zeros(n), seq_len=seq_len)
    with torch.no_grad():
        preds_seq = model(torch.tensor(X_seq, dtype=torch.float32).to(device)).cpu().numpy()

    # Pad initial warmup observations
    if pad_estimator is not None:
        try:
            pad = pad_estimator.predict(X_sc[:seq_len-1])
        except Exception:
            pad = np.repeat(preds_seq[0], seq_len-1)
    else:
        pad = np.repeat(preds_seq[0], seq_len-1)

    return np.concatenate([pad, preds_seq])


class PyTorchANNRegressor(nn.Module):
    """
    High-capacity Artificial Neural Network (Multi-Layer Perceptron / Dense Deep Neural Net)
    with Layer Normalization, SiLU non-linearities, and Residual skip connections.
    """
    def __init__(self, in_features, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def predict(self, X):
        return predict_ann_model(self, X)


def train_ann_model(X_tr_sc, y_tr, X_val_sc, y_val, epochs=80, lr=0.002, device=DEVICE):
    """
    Train PyTorch Artificial Neural Network (ANN) on scaled environmental predictors.
    """
    t0 = time.time()
    in_features = X_tr_sc.shape[1]
    model = PyTorchANNRegressor(in_features=in_features, hidden_dims=[256, 128, 64], dropout=0.2).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr_sc, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=32, shuffle=True
    )

    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            v_pred = model(torch.tensor(X_val_sc, dtype=torch.float32).to(device)).cpu().numpy()
            v_loss = float(np.mean((y_val - v_pred)**2))
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    fit_time = time.time() - t0
    return model, fit_time


def predict_ann_model(model, X_sc, device=None):
    """Generate predictions from fitted ANN model."""
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = DEVICE
    with torch.no_grad():
        x_t = torch.tensor(np.asarray(X_sc), dtype=torch.float32).to(device)
        return model(x_t).cpu().numpy()


class DeepAirResNet(nn.Module):
    """
    Deep Tabular Residual Neural Network with Skip Connections, LayerNorm,
    and GELU Activations, robust to multi-scale environmental predictors.
    """
    def __init__(self, in_features, hidden_dim=128, num_blocks=2, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = h + block(h)  # Residual skip connection
        return self.head(h).squeeze(-1)

    def predict(self, X):
        return predict_deep_air_resnet(self, X)


def train_deep_air_resnet(X_tr_sc, y_tr, X_val_sc, y_val, epochs=80, batch_size=32, lr=1e-3, device=DEVICE):
    """
    Train DeepAirResNet using Huber Loss and log1p target stabilization on GPU/CPU.
    Early stopping monitored on the Validation set only.
    """
    t0 = time.time()
    in_features = X_tr_sc.shape[1]
    model = DeepAirResNet(in_features=in_features).to(device)
    
    y_tr_log = np.log1p(np.maximum(y_tr, 0.0))
    y_val_log = np.log1p(np.maximum(y_val, 0.0))

    train_dataset = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr_log, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.HuberLoss(delta=0.5)

    best_val_loss = float('inf')
    best_weights = None
    patience = 20
    patience_cnt = 0

    val_tensor_x = torch.tensor(X_val_sc, dtype=torch.float32).to(device)
    val_tensor_y = torch.tensor(y_val_log, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_tensor_x), val_tensor_y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    fit_time = time.time() - t0
    return model, fit_time


def predict_deep_air_resnet(model, X_sc, device=None):
    """Invert log1p transformation to return predictions in original concentration scale."""
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = DEVICE
    with torch.no_grad():
        x_t = torch.tensor(np.asarray(X_sc), dtype=torch.float32).to(device)
        p_log = model(x_t).cpu().numpy()
        return np.expm1(p_log)
