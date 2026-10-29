<<<<<<< HEAD
# Multi-Paradigm Ambient Air Quality Forecasting & Telemetric Nowcasting System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: Passing](https://img.shields.io/badge/tests-10%2F10%20passing-brightgreen.svg)](tests/test_pipeline.py)
[![Data: CPCB India](https://img.shields.io/badge/data-CPCB%20CAAQMS-orange.svg)](https://cpcb.nic.in/)

Research-grade, leakage-free ambient air quality modeling, 24-hour causal forecasting, and multi-pollutant telemetric nowcasting system developed in collaboration with **CSIR - Central Building Research Institute (CBRI), Roorkee**.

---

## 👨‍🔬 Author & Project Metadata

* **Author:** **Aman Kumar Ranjan**
* **Roll Number:** **23010301**
* **Degree Program:** **B.Tech in Computer Science and Engineering (Data Science)**
* **Department:** Department of Computer Science & Engineering
* **Project Partner:** **CSIR - Central Building Research Institute (CBRI), Roorkee**
* **Data Authority:** Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change, Govt. of India
* **Target Publication:** *Elsevier / ScienceDirect* (*Environmental Modelling & Software* / *Atmospheric Environment*)

---

## 📌 Executive Overview

Atmospheric particulate matter ($PM_{2.5}$) and surface criteria pollutants present severe public health hazards across Indian urban centers. In northern India (e.g., Delhi NCR), seasonal meteorology traps emissions beneath shallow atmospheric boundary layers, causing particulate concentrations to exceed $400\ \mu\text{g/m}^3$—nearly an order of magnitude above National Ambient Air Quality Standards (NAAQS).

This repository contains an end-to-end, reproducible, and mathematically audited framework evaluating **54 model configurations** across **five distinct Indian megacities** representing diverse climatic and geographical zones, alongside a unified **Global Multi-City System**.

```
+-------------------------------------------------------------------------------------------------+
|                                    CSIR-CBRI MODELING SYSTEM                                    |
+-------------------------------------------------------------------------------------------------+
                                                 |
             +-----------------------------------+-----------------------------------+
             |                                                                       |
    [PARADIGM 1: CAUSAL FORECASTING]                                [PARADIGM 2: TELEMETRIC NOWCASTING]
    - Predicts t+1 strictly from t-1                                 - Estimates PM2.5 from co-located
    - True early warning (24-72 hrs)                                   sensor arrays at time t
    - R²: 0.7224 - 0.8218                                            - Virtual sensing / Sensor recovery
    - Zero lookahead leakage                                         - R²: 0.9039 - 0.9912 (Global: 0.9729)
             |                                                                       |
             +-----------------------------------+-----------------------------------+
                                                 |
                                                 v
                       +---------------------------------------------------+
                       |              MULTI-PARADIGM MODEL SUITE           |
                       |  1. State-Space SARIMAX (Kalman Filter)           |
                       |  2. L2-Regularized Ridge Regression               |
                       |  3. CatBoost (Oblivious Trees, SMBO Tuned)        |
                       |  4. LightGBM (GOSS Leaf-Wise Trees)               |
                       |  5. XGBoost (Exact Greedy Split Regularized)      |
                       |  6. Deep PyTorch LSTM (LayerNorm + Huber Loss)    |
                       |  7. Deep PyTorch GRU (LayerNorm + Huber Loss)     |
                       |  8. SARIMAX–CatBoost Error Correction Hybrid      |
                       |  9. SLSQP Constrained SuperBlend Stacker          |
                       +---------------------------------------------------+
                                                 |
                                                 v
                       +---------------------------------------------------+
                       |             DECISION SUPPORT & DEPLOYMENT         |
                       |  - Split-Conformal Prediction (95.8% coverage)    |
                       |  - 5-Year Forward Multi-Scenario Policy Engine    |
                       |  - FastAPI REST API (api.py)                      |
                       |  - Streamlit Interactive Web Dashboard (app.py)   |
                       +---------------------------------------------------+
```

---

## 🔬 Dual-Paradigm Atmospheric Science Formulation

To resolve methodological ambiguities common in air quality machine learning literature, this project explicitly formalizes two distinct analytical paradigms:

### Paradigm 1: 24-Hour Predictive Causal Forecasting (Early-Warning Standard)
Designed for public health alerts, school advisories, and vehicular curfews. The feature representation $X_{\text{causal}}(t)$ uses **strictly lagged historical information** available at or before day $t-1$:
$$X_{\text{causal}}(t) = \left\{ y_{t-1}, y_{t-2}, \dots, y_{t-k}, P_{i, t-1}, \text{Roll}_w(y)_{t-1}, \text{Diff}_d(y)_{t-1}, \text{Harmonics}(t) \right\}$$
*All contemporaneous measurements at day $t$ are excluded.* This yields authentic out-of-sample test scores between **$R^2 = 0.7224$ and $0.8218$**.

### Paradigm 2: Multi-Pollutant Telemetric Nowcasting (Virtual Sensor Array)
Designed for virtual sensing, sensor failure imputation, and compliance backup when specialized optical or beta-attenuation $PM_{2.5}$ monitors go offline for maintenance:
$$X_{\text{nowcast}}(t) = \left\{ PM_{10}(t), NO_2(t), SO_2(t), CO(t), O_3(t), \text{VOCs}(t), \text{Lags}_{t-1}, \text{Ratios}_{t-1}, \text{Harmonics}(t) \right\}$$
Exploiting chemical equilibrium (e.g. secondary inorganic ammonium nitrate/sulfate aerosol pathways) enables virtual sensing with **$R^2 = 0.9039\text{--}0.9912$** (Global: **$0.9729$**).

---

## 🏙️ Geographical Diversity & Study Zones

| Urban Center | Geographic Zone | Köppen Climate | Elevation & Terrain | Dominant Primary Sources |
| :--- | :--- | :--- | :--- | :--- |
| **Delhi (NCR)** | Indo-Gangetic Plain | Cwa (Humid Subtropical) | 216 m ASL; Flat alluvial basin | Vehicular exhaust, crop stubble burning, industrial stacks, dust |
| **Kolkata** | Lower Gangetic Delta | Aw (Tropical Wet/Dry) | 9 m ASL; Low coastal delta | Thermal power, brick kilns, cross-border transit, diesel fleets |
| **Mumbai** | Konkan Coast (West) | Am (Tropical Monsoon) | 14 m ASL; Coastal peninsula | Maritime shipping, petrochemical refineries, dense traffic corridors |
| **Bengaluru** | Deccan Plateau (South) | Aw (Tropical Savanna) | 920 m ASL; Elevated plateau | Heavy vehicular traffic, construction dust, urban expansion |
| **Visakhapatnam** | Coromandel Coast (East)| Aw (Tropical Wet/Dry) | 45 m ASL; Industrial coastal harbor | Major seaport, integrated steel plants, petrochemical complexes |

> **Note on Bengaluru Instrumentation Transition:** Prior to 2018, Bengaluru CAAQMS stations logged non-functional constant values for $PM_{10}$ ($70.46\ \mu\text{g/m}^3$). Continuous multi-pollutant analyzers were commissioned in 2018. Restricting Bengaluru analysis to its active calibrated window (`Date >= 2018-01-01`) restored physical coherence and elevated performance from $0.59$ to **$0.9832$**.

---

## 📊 Empirical Benchmarks & Leaderboards

### 1. Champion Model Performance by City & Global (Telemetric Nowcasting)
Every urban center and the unified Global model achieve Test $R^2 > \mathbf{0.90}$ with minimal generalization gaps ($|\Delta R^2| \le 0.03$):

| Region / City | Champion Architecture | Train $R^2$ | Val $R^2$ | **Test $R^2$** | Gap ($\Delta R^2$) | Test RMSE | Test MAE | Test sMAPE | FAC2 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Global (All 5 Cities)** | **SuperBlend Stacker** | 0.9721 | 0.9755 | **0.9729** | **$-0.0008$** | $9.03\ \mu\text{g/m}^3$ | $5.63\ \mu\text{g/m}^3$ | 13.47% | 99.81% |
| **Visakhapatnam** | **SuperBlend Stacker** | 0.9972 | 0.9763 | **0.9912** | **$+0.0059$** | $2.08\ \mu\text{g/m}^3$ | $1.36\ \mu\text{g/m}^3$ | 5.26% | 100.00% |
| **Bengaluru** | **Deep LSTM Regressor** | 0.9951 | 0.9904 | **0.9832** | **$+0.0120$** | $1.73\ \mu\text{g/m}^3$ | $1.17\ \mu\text{g/m}^3$ | 4.08% | 100.00% |
| **Kolkata** | **State-Space SARIMAX** | 0.9880 | 0.9368 | **0.9755** | **$+0.0124$** | $2.90\ \mu\text{g/m}^3$ | $2.21\ \mu\text{g/m}^3$ | 10.57% | 100.00% |
| **Delhi (NCR)** | **SARIMAX–CatBoost Hybrid**| 0.9789 | 0.9750 | **0.9678** | **$+0.0111$** | $13.77\ \mu\text{g/m}^3$| $9.51\ \mu\text{g/m}^3$ | 10.99% | 100.00% |
| **Mumbai** | **State-Space SARIMAX** | 0.9333 | 0.8425 | **0.9039** | **$+0.0294$** | $3.35\ \mu\text{g/m}^3$ | $2.14\ \mu\text{g/m}^3$ | 10.58% | 100.00% |
| *Mumbai (Global Transferred)* | *SuperBlend Stacker* | 0.9721 | 0.9755 | **0.9059** | *N/A* | $3.32\ \mu\text{g/m}^3$ | $2.37\ \mu\text{g/m}^3$ | 11.20% | 100.00% |

---

### 2. Dual-Paradigm Methodological Ablation
Systematic ablation demonstrating the exact transition from telemetric nowcasting to strictly causal forecasting:

| City | A: As-Shipped (Nowcasting) | B: Ratio Ablated (Excl. $PM_{2.5}/PM_{10}$) | C: Ratio & AQI Ablated | **D: Strictly Causal ($t-1$ Lags Only)** |
| :--- | :---: | :---: | :---: | :---: |
| **Delhi (NCR)** | 0.9761 | 0.9580 | 0.9582 | **0.8218** |
| **Kolkata** | 0.9245 | 0.9737 | 0.9559 | **0.7766** |
| **Visakhapatnam** | 0.9647 | 0.9204 | 0.9186 | **0.7493** |
| **Mumbai** | 0.9312 | 0.8457 | 0.8714 | **0.7236** |
| **Bengaluru** | 0.9076 | 0.8518 | 0.8435 | **0.7224** |

---

### 3. Spatial Cross-City Transferability of the Global Model
Zero-shot evaluation of the unified Global SuperBlend Stacker across individual held-out test splits:

| Evaluation City | Zero-Shot Test $R^2$ | Zero-Shot Test RMSE | Zero-Shot Test MAE |
| :--- | :---: | :---: | :---: |
| **Delhi (NCR)** | **0.9648** | $14.39\ \mu\text{g/m}^3$ | $9.82\ \mu\text{g/m}^3$ |
| **Kolkata** | **0.9442** | $4.37\ \mu\text{g/m}^3$ | $3.12\ \mu\text{g/m}^3$ |
| **Visakhapatnam** | **0.9162** | $6.43\ \mu\text{g/m}^3$ | $4.85\ \mu\text{g/m}^3$ |
| **Mumbai** | **0.9059** | $3.32\ \mu\text{g/m}^3$ | $2.37\ \mu\text{g/m}^3$ |
| **Bengaluru** | **0.7941** | $6.05\ \mu\text{g/m}^3$ | $4.27\ \mu\text{g/m}^3$ |

---

## ⚙️ Hyperparameter Tuning (Optuna SMBO)

Hyperparameters were optimized via Optuna Sequential Model-Based Optimization (SMBO) using chronological time-series cross-validation:

```python
# Optimal SMBO Parameter Profiles
CATBOOST_DELHI = {
    'iterations': 250, 'depth': 6, 'learning_rate': 0.0634,
    'l2_leaf_reg': 4.71, 'subsample': 0.7045
}
LIGHTGBM_DELHI = {
    'n_estimators': 350, 'max_depth': 4, 'num_leaves': 7,
    'learning_rate': 0.0428, 'min_child_samples': 20,
    'reg_alpha': 0.62, 'reg_lambda': 26.12
}
XGBOOST_DELHI = {
    'n_estimators': 350, 'max_depth': 3, 'learning_rate': 0.0674,
    'min_child_weight': 21, 'reg_alpha': 12.21, 'reg_lambda': 22.56
}
RIDGE_DELHI = {'alpha': 7.017}
DEEP_RECURRENT_PARAMS = {
    'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.20,
    'lr': 1e-3, 'weight_decay': 1e-4, 'seq_len': 14, 'loss': 'Huber'
}
```

---

## 🛡️ Conformal Prediction & 5-Year Forward Scenarios

* **Split-Conformal Prediction Coverage:** Computes finite-sample non-conformity scores on validation partitions, establishing distribution-free prediction bounds $[\hat{y} - q_{\text{val}}, \hat{y} + q_{\text{val}}]$ that achieve **$95.8\%$ empirical coverage** on held-out test splits during severe pollution episodes.
* **5-Year Forward Projections (2022–2026):**
  1. **Business-As-Usual (BAU):** Continuation of historical secular drift and Fourier seasonal harmonics.
  2. **Strict NCAP Enforcement ($-25\%$):** Models aggressive vehicular fleet electrification, industrial transition, and stubble burning bans.
  3. **Accelerated Urban Growth ($+15\%$):** Simulates expanded industrialization and construction dust.

---

## 🗂️ Repository Structure

```
csir-cbri-air-quality/
├── data/                                # Curated CPCB atmospheric telemetry
│   ├── cpcb_5_cities_india.csv          # Primary 5-city daily continuous dataset
│   ├── cpcb_all_26_cities_india.csv     # Extended multi-city registry
│   └── cpcb_cities/                     # Station-level raw telemetry subsets
├── src/                                 # Production-grade modular source packages
│   ├── hybrid_forecasting.py            # SARIMAX, CatBoost Hybrid, LSTM/GRU, SuperBlend
│   ├── multi_city.py                    # Multi-city data loaders and city cleaners
│   ├── feature_engineering.py           # Lags, EWMA, rolling stats, Fourier harmonics
│   ├── conformal.py                     # Split conformal prediction and coverage metrics
│   ├── smbo.py                          # Optuna SMBO Bayesian optimization routines
│   ├── automl.py                        # FLAML BlendSearch automated selection
│   ├── inference.py                     # Standalone production inference engine
│   └── train_pipeline.py                # End-to-end training and serialization pipeline
├── models/                              # Trained model artifacts and weights
│   ├── saved_models/                    # Single-city baseline serialized models
│   └── saved_models_v2/                 # Upgraded multi-city champion artifacts
├── tests/                               # Comprehensive unit and integration test suite
│   └── test_pipeline.py                 # 10/10 automated tests (leakage, bounds, splits)
├── airQuality_cbri_pipeline.ipynb       # Executed master research notebook (40 cells)
├── airQuality_cbri_pipeline_upgraded.ipynb # Upgraded standalone copy notebook
├── api.py                               # High-performance FastAPI REST API service
├── app.py                               # Interactive Streamlit Web Dashboard
├── config.yaml                          # Central configuration specification
├── requirements.txt                     # Pinned project dependencies
├── LICENSE                              # MIT Open-Source License
├── CITATION.cff                         # Academic citation metadata
└── README.md                            # Comprehensive project documentation
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/csir-cbri-air-quality.git
cd csir-cbri-air-quality
```

### 2. Set Up Virtual Environment
```bash
# Using conda
conda create -n airquality python=3.10 -y
conda activate airquality

# OR using venv
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Unit Test Suite
Verify that all 10 unit tests pass (zero data leakage, disjoint splits, conformal coverage):
```bash
pytest tests/test_pipeline.py -v
```
*Expected Result:* `10 passed in ~12s`.

---

## 💻 Usage & Deployment

### 1. Running the Interactive Research Notebooks
Open either notebook in Jupyter or VS Code:
```bash
jupyter notebook airQuality_cbri_pipeline.ipynb
```

### 2. Launching the FastAPI REST Service
The REST API provides high-throughput inference endpoints:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
* Interactive Swagger Docs: `http://localhost:8000/docs`
* Health Check: `GET /health`
* City Inference: `POST /predict/city`
* Conformal Bound: `POST /predict/conformal`

### 3. Launching the Streamlit Web Dashboard
Launch the interactive dashboard for visual exploration and scenario analysis:
```bash
streamlit run app.py
```
* Access dashboard at `http://localhost:8501`.

---

## 📤 Step-by-Step GitHub Upload Guide

To upload this repository to your GitHub account:

```bash
# 1. Initialize git in the project root
git init

# 2. Add all files (respects .gitignore: reports and caches are excluded)
git add .

# 3. Check status to confirm reports/ is excluded
git status

# 4. Create initial commit
git commit -m "Initial commit: CSIR-CBRI Multi-City Air Quality Forecasting System"

# 5. Set default branch to main
git branch -M main

# 6. Add your remote repository URL (create an empty repo on github.com first)
git remote add origin https://github.com/<your-username>/csir-cbri-air-quality.git

# 7. Push to GitHub
git push -u origin main
```

---

## 🏛️ Environmental Policy & Building Engineering Applications

1. **Dynamic Smart Building Ventilation (CSIR-CBRI):** Building Management Systems (BMS) can modulate dual-stage HEPA / electrostatic filtration units 24 hours in advance of anticipated inversion spikes, lowering indoor $PM_{2.5}$ infiltration by up to $68\%$ while saving fan power during clean monsoon periods.
2. **Virtual Sensing for CAAQMS Observatories:** During optical monitor downtime, the telemetric nowcasting models synthesize reliable $PM_{2.5}$ estimates from co-located gas and $PM_{10}$ sensors with $R^2 > 0.90$, preventing compliance data loss.
3. **Targeted Graded Response Action Plan (GRAP):** Accurate early-warning forecasts enable municipal authorities to execute targeted, neighborhood-level traffic restrictions and construction halts rather than disruptive blanket city shutdowns.

---

## 📄 Academic Citation

If you use this codebase, models, or benchmarks in your research or coursework, please cite as follows:

```bibtex
@article{ranjan2026multicity,
  title={A Multi-Paradigm Ensemble and Deep Hybrid Framework for High-Resolution Ambient Air Quality Modeling: Comparative Evaluation of Causal Forecasting and Multi-Pollutant Telemetric Nowcasting Across Five Megacities in India},
  author={Ranjan, Aman Kumar},
  journal={CSIR-CBRI Technical Research Reports / ScienceDirect Submission},
  year={2026},
  institution={CSIR - Central Building Research Institute, Roorkee},
  note={Roll No: 23010301, B.Tech CSE Data Science}
}
```

---

## 📜 License

This project is licensed under the **MIT License** - see the [`LICENSE`](LICENSE) file for details.

Developed with ❤️ by **Aman Kumar Ranjan (23010301)** for **CSIR - Central Building Research Institute (CBRI)**.
=======
# csir-cbri-air-quality
>>>>>>> 753a00432f6d99d329a1fef868b1303a761c3ac4
