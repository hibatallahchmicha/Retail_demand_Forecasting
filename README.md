# 📦 Retail Demand Forecasting — Probabilistic & Multi-Horizon

[![CI](https://github.com/chm-hibatallah/Retail_demand_Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/chm-hibatallah/Retail_demand_Forecasting/actions/workflows/ci.yml)

> A production-grade time series forecasting system for retail supply chain, built on the M5 Walmart dataset.  
> Benchmarks 6+ models, produces probabilistic forecasts with uncertainty quantification, and exposes results via a live Streamlit dashboard.

---

## 🎯 Project Goals

| Goal | Details |
|---|---|
| **Multi-horizon forecasting** | Day-ahead to 28-day ahead predictions |
| **Probabilistic output** | Quantile forecasts (P10, P50, P90) + prediction intervals |
| **Model benchmarking** | SARIMA → LightGBM → TFT, evaluated with WRMSSE |
| **Real-world features** | Promotions, holidays, price effects, stockout handling |
| **Serving** | Streamlit dashboard with interactive filters |

---



## 🧱 Model Pipeline

```
Raw M5 Data
    │
    ▼
Preprocessing (missing values, stockouts, price normalization)
    │
    ▼
Feature Engineering (lags, rolling stats, calendar, price features)
    │
    ├──► SARIMA / ETS (per-series)
    ├──► LightGBM / XGBoost (global model)
    └──► Temporal Fusion Transformer (deep learning)
            │
            ▼
    Probabilistic Output (quantile regression / conformal)
            │
            ▼
    Evaluation (WRMSSE, MASE, calibration plots)
            │
            ▼
    Streamlit Dashboard
```

---

## 📊 Dataset: M5 (Walmart)

- **42,840 time series** — products × stores
- **1,913 days** of daily sales (2011–2016)
- **3 US states**, 10 stores, 3 product categories
- Includes: prices, calendar events, SNAP flags

Download from: [Kaggle M5 Competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)

Place files in `data/raw/`:
- `sales_train_evaluation.csv`
- `calendar.csv`
- `sell_prices.csv`

---

## ⚙️ Setup

```bash
# Clone
git clone https://github.com/chm-hibatallah/Retail_demand_Forecasting.git
cd Retail_demand_Forecasting

# Install
pip install -r Requirements.txt

# Run preprocessing
python -m src.data.preprocessing

# Build features
python -m src.features.engineering

# Train models
python -m src.models.naive_baseline
python -m src.models.lgbm_model
python -m src.models.xgboost_model

# Launch dashboard
streamlit run dashboard/dashboard_app.py
```

---

## 📈 Evaluation Metrics

| Metric | What it measures |
|---|---|
| **WRMSSE** | Weighted Root Mean Scaled Squared Error (M5 official) |
| **MASE** | Mean Absolute Scaled Error (scale-independent) |
| **Quantile Loss** | Calibration of probabilistic forecasts |
| **Coverage** | % of actuals within prediction interval |

---

## 🏆 Results Summary

| Rank | Model | MASE | Coverage 80% | Notes |
|------|-------|------|-------------|-------|
| 1 | LightGBM (Tuned) | ~0.817 | ~89% | Optuna-optimised hyperparameters |
| 2 | LightGBM (Default) | 0.842 | 88.9% | 5 min training time |
| 3 | XGBoost | 0.842 | 88.6% | 2 hrs training time |
| 4 | SARIMA | 1.871 | — | Worse than naive on item level |
| — | Naive Seasonal | 1.000 | — | Baseline (repeat last week) |

See [`results/benchmark.md`](results/benchmark.md) for detailed analysis.

---

## 📁 Project Structure

```
├── dashboard/
│   └── dashboard_app.py        # Streamlit dashboard with interactive filters
├── data/
│   ├── raw/                    # M5 CSV files (download from Kaggle)
│   └── processed/              # Parquet files, trained models, features
├── notebooks/
│   └── 01_EDA.ipynb            # Exploratory data analysis (33 cells)
├── results/
│   └── benchmark.md            # Detailed model comparison & findings
├── src/
│   ├── data/preprocessing.py   # ETL: raw CSV → long-format parquet
│   ├── evaluation/metrics.py   # MAE, MASE, WRMSSE, Coverage
│   ├── features/engineering.py # Lags, rolling stats, calendar, price features
│   └── models/
│       ├── naive_baseline.py   # Seasonal naive (repeat last week)
│       ├── sarima_model.py     # SARIMA per-series forecasting
│       ├── lgbm_model.py       # LightGBM global model
│       ├── xgboost_model.py    # XGBoost global model
│       ├── optuna_tuning.py    # Bayesian hyperparameter search
│       └── tft_model.py        # Temporal Fusion Transformer (GPU)
├── tests/                      # Unit tests for metrics & features
├── Requirements.txt
└── pyproject.toml
```

---

## 🗺️ Roadmap

- [x] Data preprocessing & ETL pipeline
- [x] Feature engineering (lags, rolling, calendar, price, target encoding)
- [x] Exploratory data analysis notebook
- [x] Naive seasonal baseline
- [x] SARIMA model
- [x] LightGBM model with quantile regression
- [x] XGBoost model with conformal prediction
- [x] Optuna hyperparameter tuning
- [x] Streamlit dashboard
- [x] Model benchmark report
- [ ] Temporal Fusion Transformer (GPU required)
- [x] CI/CD pipeline (GitHub Actions — lint + test + coverage)


