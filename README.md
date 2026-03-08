# 📦 Retail Demand Forecasting — Probabilistic & Multi-Horizon

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
| **Serving** | FastAPI backend + Streamlit dashboard |

---

## 🗂️ Repository Structure

```
demand-forecast/
├── data/
│   ├── raw/                    # Original M5 files (not committed)
│   ├── processed/              # Cleaned, merged datasets
│   └── external/               # Calendar, prices
├── src/
│   ├── data/                   # Ingestion & preprocessing
│   ├── features/               # Feature engineering
│   ├── models/                 # Classical, ML, DL model wrappers
│   ├── evaluation/             # Metrics: WRMSSE, MASE, coverage
│   └── api/                    # FastAPI serving layer
├── dashboard/                  # Streamlit app
├── notebooks/                  # EDA, experiments
├── configs/                    # YAML configs per model
├── tests/                      # Unit tests
└── scripts/                    # Train, evaluate, serve entrypoints
```

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
    FastAPI + Streamlit Dashboard
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
cd demand-forecast

# Install
pip install -r requirements.txt

# Run preprocessing
python scripts/preprocess.py

# Train all models
python scripts/train.py --config configs/lgbm.yaml

# Launch dashboard
streamlit run dashboard/app.py
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

## 🗺️ Roadmap

- [x] Repo structure & data pipeline
- [ ] EDA notebook
- [ ] Feature engineering module
- [ ] Baseline models (SARIMA, ETS)
- [ ] LightGBM global model
- [ ] Temporal Fusion Transformer
- [ ] Conformal prediction intervals
- [ ] Evaluation dashboard
- [ ] FastAPI serving layer
- [ ] Streamlit app

---
