""" 
the metrics i  used in this script are 

    WRMSSE   → official M5 metric — weighted accuracy across all series
    MAE      → simple, explainable average error
    MASE     → are we beating "just repeat last week"?
    Coverage → are our prediction intervals honest?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# METRIC 1 , MAE (Mean Absolute Error)


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Average absolute difference between actual and predicted sales.

    Example:
      actual   = [10, 20, 15]
      forecast = [8,  25, 14]
      errors   = [2,   5,  1]  → MAE = (2+5+1)/3 = 2.67 units

    Interpretation: on average, our forecast is off by 2.67 units per day.
    Unit is the same as your target (units sold).
    Lower is better.
    """
    return float(np.mean(np.abs(actual - forecast)))


# METRIC 2 , MASE (Mean Absolute Scaled Error)

def mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    train_series: np.ndarray,
    seasonality: int = 7,
) -> float:
    """
    Are we better than the dumbest reasonable baseline?

    The baseline (naive forecast):
      "Tomorrow's sales = same day last week"
      e.g. Monday's forecast = last Monday's actual sales

    How MASE works:
      1. Compute our MAE on the test set
      2. Compute the naive forecast's MAE on the training set
      3. Return the ratio: our MAE / naive MAE

    Interpretation:
      MASE = 0.6  → we are 40% better than naive           good
      MASE = 1.0  → same performance as naive              not useful
      MASE = 1.4  → 40% WORSE than just repeating history  bad

    We use seasonality=7 because ACF confirmed weekly cycles dominate (lag-7=0.855).
    Lower is better.
    """
    # Naive forecast error: how wrong is "copy same day last week" on training data
    naive_errors = np.abs(train_series[seasonality:] - train_series[:-seasonality])
    naive_mae    = naive_errors.mean() + 1e-8   # +1e-8 avoids division by zero

    return float(mae(actual, forecast) / naive_mae)


# METRIC 3 , WRMSSE (M5 Official Metric)

def wrmsse(
    actuals: np.ndarray,
    forecasts: np.ndarray,
    train_sales: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """
    Weighted Root Mean Scaled Squared Error — the M5 competition's official score.

    The problem with plain RMSE across 30,000+ series:
      - A 10-unit error on a product selling 5/day is catastrophic
      - A 10-unit error on a product selling 500/day is negligible
      Plain RMSE treats them the same. WRMSSE does not.

    WRMSSE fixes this in two ways:

      1. SCALING — each series' errors are divided by that series' own
         historical variance. This makes errors comparable across
         fast-moving and slow-moving items.

      2. WEIGHTING — each series is weighted by its sales volume.
         High-revenue items matter more to the retailer, so errors
         on those series are penalised more.

    Parameters
    actuals    : shape (n_series, 28) — ground truth for forecast horizon
    forecasts  : shape (n_series, 28) — model predictions
    train_sales: shape (n_series, n_days) — full training history
                 used to compute the scale for each series
    weights    : shape (n_series,) — optional custom weights
                 defaults to sales-volume proportional weights

    Lower is better. Best possible = 0.
    """
    horizon = actuals.shape[1]

    # Step 1: Scale 

    last_window = train_sales[:, -horizon:]            # last 28 days
    prev_window = train_sales[:, -2 * horizon:-horizon] # 28 days before that
    scale = np.mean((last_window - prev_window) ** 2, axis=1) + 1e-8

    #  Step 2: RMSSE per series 
    # Root Mean Squared Error divided by the scale computed above
    squared_errors   = (actuals - forecasts) ** 2          # (n_series, 28)
    rmsse_per_series = np.sqrt(np.mean(squared_errors, axis=1) / scale)

    #  Step 3: Weights 
    # Default: weight each series by its share of total sales volume
    if weights is None:
        total_sales = train_sales[:, -horizon:].sum(axis=1).astype(float)
        weights     = total_sales / (total_sales.sum() + 1e-8)

    # Step 4: Weighted average
    return float(np.sum(weights * rmsse_per_series))


# METRIC 4 — Coverage

def coverage(
    actual: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
) -> float:
    """
    What fraction of actual values fell inside our 80% prediction interval?

    Our model produces three forecasts per day:
      q10 = lower bound (we expect only 10% of actuals to fall below this)
      q90 = upper bound (we expect only 10% of actuals to exceed this)
      → The interval [q10, q90] should capture ~80% of actuals

    Interpretation:
      coverage = 0.80 → perfectly calibrated              
      coverage = 0.55 → intervals too narrow, overconfident 
      coverage = 0.97 → intervals too wide, not useful     

    A model that says "sales will be between 0 and 10,000 units" has
    perfect coverage but is completely useless. That's why we report
    coverage alongside the forecast — not as the only metric.

    Returns a value between 0 and 1.
    """
    inside = (actual >= q10) & (actual <= q90)
    return float(np.mean(inside))



# UNIFIED REPORT


def evaluate(
    actual: np.ndarray,
    point_forecast: np.ndarray,
    train_series: np.ndarray,
    q10: np.ndarray | None = None,
    q90: np.ndarray | None = None,
    model_name: str = "model",
) -> dict:
    """
    Run all metrics for one model in a single call.

    Parameters

    actual         : ground truth sales values
    point_forecast : model's central (point) prediction
    train_series   : training history — needed to compute MASE baseline
    q10            : lower bound of 80% prediction interval (optional)
    q90            : upper bound of 80% prediction interval (optional)
    model_name     : label shown in logs and leaderboard

    Returns
    dict with keys: model, MAE, MASE, Coverage (if q10/q90 provided)
    """
    report = {
        "model": model_name,
        "MAE":   mae(actual, point_forecast),
        "MASE":  mase(actual, point_forecast, train_series),
    }

    # Coverage is only computed when probabilistic forecasts are provided
    if q10 is not None and q90 is not None:
        report["Coverage_80%"] = coverage(actual, q10, q90)

    # Log a clean one-liner summary
    cov_str = f" | Coverage_80%={report['Coverage_80%']:.3f}" if "Coverage_80%" in report else ""
    logger.info(
        f"[{model_name}] "
        f"MAE={report['MAE']:.3f} | "
        f"MASE={report['MASE']:.3f}"
        f"{cov_str}"
    )

    return report


def make_leaderboard(reports: list[dict]) -> pd.DataFrame:
    """
    Combine multiple model reports into one ranked table.
    Sorted by MAE ascending — best model at the top.
    """
    df = pd.DataFrame(reports)
    df = df.sort_values("MAE").reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"

    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].round(4)

    return df