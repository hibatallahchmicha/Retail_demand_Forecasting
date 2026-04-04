"""


What it does:
  For each series, forecast the next 28 days by repeating
  the last 28 days of known sales (same-weekday logic).

"""

import numpy as np
import pandas as pd
from loguru import logger


class NaiveSeasonalForecast:
    """
    Seasonal Naive forecaster with period=7 (weekly).

    Usage

        model = NaiveSeasonalForecast(seasonality=7)
        model.fit(train_series)          # just stores the last 7 values
        predictions = model.predict(horizon=28)
    """

    def __init__(self, seasonality: int = 7):
        """
        Parameters
        seasonality : int
            The seasonal period. 7 = weekly (default for M5).
            The last `seasonality` values are repeated to form the forecast.
        """
        self.seasonality = seasonality
        self._last_season: np.ndarray | None = None   # stores last 7 days

    def fit(self, train_series: np.ndarray) -> "NaiveSeasonalForecast":
        """
        "Training" = just memorise the last `seasonality` values.
        Parameters
        train_series : np.ndarray
            1D array of historical sales for one series.
            Shape: (n_days,)
        """
        if len(train_series) < self.seasonality:
            raise ValueError(
                f"Training series too short. "
                f"Need at least {self.seasonality} days, got {len(train_series)}."
            )
        # Store only the last `seasonality` values
        self._last_season = train_series[-self.seasonality:].copy()
        return self

    def predict(self, horizon: int = 28) -> np.ndarray:
        """
        Forecast the next `horizon` days by tiling the last season.

        Parameters
        horizon : int
            Number of days to forecast. Default = 28 (M5 horizon).
        Returns
        np.ndarray of shape (horizon,) with non-negative forecasts.
        """
        if self._last_season is None:
            raise RuntimeError("Call .fit() before .predict()")

        # Tile the last season enough times to cover the full horizon
        # then slice to exactly `horizon` days
        n_repeats = int(np.ceil(horizon / self.seasonality))
        forecast  = np.tile(self._last_season, n_repeats)[:horizon]

        # Sales can't be negative , clip just in case
        return np.clip(forecast, 0, None)

# Multi-series helper

def run_naive_forecast(
    sales_wide: pd.DataFrame,
    train_cutoff_day: str = "d_1913",
    horizon: int = 28,
) -> pd.DataFrame:
    """
    Run the Naive Seasonal forecast across ALL series at once.

    Parameters
    sales_wide : pd.DataFrame
        Wide-format sales DataFrame (rows = series, cols = d_1 ... d_1941).
        Same format as sales_train_evaluation.csv.
    train_cutoff_day : str
        Last day of training data. Default = "d_1913" (M5 validation split).
    horizon : int
        Forecast horizon in days. Default = 28.
    Returns
    pd.DataFrame
        Shape: (n_series, horizon)
        Index: series id
        Columns: F1, F2, ..., F28
    """
    # Identify training columns
    all_d_cols   = [c for c in sales_wide.columns if c.startswith("d_")]
    cutoff_idx   = all_d_cols.index(train_cutoff_day)
    train_cols   = all_d_cols[:cutoff_idx + 1]

    logger.info(f"Running Naive forecast — {len(sales_wide):,} series, horizon={horizon} days")

    forecasts = []
    model = NaiveSeasonalForecast(seasonality=7)

    for _, row in sales_wide.iterrows():
        train_series = row[train_cols].values.astype(float)
        model.fit(train_series)
        forecasts.append(model.predict(horizon=horizon))

    # Build output DataFrame matching sample_submission format
    forecast_df = pd.DataFrame(
        forecasts,
        index=sales_wide["id"],
        columns=[f"F{i}" for i in range(1, horizon + 1)],
    )

    logger.success(f" Naive forecast complete — shape: {forecast_df.shape}")
    return forecast_df

# CLI — run and evaluate the naive baseline directly

if __name__ == "__main__":
    from pathlib import Path
    from src.evaluation.metrics import evaluate, mae

    DATA_DIR = Path("data/raw")

    # Load data
    logger.info("Loading sales data …")
    sales = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv")

    all_d_cols = [c for c in sales.columns if c.startswith("d_")]

    # Split: train = d_1 to d_1913, test = d_1914 to d_1941
    train_cols = [c for c in all_d_cols if int(c.split("_")[1]) <= 1913]
    test_cols  = [c for c in all_d_cols if int(c.split("_")[1]) >  1913]

    logger.info(f"Train days: {len(train_cols)} | Test days: {len(test_cols)}")

    # Run naive forecast
    forecast_df = run_naive_forecast(sales, train_cutoff_day="d_1913", horizon=28)

    # Evaluate on first series as a sanity check
    first_id     = sales["id"].iloc[0]
    actual_vals  = sales.loc[sales["id"] == first_id, test_cols].values.flatten()
    pred_vals    = forecast_df.loc[first_id].values
    train_vals   = sales.loc[sales["id"] == first_id, train_cols].values.flatten()

    report = evaluate(
        actual         = actual_vals,
        point_forecast = pred_vals,
        train_series   = train_vals,
        model_name     = "Naive Seasonal",
    )

    print("\n── Naive Baseline Results (single series sanity check) ──")
    for k, v in report.items():
        print(f"  {k:<15} {v}")

    # Save forecasts
    out = Path("data/processed/naive_forecast.csv")
    forecast_df.to_csv(out)
    logger.success(f"✔ Saved → {out}")