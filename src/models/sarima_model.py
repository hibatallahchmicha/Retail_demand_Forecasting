"""

SARIMA 

It captures three things our EDA confirmed:
  1. Weekly seasonality     (ACF lag-7 = 0.855)
  2. Short-term momentum    (PACF significant at lags 1-2)
  3. Seasonal unit root     (D=1 removes the weekly cycle)

Important limitation:
  SARIMA trains one model per series — it cannot share information
  across items or stores. With 30,490 series this is slow, so we
  run it on a sample for benchmarking purposes.

"""

import warnings
import numpy as np
import pandas as pd
from loguru import logger

# Suppress SARIMA convergence warnings during grid search
warnings.filterwarnings("ignore")


class SARIMAForecaster:
    """
    SARIMA(p,d,q)(P,D,Q)[s] forecaster for a single time series.

    Default order (1,0,1)(1,1,0)[7] is informed by our EDA:
      p=1  → AR(1): yesterday predicts today         (PACF lag-1 significant)
      d=0  → no differencing needed                  (ADF test: stationary)
      q=1  → MA(1): correct for recent forecast error
      P=1  → seasonal AR: same weekday last week     (PACF lag-6/7 significant)
      D=1  → seasonal differencing                   (removes weekly unit root)
      Q=0  → no seasonal MA term
      s=7  → weekly seasonal period                  (ACF lag-7 = 0.855)
    """

    def __init__(
        self,
        order: tuple = (1, 0, 1),
        seasonal_order: tuple = (1, 1, 0, 7),
        auto: bool = False,
    ):
        """
        Parameters
    
        order : tuple (p, d, q)
            Non-seasonal ARIMA orders.
        seasonal_order : tuple (P, D, Q, s)
            Seasonal orders + period s.
        auto : bool
            If True, uses pmdarima auto_arima to find the best order
            automatically. Slower but more accurate per series.
            If False, uses the fixed order above (fast, good default).
        """
        self.order         = order
        self.seasonal_order = seasonal_order
        self.auto          = auto
        self._model        = None    # fitted model stored here

    def fit(self, train_series: np.ndarray) -> "SARIMAForecaster":
        """
        Fit SARIMA to a single time series.

        """
        # Replace zeros with small value , it  helps SARIMA convergence
        # (pure zeros can cause log-transform issues internally)
        series = train_series.astype(float)
        series = np.where(series == 0, 0.1, series)

        if self.auto:
            # Auto-search for best SARIMA order using AIC criterion
            # This is slower (~10s per series) but finds optimal orders
            try:
                from pmdarima import auto_arima
                self._model = auto_arima(
                    series,
                    start_p=0, max_p=3,
                    start_q=0, max_q=2,
                    d=0,                    # ADF confirmed stationary
                    D=1,                    # seasonal differencing
                    m=7,                    # weekly period
                    seasonal=True,
                    information_criterion="aic",
                    stepwise=True,          # faster than exhaustive search
                    suppress_warnings=True,
                    error_action="ignore",
                )
            except ImportError:
                logger.warning("pmdarima not installed. Falling back to fixed order.")
                self.auto = False

        if not self.auto:
            # Use fixed order , fast and works well as a benchmark
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            fitted = SARIMAX(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)    # disp=False suppresses iteration output
            self._model = fitted

        return self

    def predict(self, horizon: int = 28) -> np.ndarray:
        """
        Forecast the next `horizon` days.
        Returns
        np.ndarray of shape (horizon,) with non-negative forecasts.
        """
        if self._model is None:
            raise RuntimeError("Call .fit() before .predict()")

        if self.auto:
            # pmdarima interface
            forecast = self._model.predict(n_periods=horizon)
        else:
            # statsmodels interface
            forecast = self._model.forecast(steps=horizon)

        # Clip negatives — sales cannot be negative
        return np.clip(forecast, 0, None)
    
# Multi-series benchmark runner


def run_sarima_benchmark(
    sales_wide: pd.DataFrame,
    n_series: int = 50,
    train_cutoff: str = "d_1913",
    horizon: int = 28,
    auto: bool = False,
    random_state: int = 42,
) -> dict:
    """
    Why a sample?
      SARIMA trains one model per series (~1-3 seconds each).
      30,490 series × 2 seconds = ~17 hours. Not practical.
      A sample of 50-200 series gives a reliable benchmark in minutes.

    Parameters
    sales_wide    : wide-format sales DataFrame (rows=series, cols=d_1...d_1941)
    n_series      : number of series to sample for benchmarking
    train_cutoff  : last training day column name
    horizon       : forecast horizon in days
    auto          : use auto_arima (slower, better) or fixed order (faster)
    random_state  : for reproducible sampling

    Returns
    dict with keys:
        forecasts   : np.ndarray (n_series, horizon)
        actuals     : np.ndarray (n_series, horizon)
        train_data  : np.ndarray (n_series, n_train_days)
        series_ids  : list of series ids
        failed      : list of series that failed to fit
    """
    # Identify day columns
    all_d_cols  = [c for c in sales_wide.columns if c.startswith("d_")]
    cutoff_idx  = all_d_cols.index(train_cutoff)
    train_cols  = all_d_cols[:cutoff_idx + 1]
    test_cols   = all_d_cols[cutoff_idx + 1: cutoff_idx + 1 + horizon]

    if len(test_cols) < horizon:
        raise ValueError(
            f"Not enough test days after {train_cutoff}. "
            f"Need {horizon}, found {len(test_cols)}."
        )

    # Sample series
    sample = sales_wide.sample(n=min(n_series, len(sales_wide)), random_state=random_state)
    logger.info(f"Running SARIMA on {len(sample)} series (horizon={horizon} days) …")

    forecasts  = []
    actuals    = []
    train_data = []
    series_ids = []
    failed     = []

    for i, (_, row) in enumerate(sample.iterrows()):
        series_id    = row["id"]
        train_series = row[train_cols].values.astype(float)
        test_series  = row[test_cols].values.astype(float)

        try:
            model = SARIMAForecaster(auto=auto)
            model.fit(train_series)
            pred = model.predict(horizon=horizon)

            forecasts.append(pred)
            actuals.append(test_series)
            train_data.append(train_series)
            series_ids.append(series_id)

            if (i + 1) % 10 == 0:
                logger.debug(f"  Progress: {i+1}/{len(sample)} series done")

        except Exception as e:
            logger.warning(f"  SARIMA failed for {series_id}: {e}")
            failed.append(series_id)

    logger.success(
        f"SARIMA complete — "
        f"{len(forecasts)} succeeded, {len(failed)} failed"
    )

    return {
        "forecasts":  np.array(forecasts),
        "actuals":    np.array(actuals),
        "train_data": np.array(train_data),
        "series_ids": series_ids,
        "failed":     failed,
    }


# CLI — run SARIMA benchmark and print results

if __name__ == "__main__":
    from pathlib import Path
    from src.evaluation.metrics import evaluate, make_leaderboard
    from src.models.naive_baseline import run_naive_forecast

    DATA_DIR = Path("data/raw")

    # Load data
    logger.info("Loading sales data …")
    sales = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv")

    # Run SARIMA on 50 series
    results = run_sarima_benchmark(sales, n_series=50, train_cutoff="d_1913")

    # Evaluate , average metrics across all series
    all_mae, all_mase = [], []
    for i in range(len(results["forecasts"])):
        from src.evaluation.metrics import mae, mase
        all_mae.append(mae(results["actuals"][i], results["forecasts"][i]))
        all_mase.append(mase(
            results["actuals"][i],
            results["forecasts"][i],
            results["train_data"][i],
        ))

    sarima_report = {
        "model": "SARIMA(1,0,1)(1,1,0)[7]",
        "MAE":   float(np.mean(all_mae)),
        "MASE":  float(np.mean(all_mase)),
    }

    print("\n── SARIMA Benchmark Results (50 series sample) ──")
    print(f"  MAE  : {sarima_report['MAE']:.4f}")
    print(f"  MASE : {sarima_report['MASE']:.4f}")
    print(f"  Failed series: {len(results['failed'])}")
    print("\nMASE < 1.0 means SARIMA beats the naive baseline ")
    print("MASE > 1.0 means SARIMA is worse than naive        ")
    
    print(sales.sample(50, random_state=42)[['cat_id','store_id']].value_counts())