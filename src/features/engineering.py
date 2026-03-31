"""
Transforms the clean long-format M5 DataFrame into a feature matrix
ready for LightGBM / XGBoost (and as the basis for TFT inputs).

Feature groups
──────────────
1.  Lag features          — sales at t-7, t-14, t-28, t-56  (backed by ACF > 0.82)
2.  Rolling statistics    — mean / std over 7 / 28 / 56 day windows
3.  Calendar features     — day-of-week (cyclical), week-of-year, month, is_weekend
4.  Event / holiday flags — binary flags from M5 calendar
5.  SNAP flags            — state-specific SNAP benefit days
6.  Price features        — sell_price, price_momentum, price_vs_dept_mean
7.  Target encoding       — mean sales per (item × store) over training window

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


# Constants 
# Backed by ACF analysis: lag-7=0.855, lag-14=0.828, lag-21=0.819, lag-28=0.845
LAG_DAYS        = [7, 14, 21, 28, 56]
ROLLING_WINDOWS = [7, 28, 56]

# Burn-in: we need at least 56 days of history before features are valid
MIN_TRAIN_DATE = "2012-04-01"


# Main entry point 
def build_features(
    df: pd.DataFrame,
    is_train: bool = True,
    target_enc_stats: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Output of preprocessing pipeline (long format).
        Must contain: id, item_id, dept_id, cat_id, store_id, state_id,
                      date, d, sales, wm_yr_wk, sell_price,
                      event_name_1/2, event_type_1/2, snap_CA/TX/WI
    is_train : bool
        True  → compute and return target encoding stats from this data.
        False → apply precomputed stats (pass via target_enc_stats).
    target_enc_stats : dict | None
        Required when is_train=False. Dict with keys:
        'item_store_mean', 'dept_store_mean'

    Returns
    -------
    (feature_df, stats_dict)
        feature_df : DataFrame with all features + target column 'sales'
        stats_dict : target encoding stats to save and reuse at inference
    """
    logger.info("Building feature matrix …")
    df = df.copy()

    # Ensure sorted — critical for lag/rolling features to be correct
    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    df = _add_calendar_features(df)
    df = _add_event_features(df)
    df = _add_snap_features(df)
    df = _add_price_features(df)
    df, stats = _add_target_encoding(df, is_train, target_enc_stats)

    # Drop burn-in rows where lags are NaN (not valid for training)
    if is_train:
        before = len(df)
        df = df[df["date"] >= MIN_TRAIN_DATE].reset_index(drop=True)
        logger.debug(f"  Dropped {before - len(df):,} burn-in rows (before {MIN_TRAIN_DATE})")

    logger.success(f" Feature matrix ready — shape: {df.shape}")
    return df, stats


# 1. Lag features 
def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("  [1/6] Lag features …")
    for lag in LAG_DAYS:
        df[f"sales_lag_{lag}"] = df.groupby("id")["sales"].shift(lag)
    return df


#  2. Rolling statistics 
def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("  [2/6] Rolling statistics …")
    # Shift by 1 first to avoid data leakage (don't use today's sales)
    for window in ROLLING_WINDOWS:
        rolled = (
            df.groupby("id")["sales"]
            .shift(1)
            .groupby(df["id"])
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"sales_roll_mean_{window}"] = rolled

        rolled_std = (
            df.groupby("id")["sales"]
            .shift(1)
            .groupby(df["id"])
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )
        df[f"sales_roll_std_{window}"] = rolled_std.fillna(0)

    return df


#  3. Calendar features 
def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("  [3/6] Calendar features …")

    df["day_of_week"]  = df["date"].dt.dayofweek        # 0=Monday, 6=Sunday
    df["day_of_month"] = df["date"].dt.day
    df["month"]        = df["date"].dt.month
    df["year"]         = df["date"].dt.year
    df["quarter"]      = df["date"].dt.quarter
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(np.int16)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(np.int8)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(np.int8)
    df["is_month_end"]   = df["date"].dt.is_month_end.astype(np.int8)

    # Cyclical encoding — prevents the model seeing Mon(0) and Sun(6) as far apart
    # ACF confirmed lag-7=0.855, so the weekly cycle must be encoded smoothly
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["woy_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["woy_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    return df


#  4. Event / holiday features 
def _add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("  [4/6] Event features …")

    df["has_event"]      = df["event_name_1"].notna().astype(np.int8)
    df["has_two_events"] = df["event_name_2"].notna().astype(np.int8)

    # One-hot encode event types (National, Sporting, Religious, Cultural)
    for col in ["event_type_1", "event_type_2"]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False).astype(np.int8)
            df = pd.concat([df, dummies], axis=1)

    return df


# 5. SNAP features
def _add_snap_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("  [5/6] SNAP features …")

    # Map each row to the correct state's SNAP flag
    snap_map = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}

    def get_snap(row):
        col = snap_map.get(str(row["state_id"]), None)
        return int(row[col]) if col and col in row.index else 0

    df["is_snap_day"] = df.apply(get_snap, axis=1).astype(np.int8)
    return df


#  6. Price features 
def _add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("  [6/6] Price features …")

    # Log price — handles heavy right skew across categories
    df["log_price"] = np.log1p(df["sell_price"].fillna(0))

    # Week-over-week price momentum — catches markdowns and promotions
    df["price_lag_1w"]   = df.groupby("id")["sell_price"].shift(7)
    df["price_momentum"] = (
        (df["sell_price"] - df["price_lag_1w"]) / (df["price_lag_1w"] + 1e-6)
    )
    df.drop(columns=["price_lag_1w"], inplace=True)

    # Price relative to dept mean — signals whether item is on promotion
    dept_price_mean = df.groupby(["dept_id", "date"])["sell_price"].transform("mean")
    df["price_vs_dept_mean"] = df["sell_price"] / (dept_price_mean + 1e-6)

    # Price relative to store mean — captures store-level pricing strategy
    store_price_mean = df.groupby(["store_id", "date"])["sell_price"].transform("mean")
    df["price_vs_store_mean"] = df["sell_price"] / (store_price_mean + 1e-6)

    return df


# 7. Target encoding 
def _add_target_encoding(
    df: pd.DataFrame,
    is_train: bool,
    precomputed: dict | None,
) -> tuple[pd.DataFrame, dict]:
    stats = {}

    if is_train:
        logger.debug("  Computing target encoding stats from training data …")
        item_store_mean = (
            df.groupby(["item_id", "store_id"])["sales"]
            .mean()
            .rename("item_store_enc")
        )
        dept_store_mean = (
            df.groupby(["dept_id", "store_id"])["sales"]
            .mean()
            .rename("dept_store_enc")
        )
        cat_store_mean = (
            df.groupby(["cat_id", "store_id"])["sales"]
            .mean()
            .rename("cat_store_enc")
        )
        stats = {
            "item_store_mean": item_store_mean,
            "dept_store_mean": dept_store_mean,
            "cat_store_mean":  cat_store_mean,
        }
    else:
        if precomputed is None:
            raise ValueError("Must pass target_enc_stats when is_train=False")
        item_store_mean = precomputed["item_store_mean"]
        dept_store_mean = precomputed["dept_store_mean"]
        cat_store_mean  = precomputed["cat_store_mean"]

    df = df.merge(item_store_mean.reset_index(), on=["item_id", "store_id"], how="left")
    df = df.merge(dept_store_mean.reset_index(), on=["dept_id", "store_id"], how="left")
    df = df.merge(cat_store_mean.reset_index(),  on=["cat_id",  "store_id"], how="left")

    return df, stats


# Feature column catalogue 
def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Returns all numeric feature columns suitable for modelling.
    Excludes raw identifiers, the target, and non-numeric columns.
    """
    exclude = {
        # Identifiers
        "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
        # Target
        "sales",
        # Raw date / calendar strings
        "date", "d", "wm_yr_wk", "weekday",
        # Raw event strings (one-hotted versions are included)
        "event_name_1", "event_name_2", "event_type_1", "event_type_2",
        # Raw SNAP columns (is_snap_day is the engineered version)
        "snap_CA", "snap_TX", "snap_WI",
        # Stockout flag (used for filtering, not as a feature)
        "is_stockout",
    }
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


# CLI 
if __name__ == "__main__":
    from pathlib import Path
    import joblib

    PROCESSED_DIR = Path("data/processed")
    parquet = PROCESSED_DIR / "m5_long.parquet"

    if not parquet.exists():
        raise FileNotFoundError(
            f"Processed data not found at {parquet}\n"
            "Run first: python -m src.data.preprocessing"
        )

    logger.info(f"Loading {parquet} …")
    df_raw = pd.read_parquet(parquet)

    df_feat, enc_stats = build_features(df_raw, is_train=True)

    # Save features
    out_feat = PROCESSED_DIR / "m5_features.parquet"
    df_feat.to_parquet(out_feat, index=False)
    logger.success(f"✔ Features saved → {out_feat}")

    # Save encoding stats for inference
    out_stats = PROCESSED_DIR / "target_enc_stats.pkl"
    joblib.dump(enc_stats, out_stats)
    logger.success(f"✔ Encoding stats saved → {out_stats}")

    # Print feature summary
    feat_cols = get_feature_columns(df_feat)
    logger.info(f"Total feature columns: {len(feat_cols)}")
    logger.info(f"Feature list:\n{feat_cols}")