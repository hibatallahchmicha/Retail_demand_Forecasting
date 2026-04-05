"""


The core problem:
  59M rows × 50+ columns = too large to hold in RAM on a typical machine.

The solution:
  Process one store at a time → write each store's features to parquet → done.
  Never concatenate all stores in memory at once.

"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


# ── Constants ──────────────────────────────────────────────────────────────────
LAG_DAYS        = [7, 14, 21, 28, 56]
ROLLING_WINDOWS = [7, 28, 56]
MIN_TRAIN_DATE  = "2012-04-01"   # burn-in: need 56 days history for lags
FEATURES_DIR    = Path("data/processed/features")


# ── Main entry point ───────────────────────────────────────────────────────────
def build_and_save_features(
    df: pd.DataFrame,
    is_train: bool = True,
    target_enc_stats: dict | None = None,
    output_dir: Path = FEATURES_DIR,
) -> dict:
    """
    Build features store-by-store and save each to its own parquet file.
    Never loads more than one store into memory at a time.

    Parameters
    ----------
    df              : full long-format DataFrame from preprocessing
    is_train        : if True, compute and save target encoding stats
    target_enc_stats: precomputed stats for inference
    output_dir      : where to save parquet files (one per store)

    Returns
    -------
    stats dict (target encoding — needed at inference time)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stores = df["store_id"].unique()
    logger.info(f"▶ Building features for {len(stores)} stores → {output_dir}")

    # Compute target encoding stats on full df BEFORE chunking
    # (needs all data to compute global means)
    logger.info("  Computing target encoding stats …")
    stats = _compute_target_enc_stats(df) if is_train else target_enc_stats

    # Process one store at a time
    for i, store in enumerate(stores):
        out_path = output_dir / f"{store}.parquet"

        # Skip if already built (useful for resuming interrupted runs)
        if out_path.exists():
            logger.debug(f"  [{i+1}/{len(stores)}] {store} — already exists, skipping")
            continue

        logger.info(f"  [{i+1}/{len(stores)}] Processing {store} …")
        chunk = df[df["store_id"] == store].copy()
        chunk = chunk.sort_values(["id", "date"]).reset_index(drop=True)

        # Add all feature groups
        chunk = _add_lag_features(chunk)
        chunk = _add_rolling_features(chunk)
        chunk = _add_calendar_features(chunk)
        chunk = _add_event_features(chunk)
        chunk = _add_snap_features(chunk)
        chunk = _add_price_features(chunk)
        chunk = _apply_target_encoding(chunk, stats)

        # Drop burn-in rows
        if is_train:
            chunk = chunk[chunk["date"] >= MIN_TRAIN_DATE]

        # Drop rows with NaN in lag features (first 56 days)
        chunk = chunk.dropna(subset=[f"sales_lag_{LAG_DAYS[-1]}"])

        # Save to disk — free memory immediately
        chunk.to_parquet(out_path, index=False)
        logger.debug(f"    Saved {len(chunk):,} rows → {out_path.name}")

        # Explicitly free memory
        del chunk

    logger.success(f"✔ All stores processed → {output_dir}")
    return stats


def load_features(
    output_dir: Path = FEATURES_DIR,
    stores: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load pre-built features from parquet files.
    Optionally filter to specific stores.

    Parameters
    ----------
    output_dir : directory containing per-store parquet files
    stores     : list of store_ids to load (None = all)

    Returns
    -------
    pd.DataFrame with all features
    """
    parquet_files = sorted(output_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No feature files found in {output_dir}.\n"
            "Run first: python -m src.features.engineering"
        )

    if stores:
        parquet_files = [f for f in parquet_files if f.stem in stores]

    logger.info(f"Loading features from {len(parquet_files)} store files …")
    df = pd.concat(
        [pd.read_parquet(f) for f in parquet_files],
        ignore_index=True,
    )
    logger.success(f"✔ Features loaded — shape: {df.shape}")
    return df


# ── Feature groups ─────────────────────────────────────────────────────────────

def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for lag in LAG_DAYS:
        df[f"sales_lag_{lag}"] = df.groupby("id")["sales"].shift(lag)
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    for window in ROLLING_WINDOWS:
        shifted = df.groupby("id")["sales"].shift(1)
        df[f"sales_roll_mean_{window}"] = (
            shifted.groupby(df["id"])
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"sales_roll_std_{window}"] = (
            shifted.groupby(df["id"])
            .transform(lambda x: x.rolling(window, min_periods=1).std())
            .fillna(0)
        )
    return df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"]    = df["date"].dt.dayofweek.astype(np.int8)
    df["day_of_month"]   = df["date"].dt.day.astype(np.int8)
    df["month"]          = df["date"].dt.month.astype(np.int8)
    df["year"]           = df["date"].dt.year.astype(np.int16)
    df["is_weekend"]     = (df["day_of_week"] >= 5).astype(np.int8)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(np.int8)
    df["is_month_end"]   = df["date"].dt.is_month_end.astype(np.int8)

    # Cyclical encoding — smooth representation of weekly/yearly cycles
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7).astype(np.float32)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7).astype(np.float32)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)
    return df


def _add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    df["has_event"]      = df["event_name_1"].notna().astype(np.int8)
    df["has_two_events"] = df["event_name_2"].notna().astype(np.int8)

    # Integer encoding instead of one-hot — avoids memory explosion
    event_type_map = {"National": 1, "Sporting": 2, "Religious": 3, "Cultural": 4}
    df["event_type_1_enc"] = df["event_type_1"].map(event_type_map).fillna(0).astype(np.int8)
    df["event_type_2_enc"] = df["event_type_2"].map(event_type_map).fillna(0).astype(np.int8)
    return df


def _add_snap_features(df: pd.DataFrame) -> pd.DataFrame:
    # Vectorized — no df.apply which would OOM on large DataFrames
    df["is_snap_day"] = np.where(
        df["state_id"] == "CA", df["snap_CA"],
        np.where(df["state_id"] == "TX", df["snap_TX"],
        np.where(df["state_id"] == "WI", df["snap_WI"], 0))
    ).astype(np.int8)
    return df


def _add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df["log_price"] = np.log1p(df["sell_price"].fillna(0)).astype(np.float32)

    # Price momentum — week over week change
    df["price_lag_1w"]   = df.groupby("id")["sell_price"].shift(7)
    df["price_momentum"] = (
        (df["sell_price"] - df["price_lag_1w"]) / (df["price_lag_1w"] + 1e-6)
    ).astype(np.float32)
    df.drop(columns=["price_lag_1w"], inplace=True)

    # Price vs dept mean — merge approach avoids transform OOM
    dept_mean = (
        df.groupby(["dept_id", "date"])["sell_price"]
        .mean().rename("_dept_mean").reset_index()
    )
    df = df.merge(dept_mean, on=["dept_id", "date"], how="left")
    df["price_vs_dept_mean"] = (df["sell_price"] / (df["_dept_mean"] + 1e-6)).astype(np.float32)
    df.drop(columns=["_dept_mean"], inplace=True)

    return df


# ── Target encoding ────────────────────────────────────────────────────────────

def _compute_target_enc_stats(df: pd.DataFrame) -> dict:
    """Compute mean sales per (item×store) and (dept×store) from full df."""
    return {
        "item_store": df.groupby(["item_id", "store_id"])["sales"].mean().rename("item_store_enc"),
        "dept_store": df.groupby(["dept_id", "store_id"])["sales"].mean().rename("dept_store_enc"),
        "cat_store":  df.groupby(["cat_id",  "store_id"])["sales"].mean().rename("cat_store_enc"),
    }


def _apply_target_encoding(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Merge precomputed target encoding stats into chunk."""
    df = df.merge(stats["item_store"].reset_index(), on=["item_id", "store_id"], how="left")
    df = df.merge(stats["dept_store"].reset_index(), on=["dept_id", "store_id"], how="left")
    df = df.merge(stats["cat_store"].reset_index(),  on=["cat_id",  "store_id"], how="left")
    return df


# ── Feature column catalogue ───────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Returns all numeric feature columns — excludes ids, target, raw strings."""
    exclude = {
        "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "sales", "date", "d", "wm_yr_wk", "weekday",
        "event_name_1", "event_name_2", "event_type_1", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI", "is_stockout",
    }
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import joblib

    PROCESSED_DIR = Path("data/processed")
    parquet = PROCESSED_DIR / "m5_long.parquet"

    if not parquet.exists():
        raise FileNotFoundError(
            "Run preprocessing first:\n"
            "  python -m src.data.preprocessing"
        )

    logger.info(f"Loading {parquet} …")
    df_raw = pd.read_parquet(parquet)

    stats = build_and_save_features(df_raw, is_train=True)

    # Save encoding stats for inference
    stats_path = PROCESSED_DIR / "target_enc_stats.pkl"
    joblib.dump(stats, stats_path)
    logger.success(f"✔ Encoding stats saved → {stats_path}")

    # Show feature list from first store file
    sample = pd.read_parquet(FEATURES_DIR / f"{df_raw['store_id'].iloc[0]}.parquet")
    feat_cols = get_feature_columns(sample)
    logger.info(f"Total features: {len(feat_cols)}")
    logger.info(f"Features: {feat_cols}")