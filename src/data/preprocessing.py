"""
Loads and preprocesses M5 Walmart dataset into a clean long-format DataFrame.

Steps:
  1. Load raw CSVs (sales, calendar, prices)
  2. Melt wide sales → long format
  3. Merge calendar (day_id → date, events, SNAP flags)
  4. Merge sell prices
  5. Handle stockouts (zero-sales detection vs true zeros)
  6. Parse hierarchy (state / store / category / dept / item)
  7. Save processed parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


# Paths
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SALES_FILE    = RAW_DIR / "sales_train_evaluation.csv"   # full eval set (1941 days)
SALES_VAL_FILE= RAW_DIR / "sales_train_validation.csv"  # validation set (1913 days)
CALENDAR_FILE = RAW_DIR / "calendar.csv"
PRICES_FILE   = RAW_DIR / "sell_prices.csv"
SUBMISSION_FILE = RAW_DIR / "sample_submission.csv"     # defines forecast ids


# Main entry point 
def run_preprocessing(
    sales_path: Path = SALES_FILE,
    calendar_path: Path = CALENDAR_FILE,
    prices_path: Path = PRICES_FILE,
    output_path: Path = PROCESSED_DIR / "m5_long.parquet",
    sample_stores: list[str] | None = None,   # e.g. ["CA_1"] for fast dev
) -> pd.DataFrame:
    """
    Full preprocessing pipeline. Returns clean long-format DataFrame.

    Parameters :
    sales_path, calendar_path, prices_path : Path
        Paths to raw M5 CSV files.
    output_path : Path
        Where to write the processed parquet.
    sample_stores : list[str] | None
        Subset of store_ids to keep (useful during development).

    Returns :
    
    pd.DataFrame with columns:
        id, item_id, dept_id, cat_id, store_id, state_id,
        date, d, sales, wm_yr_wk,
        event_name_1, event_type_1, event_name_2, event_type_2,
        snap_CA, snap_TX, snap_WI,
        sell_price,
        is_stockout (bool)
    """
    logger.info(" Loading raw M5 files …")
    sales_df    = _load_sales(sales_path, sample_stores)
    calendar_df = _load_calendar(calendar_path)
    prices_df   = _load_prices(prices_path)

    logger.info(" Melting sales to long format …")
    long_df = _melt_sales(sales_df)

    logger.info(" Merging calendar …")
    long_df = _merge_calendar(long_df, calendar_df)

    logger.info(" Merging sell prices …")
    long_df = _merge_prices(long_df, prices_df)

    logger.info(" Detecting stockouts …")
    long_df = _detect_stockouts(long_df)

    logger.info(" Parsing hierarchy …")
    long_df = _parse_hierarchy(long_df)

    logger.info(f" Writing parquet → {output_path}")
    long_df.to_parquet(output_path, index=False)

    logger.success(f"Preprocessing complete. Shape: {long_df.shape}")
    return long_df


#  Sub-steps 

def _load_sales(path: Path, sample_stores: list[str] | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if sample_stores:
        df = df[df["store_id"].isin(sample_stores)].reset_index(drop=True)
        logger.debug(f"  Sampled {len(df):,} series for stores: {sample_stores}")
    logger.debug(f"  Sales shape: {df.shape}")
    return df


def _load_calendar(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    # Keep only columns we need
    cols = [
        "date", "wm_yr_wk", "weekday", "wday", "month", "year", "d",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]
    return df[cols]


def _load_prices(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)  # store_id, item_id, wm_yr_wk, sell_price


def _melt_sales(sales_df: pd.DataFrame, chunk_size: int = 2000) -> pd.DataFrame:
    """
    Memory-safe melt using row chunks.
    Melting all 30,490 series at once causes OOM on most machines.
    Processing in chunks of `chunk_size` rows keeps peak RAM under ~1 GB.
    """
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    d_cols  = [c for c in sales_df.columns if c.startswith("d_")]

    chunks = []
    n_chunks = int(np.ceil(len(sales_df) / chunk_size))

    for i in range(n_chunks):
        chunk = sales_df.iloc[i * chunk_size : (i + 1) * chunk_size]
        melted = chunk[id_cols + d_cols].melt(
            id_vars=id_cols,
            value_vars=d_cols,
            var_name="d",
            value_name="sales",
        )
        melted["sales"] = melted["sales"].astype(np.int32)
        for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
            melted[col] = melted[col].astype("category")
        chunks.append(melted)
        logger.debug(f"  Melted chunk {i+1}/{n_chunks}")

    long = pd.concat(chunks, ignore_index=True)
    logger.debug(f"  Long format shape: {long.shape}")
    return long


def _merge_calendar(long_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    return long_df.merge(calendar_df, on="d", how="left")


def _merge_prices(long_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    return long_df.merge(
        prices_df,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )


def _detect_stockouts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristic: a zero-sales day is a *stockout* if:
      - sell_price is non-null (item was listed) AND
      - it's surrounded by non-zero sales (rolling window check)
    We flag it with `is_stockout` but do NOT impute — models handle it.
    """
    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    # Rolling sum of sales in a 7-day window (forward + backward)
    df["_roll7"] = (
        df.groupby("id")["sales"]
        .transform(lambda x: x.rolling(7, min_periods=1, center=True).sum())
    )

    df["is_stockout"] = (
        (df["sales"] == 0) &
        (df["sell_price"].notna()) &
        (df["_roll7"] > 0)
    )
    df.drop(columns=["_roll7"], inplace=True)
    logger.debug(f"  Stockout days flagged: {df['is_stockout'].sum():,}")
    return df


def _parse_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure hierarchy columns are categorical to save memory."""
    cat_cols = ["state_id", "store_id", "cat_id", "dept_id", "item_id"]
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df


# CLI 
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="M5 Preprocessing Pipeline")
    parser.add_argument("--sample-stores", nargs="*", default=None,
                        help="e.g. CA_1 TX_2 (for fast dev iteration)")
    args = parser.parse_args()

    run_preprocessing(sample_stores=args.sample_stores)