"""
Hyperparameter optimization for LightGBM using Optuna.

What this does:
  Runs 50 trials of LightGBM with different hyperparameters.
  Each trial Optuna suggests smarter values based on previous results.
  At the end, prints the best parameters found.

How to use:
  1. Run this script → finds best params (takes ~1-2 hours)
  2. Copy best params into lgbm_model.py → retrain → better MASE

Run:
  python -m src.models.optuna_tuning
"""


import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from loguru import logger

from src.evaluation.metrics import mase
from src.features.engineering import get_feature_columns, load_features

# Silence Optuna's own logs — we'll print what matters ourselves
optuna.logging.set_verbosity(optuna.logging.WARNING)


# Data — load once, reuse across all trials

def load_data() -> tuple:
    """
    Load train/val/test splits.
    We use only 3 stores — same as lgbm_model.py — for speed.
    """
    logger.info("Loading features (3 stores) …")
    df = load_features(stores=["CA_1", "TX_1", "WI_1"])

    TRAIN_END = pd.Timestamp("2016-02-29")
    VAL_END   = pd.Timestamp("2016-03-27")

    train_df = df[df["date"] <= TRAIN_END]
    val_df   = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)]
    test_df  = df[df["date"] > VAL_END]

    feature_cols = get_feature_columns(train_df)

    logger.info(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    return train_df, val_df, test_df, feature_cols


# Objective function — called once per trial

def objective(
    trial: optuna.Trial,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
) -> float:
    """
    One Optuna trial = one LightGBM training run.

    Optuna calls this function 50 times with different params.
    Each time we return the validation MASE.
    Optuna tries to minimize it.

    Parameters
    ----------
    trial      : Optuna trial object — suggests hyperparameter values
    train_df   : training data
    val_df     : validation data (used to score each trial)
    feature_cols: list of feature column names

    Returns
    -------
    float : validation MASE (lower = better)
    """

    #  Optuna suggests values within the ranges we define 
    params = {
        # How fast the model learns
        # Small = slower but more accurate, Large = faster but may overfit
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),

        # Number of leaves per tree — controls complexity
        # More leaves = more complex model = risk of overfitting
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),

        # Maximum tree depth — another complexity control
        "max_depth": trial.suggest_int("max_depth", 3, 10),

        # Fraction of rows used per tree — adds randomness, prevents overfitting
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),

        # Fraction of features used per tree — same idea
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),

        # Minimum samples required in a leaf — prevents overfitting on rare items
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),

        # Regularization — penalises large weights to prevent overfitting
        "reg_alpha":  trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),

        # Fixed settings — not tuned
        "objective":  "regression_l1",
        "n_jobs":     -1,
        "verbose":    -1,
    }

    #  Train LightGBM with these params 
    dtrain = lgb.Dataset(
        train_df[feature_cols],
        label=train_df["sales"].values,
        free_raw_data=True,
    )
    dval = lgb.Dataset(
        val_df[feature_cols],
        label=val_df["sales"].values,
        reference=dtrain,
        free_raw_data=True,
    )

    # Use early stopping so bad trials stop quickly
    # This saves time — a bad config fails fast instead of running all 500 rounds
    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=9999),   # suppress per-round output
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    #  Score on validation set 
    val_preds   = np.clip(model.predict(val_df[feature_cols]), 0, None)
    val_actual  = val_df["sales"].values
    train_sales = train_df["sales"].values

    val_mase = mase(val_actual, val_preds, train_sales)

    return val_mase


# Run the study

def run_tuning(n_trials: int = 50) -> dict:
    """
    Run Optuna hyperparameter search.

    Parameters
    ----------
    n_trials : int
        Number of trials to run. More = better params but slower.
        50 is a good balance for portfolio purposes.
        Rule of thumb: 10 trials per hyperparameter being tuned.

    Returns
    -------
    dict : best hyperparameters found
    """
    # Load data once — shared across all trials
    train_df, val_df, test_df, feature_cols = load_data()

    logger.info(f"▶ Starting Optuna study — {n_trials} trials …")
    logger.info("  Each trial trains one LightGBM model and returns val MASE.")
    logger.info("  This will take approximately 1-2 hours.")

    # Create study — direction="minimize" because we want lowest MASE
    study = optuna.create_study(
        direction="minimize",
        study_name="lgbm_demand_forecast",
        # TPE = Tree-structured Parzen Estimator — Optuna's default smart sampler
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Run trials — lambda passes the data into the objective function
    study.optimize(
        lambda trial: objective(trial, train_df, val_df, feature_cols),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    #  Results 
    best_params = study.best_params
    best_mase   = study.best_value

    logger.success("✔ Tuning complete!")
    logger.success(f"  Best val MASE : {best_mase:.4f}")
    logger.success(f"  Best params   : {best_params}")

    # Trial history ─────────────────────────────────────────────────────────
    print("\n── Top 5 Trials ──")
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values("value").head(5)
    print(trials_df[["number", "value"] + [c for c in trials_df.columns if c.startswith("params_")]].to_string())

    # ── Best params formatted for copy-paste into lgbm_model.py ──────────────
    print("\n── Copy these into lgbm_model.py PARAMS dict ──")
    print("PARAMS = {")
    for k, v in best_params.items():
        if isinstance(v, float):
            print(f'    "{k}": {v:.6f},')
        else:
            print(f'    "{k}": {v},')
    print('    "objective": "regression_l1",')
    print('    "n_jobs": -1,')
    print('    "verbose": -1,')
    print("}")

    return best_params


#CLI

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials (default: 50)"
    )
    args = parser.parse_args()

    best = run_tuning(n_trials=args.n_trials)

