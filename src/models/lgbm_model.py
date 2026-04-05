"""
src/models/lgbm_model.py
─────────────────────────
LightGBM — Global Gradient Boosting Model for Demand Forecasting

Strategy: Direct multi-step forecasting
  We predict all 28 horizon days at once using lag features
  that are already known at prediction time (lag_28 and beyond).
  This avoids the error accumulation of recursive forecasting.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from loguru import logger

from src.features.engineering import get_feature_columns


# ── Hyperparameters ────────────────────────────────────────────────────────────
PARAMS = {
    "objective":         "regression_l1",   # MAE loss — robust to outliers
    "num_leaves":        63,                # controls model complexity
    "learning_rate":     0.05,              # small = more trees, better fit
    "n_estimators":      500,               # number of boosting rounds
    "subsample":         0.8,              # use 80% of rows per tree
    "colsample_bytree":  0.8,              # use 80% of features per tree
    "min_child_samples": 20,               # min samples per leaf
    "reg_alpha":         0.1,              # L1 regularization
    "reg_lambda":        0.1,              # L2 regularization
    "random_state":      42,
    "n_jobs":            -1,               # use all CPU cores
    "verbose":           -1,               # suppress output
}

QUANTILES = [0.1, 0.5, 0.9]


class LGBMDemandForecaster:
    """
    Global LightGBM demand forecaster with probabilistic output.

    Trains 3 models total:
      - Point model  (L1 loss)      → best estimate of actual sales
      - Q10 model    (quantile=0.1) → lower bound of prediction interval
      - Q90 model    (quantile=0.9) → upper bound of prediction interval
    """

    def __init__(self, params: dict = None, quantiles: list = None):
        self.params           = params or PARAMS
        self.quantiles        = quantiles or QUANTILES
        self._point_model     = None
        self._quantile_models = {}
        self.feature_cols_    = []

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: list = None,
        target_col: str = "sales",
        val_df: pd.DataFrame = None,
    ) -> "LGBMDemandForecaster":

        self.feature_cols_ = feature_cols or get_feature_columns(train_df)

        logger.info(
            f"Training LightGBM — "
            f"{len(train_df):,} rows × {len(self.feature_cols_)} features"
        )

        # lgb.Dataset is memory-efficient — avoids converting to numpy array
        dtrain = lgb.Dataset(
            train_df[self.feature_cols_],
            label=train_df[target_col].values,
            free_raw_data=True,
        )

        valid_sets  = [dtrain]
        valid_names = ["train"]

        if val_df is not None:
            dval = lgb.Dataset(
                val_df[self.feature_cols_],
                label=val_df[target_col].values,
                reference=dtrain,
                free_raw_data=True,
            )
            valid_sets.append(dval)
            valid_names.append("val")

        callbacks = [lgb.log_evaluation(period=100)]
        if val_df is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=False))

        # ── Point model ────────────────────────────────────────────────────────
        logger.info("  Training point model (L1 loss) …")
        point_params = {**self.params, "objective": "regression_l1"}
        point_params.pop("n_estimators", None)

        self._point_model = lgb.train(
            point_params,
            dtrain,
            num_boost_round=self.params.get("n_estimators", 500),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        logger.success("  ✔ Point model done")

        # ── Quantile models ────────────────────────────────────────────────────
        for q in self.quantiles:
            logger.info(f"  Training quantile model q={q} …")
            q_params = {**self.params, "objective": "quantile", "alpha": q}
            q_params.pop("n_estimators", None)

            self._quantile_models[q] = lgb.train(
                q_params,
                dtrain,
                num_boost_round=self.params.get("n_estimators", 500),
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )

        logger.success("✔ All LightGBM models trained")
        return self

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return np.clip(
            self._point_model.predict(df[self.feature_cols_]),
            0, None,
        )

    def predict_quantiles(self, df: pd.DataFrame) -> dict:
        self._check_fitted()
        return {
            q: np.clip(m.predict(df[self.feature_cols_]), 0, None)
            for q, m in self._quantile_models.items()
        }

    # ── Feature importance ─────────────────────────────────────────────────────

    def feature_importance(self, top_n: int = 20) -> pd.Series:
        self._check_fitted()
        importance = self._point_model.feature_importance(importance_type="gain")
        series = pd.Series(importance, index=self._point_model.feature_name())
        return series.sort_values(ascending=False).head(top_n)

    # ── Save / Load ────────────────────────────────────────────────────────────

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path) -> "LGBMDemandForecaster":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded ← {path}")
        return model

    # ── Helper ─────────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if self._point_model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

    def __repr__(self):
        status = "fitted" if self._point_model else "unfitted"
        return f"LGBMDemandForecaster({status}, quantiles={self.quantiles})"


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.evaluation.metrics import evaluate
    from src.features.engineering import load_features, build_and_save_features, get_feature_columns

    PROCESSED_DIR = Path("data/processed")

    # Step 1 — Build features if not already done
    if not Path("data/processed/features").exists():
        logger.info("Features not found — building now …")
        df_raw = pd.read_parquet(PROCESSED_DIR / "m5_long.parquet")
        build_and_save_features(df_raw, is_train=True)

    # Step 2 — Load features for 3 stores only (one per state)
    # This gives ~13M rows — manageable on most machines
    # Once you confirm it works, you can add more stores
    logger.info("Loading features (3 stores) …")
    df_feat = load_features(stores=["CA_1", "TX_1", "WI_1"])

    # Step 3 — Train / val / test split
    TRAIN_END = pd.Timestamp("2016-02-29")
    VAL_END   = pd.Timestamp("2016-03-27")

    train_df = df_feat[df_feat["date"] <= TRAIN_END]
    val_df   = df_feat[(df_feat["date"] > TRAIN_END) & (df_feat["date"] <= VAL_END)]
    test_df  = df_feat[df_feat["date"] > VAL_END]

    logger.info(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Step 4 — Train
    feature_cols = get_feature_columns(train_df)
    model = LGBMDemandForecaster()
    model.fit(train_df, feature_cols=feature_cols, val_df=val_df)

    # Step 5 — Evaluate
    preds   = model.predict(test_df)
    q_preds = model.predict_quantiles(test_df)
    actual  = test_df["sales"].values

    report = evaluate(
        actual         = actual,
        point_forecast = preds,
        train_series   = train_df["sales"].values,
        q10            = q_preds[0.1],
        q90            = q_preds[0.9],
        model_name     = "LightGBM",
    )

    print("\n── Top 10 Features ──")
    print(model.feature_importance(top_n=10).to_string())

    print("\n── LightGBM Results ──")
    for k, v in report.items():
        print(f"  {k:<15} {v}")

    model.save(PROCESSED_DIR / "lgbm_model.pkl")