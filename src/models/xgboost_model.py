"""
XGBoost

How it differs from LightGBM:
  - LightGBM : grows trees leaf-by-leaf  → faster on large data
  - XGBoost  : grows trees level-by-level → sometimes better on small data

On our 13M row dataset we expect LightGBM to win,
but XGBoost confirms the comparison is fair.

Probabilistic output:
  XGBoost doesn't have native quantile loss like LightGBM.
  We use the 'reg:quantileerror' objective introduced in XGBoost 1.7+
  to train separate P10 and P90 models.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger

from src.features.engineering import get_feature_columns


# Hyperparameters 
PARAMS = {
    "objective":        "reg:absoluteerror",  # MAE loss — same as LightGBM
    "n_estimators":     500,
    "learning_rate":    0.05,
    "max_depth":        6,                    # XGBoost uses max_depth, not num_leaves
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,                   # equivalent to min_child_samples
    "reg_alpha":        0.1,                  # L1
    "reg_lambda":       0.1,                  # L2
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,                    # suppress output
    "tree_method":      "hist",               # fast histogram method — similar to LightGBM
}

QUANTILES = [0.1, 0.5, 0.9]


class XGBoostDemandForecaster:
    """
    XGBoost demand forecaster with probabilistic output.

    Same interface as LGBMDemandForecaster — easy to swap in benchmarks.
    """

    def __init__(self, params: dict = None, quantiles: list = None):
        self.params           = params or PARAMS
        self.quantiles        = quantiles or QUANTILES
        self._point_model     = None
        self._quantile_models = {}
        self.feature_cols_    = []

    #  Training 

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: list = None,
        target_col: str = "sales",
        val_df: pd.DataFrame = None,
    ) -> "XGBoostDemandForecaster":
        """
        Train point model + quantile models.
        Uses xgb.DMatrix for memory efficiency — same idea as lgb.Dataset.
        """
        self.feature_cols_ = feature_cols or get_feature_columns(train_df)

        logger.info(
            f"Training XGBoost — "
            f"{len(train_df):,} rows × {len(self.feature_cols_)} features"
        )

        # DMatrix is XGBoost's memory-efficient data format
        # equivalent to lgb.Dataset in LightGBM
        dtrain = xgb.DMatrix(
            train_df[self.feature_cols_],
            label=train_df[target_col].values,
        )

        evals = [(dtrain, "train")]
        if val_df is not None:
            dval = xgb.DMatrix(
                val_df[self.feature_cols_],
                label=val_df[target_col].values,
            )
            evals.append((dval, "val"))

        n_rounds = self.params.get("n_estimators", 500)

        # Point model 
        logger.info("  Training point model (MAE loss) …")
        point_params = {k: v for k, v in self.params.items() if k != "n_estimators"}
        point_params["objective"] = "reg:absoluteerror"

        self._point_model = xgb.train(
            point_params,
            dtrain,
            num_boost_round=n_rounds,
            evals=evals,
            verbose_eval=100,
            early_stopping_rounds=30 if val_df is not None else None,
        )
        logger.success("  Point model done")

        #  Quantile models 
        for q in self.quantiles:
            logger.info(f"  Training quantile model q={q} …")
            q_params = {k: v for k, v in self.params.items() if k != "n_estimators"}
            q_params["objective"] = "reg:quantileerror"
            q_params["quantile_alpha"] = q   # XGBoost 1.7+ syntax

            self._quantile_models[q] = xgb.train(
                q_params,
                dtrain,
                num_boost_round=n_rounds,
                evals=evals,
                verbose_eval=100,
                early_stopping_rounds=30 if val_df is not None else None,
            )

        logger.success("All XGBoost models trained")
        return self

    #  Prediction 

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        dtest = xgb.DMatrix(df[self.feature_cols_])
        return np.clip(self._point_model.predict(dtest), 0, None)

    def predict_quantiles(self, df: pd.DataFrame) -> dict:
        self._check_fitted()
        dtest = xgb.DMatrix(df[self.feature_cols_])
        return {
            q: np.clip(m.predict(dtest), 0, None)
            for q, m in self._quantile_models.items()
        }

    # ── Feature importance ─────────────────────────────────────────────────────

    def feature_importance(self, top_n: int = 20) -> pd.Series:
        self._check_fitted()
        scores = self._point_model.get_fscore()   # gain-based importance
        series = pd.Series(scores)
        return series.sort_values(ascending=False).head(top_n)

    # ── Save / Load ────────────────────────────────────────────────────────────

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path) -> "XGBoostDemandForecaster":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded ← {path}")
        return model

    def _check_fitted(self):
        if self._point_model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

    def __repr__(self):
        status = "fitted" if self._point_model else "unfitted"
        return f"XGBoostDemandForecaster({status}, quantiles={self.quantiles})"


#  CLI

if __name__ == "__main__":
    from src.evaluation.metrics import evaluate, make_leaderboard
    from src.features.engineering import load_features, get_feature_columns

    PROCESSED_DIR = Path("data/processed")

    # Load same 3 stores as LightGBM for fair comparison
    logger.info("Loading features (3 stores) …")
    df_feat = load_features(stores=["CA_1", "TX_1", "WI_1"])

    # Same splits as LightGBM
    TRAIN_END = pd.Timestamp("2016-02-29")
    VAL_END   = pd.Timestamp("2016-03-27")

    train_df = df_feat[df_feat["date"] <= TRAIN_END]
    val_df   = df_feat[(df_feat["date"] > TRAIN_END) & (df_feat["date"] <= VAL_END)]
    test_df  = df_feat[df_feat["date"] > VAL_END]

    logger.info(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Train
    feature_cols = get_feature_columns(train_df)
    model = XGBoostDemandForecaster()
    model.fit(train_df, feature_cols=feature_cols, val_df=val_df)

    # Evaluate
    preds   = model.predict(test_df)
    q_preds = model.predict_quantiles(test_df)
    actual  = test_df["sales"].values

    report = evaluate(
        actual         = actual,
        point_forecast = preds,
        train_series   = train_df["sales"].values,
        q10            = q_preds[0.1],
        q90            = q_preds[0.9],
        model_name     = "XGBoost",
    )

    # Compare with LightGBM results
    lgbm_report = {
        "model": "LightGBM",
        "MAE":   0.851,
        "MASE":  0.842,
        "Coverage_80%": 0.889,
    }

    print("\n── Benchmark Leaderboard ──")
    print(make_leaderboard([lgbm_report, report]).to_string())

    model.save(PROCESSED_DIR / "xgboost_model.pkl")