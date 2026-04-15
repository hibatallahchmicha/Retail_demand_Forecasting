"""
Temporal Fusion Transformer (TFT) 
"""

from pathlib import Path

import pandas as pd
from loguru import logger



# Data preparation for TFT


def prepare_tft_dataset(
    df: pd.DataFrame,
    max_encoder_length: int = 56,
    max_prediction_length: int = 28,
):
    """
    Convert long-format DataFrame into pytorch-forecasting TimeSeriesDataSet.

    Key parameters informed by our EDA:
      max_encoder_length  = 56  (ACF significant up to lag-56)
      max_prediction_length = 28 (M5 forecast horizon)

    Parameters
    df                    : long-format DataFrame with all features
    max_encoder_length    : how many past days TFT looks at
    max_prediction_length : how many days ahead to forecast

    Returns
    pytorch_forecasting.TimeSeriesDataSet
    """
    try:
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
    except ImportError:
        raise ImportError(
            "pytorch-forecasting not installed.\n"
            "Run: pip install pytorch-forecasting pytorch-lightning torch"
        )

    # TFT needs a time index , integer days from start
    df = df.copy()
    df["time_idx"] = (
        (df["date"] - df["date"].min()).dt.days
    ).astype(int)

    # Features split into categories TFT understands:
    # time_varying_known   = features we know for future dates (calendar, events)
    # time_varying_unknown = features only known up to today (sales, lags)
    # static               = features that don't change per series (store, item)

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="sales",
        group_ids=["id"],                        # one series per item×store

        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,

        # Known in the future — calendar features
        time_varying_known_reals=[
            "dow_sin", "dow_cos",
            "month_sin", "month_cos",
            "is_weekend", "is_month_start", "is_month_end",
            "has_event", "is_snap_day",
            "log_price",
        ],

        # Unknown in the future — sales history and derived features
        time_varying_unknown_reals=[
            "sales",
            "sales_lag_7", "sales_lag_14", "sales_lag_28",
            "sales_roll_mean_7", "sales_roll_mean_28",
            "sales_roll_std_7", "sales_roll_std_28",
            "price_momentum", "price_vs_dept_mean",
        ],

        # Static — doesn't change over time for a series
        static_categoricals=["store_id", "cat_id", "dept_id", "state_id"],
        static_reals=["item_store_enc", "dept_store_enc"],

        # Normalize target per series — handles different sales scales
        target_normalizer=GroupNormalizer(groups=["id"], transformation="softplus"),

        # Quantile output
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    return dataset


#
# TFT Model
#

def build_tft_model(dataset):
    """
    Build the TFT model from a TimeSeriesDataSet.

    Parameters informed by M5 competition winners and TFT paper.
    """
    try:
        from pytorch_forecasting import TemporalFusionTransformer
        from pytorch_forecasting.metrics import QuantileLoss
    except ImportError:
        raise ImportError(
            "pytorch-forecasting not installed.\n"
            "Run: pip install pytorch-forecasting pytorch-lightning torch"
        )

    model = TemporalFusionTransformer.from_dataset(
        dataset,

        # Learning
        learning_rate=0.03,

        # Model size — bigger = more powerful but slower
        hidden_size=64,               # main hidden layer size
        attention_head_size=4,        # number of attention heads
        hidden_continuous_size=32,    # size of continuous variable processing

        # Regularization
        dropout=0.1,                  # prevents overfitting

        # Output — quantile loss for P10, P50, P90
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),

        # Logging
        log_interval=10,
        log_val_interval=1,
        reduce_on_plateau_patience=4,
    )

    logger.info(f"TFT model parameters: {model.size() / 1e3:.1f}k")
    return model 

# Training

def train_tft(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    max_epochs: int = 50,
    batch_size: int = 128,
    output_dir: Path = Path("data/processed"),
):
    """
    Train the TFT model.
    Parameters

    train_df    : training DataFrame
    val_df      : validation DataFrame
    max_epochs  : training epochs (50 is standard)
    batch_size  : sequences per batch (reduce if OOM on GPU)
    output_dir  : where to save the trained model
    """
    try:
        import pytorch_lightning as pl  # noqa: F401
        from torch.utils.data import DataLoader  # noqa: F401
    except ImportError:
        raise ImportError(
            "pytorch-lightning not installed.\n"
            "Run: pip install pytorch-forecasting pytorch-lightning torch"
        )

    import torch
    if not torch.cuda.is_available():
        logger.warning(
            "⚠️  No GPU detected. Training on CPU will take 8-15 hours.\n"
            "    Consider using Google Colab (free GPU) or Kaggle Notebooks."
        )

    # Prepare datasets
    logger.info("Preparing TFT datasets …")
    train_dataset = prepare_tft_dataset(train_df)

    from pytorch_forecasting import TimeSeriesDataSet
    val_dataset   = TimeSeriesDataSet.from_dataset(
        train_dataset, val_df, predict=True, stop_randomization=True
    )

    # DataLoaders
    train_loader = train_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_loader = val_dataset.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )

    # Build model
    model = build_tft_model(train_dataset)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=0.1,       # prevents exploding gradients
        enable_progress_bar=True,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,          # stop if val loss doesn't improve for 5 epochs
                mode="min",
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath=output_dir / "tft_checkpoints",
                monitor="val_loss",
                save_top_k=1,        # keep only the best checkpoint
            ),
        ],
    )

    logger.info(f"Starting TFT training — {max_epochs} epochs …")
    trainer.fit(model, train_loader, val_loader)
    logger.success("✔ TFT training complete")

    return model, trainer


# CLI

if __name__ == "__main__":
    import torch

    if not torch.cuda.is_available():
        print("\n" + "="*60)
        print(" NO GPU DETECTED")
        print("="*60)
        print("\nTFT training requires a GPU for practical training times.")
        print("\nOptions:")
        print("  1. Google Colab  → free GPU (T4), ~90 mins")
        print("     https://colab.research.google.com")
        print("  2. Kaggle Notebooks → free GPU, ~90 mins")
        print("     https://www.kaggle.com/code")
        print("  3. Local GPU → if you have an NVIDIA card")
        print("\nThis code is fully implemented and ready to run.")
        print("Copy this file to Colab and run train_tft() there.")
        print("="*60)
    else:
        from src.features.engineering import load_features

        logger.info("GPU detected — starting TFT training …")
        df = load_features(stores=["CA_1", "TX_1", "WI_1"])

        TRAIN_END = pd.Timestamp("2016-02-29")
        VAL_END   = pd.Timestamp("2016-03-27")

        train_df = df[df["date"] <= TRAIN_END]
        val_df   = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)]

        model, trainer = train_tft(train_df, val_df, max_epochs=50)