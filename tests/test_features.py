"""Tests for feature engineering utilities."""
import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    _add_lag_features,
    _add_rolling_features,
    _add_calendar_features,
    _add_event_features,
    _add_snap_features,
    get_feature_columns,
)


@pytest.fixture
def sample_df():
    """Create a minimal DataFrame mimicking the M5 long format."""
    n_days = 70  # enough for lag-56
    dates = pd.date_range("2012-01-01", periods=n_days, freq="D")
    np.random.seed(42)
    df = pd.DataFrame({
        "id": "FOODS_1_001_CA_1_evaluation",
        "item_id": "FOODS_1_001",
        "dept_id": "FOODS_1",
        "cat_id": "FOODS",
        "store_id": "CA_1",
        "state_id": "CA",
        "date": dates,
        "sales": np.random.poisson(5, n_days).astype(float),
        "sell_price": np.random.uniform(1, 10, n_days).astype(np.float32),
        "event_name_1": [None] * n_days,
        "event_name_2": [None] * n_days,
        "event_type_1": [None] * n_days,
        "event_type_2": [None] * n_days,
        "snap_CA": np.random.randint(0, 2, n_days),
        "snap_TX": 0,
        "snap_WI": 0,
        "weekday": [d.strftime("%A") for d in dates],
    })
    # Add some events
    df.loc[10, "event_name_1"] = "SuperBowl"
    df.loc[10, "event_type_1"] = "Sporting"
    df.loc[20, "event_name_1"] = "Easter"
    df.loc[20, "event_type_1"] = "Religious"
    df.loc[20, "event_name_2"] = "OrthodoxEaster"
    df.loc[20, "event_type_2"] = "Religious"
    return df


class TestLagFeatures:
    def test_creates_lag_columns(self, sample_df):
        result = _add_lag_features(sample_df.copy())
        for lag in [7, 14, 21, 28, 56]:
            assert f"sales_lag_{lag}" in result.columns

    def test_lag_values_correct(self, sample_df):
        result = _add_lag_features(sample_df.copy())
        # lag-7 at row 10 should equal sales at row 3
        assert result.loc[10, "sales_lag_7"] == sample_df.loc[3, "sales"]


class TestRollingFeatures:
    def test_creates_rolling_columns(self, sample_df):
        result = _add_rolling_features(sample_df.copy())
        for window in [7, 28, 56]:
            assert f"sales_roll_mean_{window}" in result.columns
            assert f"sales_roll_std_{window}" in result.columns


class TestCalendarFeatures:
    def test_creates_calendar_columns(self, sample_df):
        result = _add_calendar_features(sample_df.copy())
        expected = ["day_of_week", "month", "year", "is_weekend",
                     "dow_sin", "dow_cos", "month_sin", "month_cos"]
        for col in expected:
            assert col in result.columns

    def test_weekend_flag(self, sample_df):
        result = _add_calendar_features(sample_df.copy())
        # Check that weekends are flagged correctly
        saturdays = result[result["date"].dt.dayofweek == 5]
        if len(saturdays) > 0:
            assert (saturdays["is_weekend"] == 1).all()


class TestEventFeatures:
    def test_creates_event_columns(self, sample_df):
        result = _add_event_features(sample_df.copy())
        assert "has_event" in result.columns
        assert "event_type_1_enc" in result.columns

    def test_event_detection(self, sample_df):
        result = _add_event_features(sample_df.copy())
        assert result.loc[10, "has_event"] == 1
        assert result.loc[0, "has_event"] == 0

    def test_two_events(self, sample_df):
        result = _add_event_features(sample_df.copy())
        assert result.loc[20, "has_two_events"] == 1
        assert result.loc[10, "has_two_events"] == 0


class TestSnapFeatures:
    def test_snap_for_ca(self, sample_df):
        result = _add_snap_features(sample_df.copy())
        assert "is_snap_day" in result.columns
        # CA store should use snap_CA values
        assert (result["is_snap_day"] == result["snap_CA"]).all()


class TestGetFeatureColumns:
    def test_excludes_identifiers(self, sample_df):
        df = _add_lag_features(sample_df.copy())
        df = _add_calendar_features(df)
        cols = get_feature_columns(df)
        assert "id" not in cols
        assert "item_id" not in cols
        assert "sales" not in cols
        assert "date" not in cols

    def test_includes_features(self, sample_df):
        df = _add_lag_features(sample_df.copy())
        df = _add_calendar_features(df)
        cols = get_feature_columns(df)
        assert "sales_lag_7" in cols
        assert "day_of_week" in cols
