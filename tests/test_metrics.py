"""Tests for evaluation metrics."""
import numpy as np
import pytest

from src.evaluation.metrics import mae, mase, coverage, wrmsse, evaluate


class TestMAE:
    def test_perfect_forecast(self):
        actual = np.array([10, 20, 30])
        assert mae(actual, actual) == 0.0

    def test_known_values(self):
        actual = np.array([10, 20, 15])
        forecast = np.array([8, 25, 14])
        assert mae(actual, forecast) == pytest.approx(8 / 3, abs=1e-6)

    def test_symmetric(self):
        actual = np.array([10.0, 20.0])
        forecast = np.array([15.0, 25.0])
        assert mae(actual, forecast) == mae(forecast, actual)


class TestMASE:
    def test_naive_equals_one(self):
        """A naive seasonal forecast on noisy data should score ~1.0."""
        np.random.seed(42)
        base_pattern = np.array([5, 3, 7, 2, 8, 4, 6])
        # Add noise so naive forecast isn't perfect
        train = np.tile(base_pattern, 10).astype(float) + np.random.normal(0, 1, 70)
        # Test set: same pattern with noise
        actual = base_pattern.astype(float) + np.random.normal(0, 1, 7)
        # Naive forecast: copy the last week from training
        forecast = train[-7:]
        result = mase(actual, forecast, train, seasonality=7)
        # Should be in the ballpark of 1.0 (not exactly, due to noise)
        assert 0.3 < result < 3.0

    def test_perfect_beats_naive(self):
        train = np.array([5, 3, 7, 2, 8, 4, 6] * 10)
        actual = np.array([5, 3, 7, 2, 8, 4, 6])
        result = mase(actual, actual, train, seasonality=7)
        assert result == 0.0

    def test_worse_than_naive(self):
        train = np.array([5, 3, 7, 2, 8, 4, 6] * 10)
        actual = np.array([5, 3, 7, 2, 8, 4, 6])
        bad_forecast = np.array([50, 30, 70, 20, 80, 40, 60])
        result = mase(actual, bad_forecast, train, seasonality=7)
        assert result > 1.0


class TestCoverage:
    def test_all_inside(self):
        actual = np.array([5, 10, 15])
        q10 = np.array([0, 5, 10])
        q90 = np.array([10, 15, 20])
        assert coverage(actual, q10, q90) == 1.0

    def test_none_inside(self):
        actual = np.array([100, 200, 300])
        q10 = np.array([0, 0, 0])
        q90 = np.array([1, 1, 1])
        assert coverage(actual, q10, q90) == 0.0

    def test_partial_coverage(self):
        actual = np.array([5, 100, 15, 200])
        q10 = np.array([0, 0, 10, 0])
        q90 = np.array([10, 10, 20, 10])
        assert coverage(actual, q10, q90) == 0.5


class TestWRMSSE:
    def test_perfect_forecast(self):
        actuals = np.array([[10, 20, 30]]).astype(float)
        train = np.tile(np.array([10, 20, 30]), (1, 3))
        result = wrmsse(actuals, actuals, train)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_nonzero_error(self):
        actuals = np.array([[10.0, 20.0, 30.0]])
        forecasts = np.array([[15.0, 25.0, 35.0]])
        train = np.tile(np.array([10.0, 20.0, 30.0]), (1, 3))
        result = wrmsse(actuals, forecasts, train)
        assert result > 0.0


class TestEvaluate:
    def test_returns_all_keys(self):
        actual = np.array([10.0, 20.0, 30.0])
        forecast = np.array([11.0, 19.0, 31.0])
        train = np.array([5, 3, 7, 2, 8, 4, 6] * 10, dtype=float)
        q10 = forecast - 5
        q90 = forecast + 5

        report = evaluate(actual, forecast, train, q10, q90, model_name="test")
        assert "model" in report
        assert "MAE" in report
        assert "MASE" in report
        assert "Coverage_80%" in report
        assert report["model"] == "test"

    def test_without_quantiles(self):
        actual = np.array([10.0, 20.0, 30.0])
        forecast = np.array([11.0, 19.0, 31.0])
        train = np.array([5, 3, 7, 2, 8, 4, 6] * 10, dtype=float)

        report = evaluate(actual, forecast, train, model_name="test")
        assert "Coverage_80%" not in report
