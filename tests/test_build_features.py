"""
test_build_features.py
======================
Unit tests for src/features/build_features.py

Checks:
- Log return computation
- Lagged return look-ahead safety
- Target construction (classification and regression)
- Rolling feature validity
"""

import pytest
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "..")

from src.features.build_features import (
    compute_log_returns,
    add_lagged_returns,
    add_rolling_volatility,
    add_momentum,
    add_mean_reversion,
    add_target,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_price_df(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic price DataFrame."""
    rng = np.random.default_rng(seed)
    closes = 10000 * np.cumprod(1 + rng.normal(0, 0.01, n))
    idx = pd.date_range("2018-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "open":  closes * 0.999,
        "high":  closes * 1.005,
        "low":   closes * 0.995,
        "close": closes,
        "volume": np.nan,
    }, index=idx)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLogReturns:

    def test_length_matches_input(self):
        df = make_price_df()
        ret = compute_log_returns(df)
        assert len(ret) == len(df)

    def test_first_return_is_nan(self):
        df = make_price_df()
        ret = compute_log_returns(df)
        assert np.isnan(ret.iloc[0])

    def test_returns_approximately_zero_mean(self):
        df = make_price_df(n=1000)
        ret = compute_log_returns(df).dropna()
        # Log returns of random walk should be close to 0
        assert abs(ret.mean()) < 0.01


class TestLaggedReturns:

    def test_ret_lag_1_is_look_ahead_safe(self):
        """
        ret_lag_1 at position t should equal the return from t-2 to t-1.
        Verify by comparing to compute_log_returns shifted by 1.
        """
        df = make_price_df()
        df_feat = add_lagged_returns(df.copy(), lags=[1])
        ret = compute_log_returns(df)
        expected_lag1 = ret.shift(1)
        pd.testing.assert_series_equal(
            df_feat["ret_lag_1"].dropna(),
            expected_lag1.dropna(),
            check_names=False,
        )

    def test_multiple_lags_created(self):
        df = make_price_df()
        df_feat = add_lagged_returns(df.copy(), lags=[1, 2, 5])
        for lag in [1, 2, 5]:
            assert f"ret_lag_{lag}" in df_feat.columns


class TestRollingVolatility:

    def test_vol_columns_created(self):
        df = make_price_df()
        df_feat = add_rolling_volatility(df.copy(), windows=[5, 21])
        assert "vol_5d" in df_feat.columns
        assert "vol_21d" in df_feat.columns

    def test_vol_is_positive_where_not_nan(self):
        df = make_price_df()
        df_feat = add_rolling_volatility(df.copy(), windows=[5])
        valid = df_feat["vol_5d"].dropna()
        assert (valid > 0).all()


class TestTarget:

    def test_classification_target_is_binary(self):
        df = make_price_df()
        df_t = add_target(df.copy(), horizon=1, task_type="classification")
        valid = df_t["target"].dropna()
        assert set(valid.unique()).issubset({1, -1})

    def test_regression_target_last_rows_are_nan(self):
        df = make_price_df(n=100)
        horizon = 3
        df_t = add_target(df.copy(), horizon=horizon, task_type="regression")
        assert df_t["target"].iloc[-horizon:].isna().all()

    def test_classification_no_lookahead_via_count(self):
        """Target at position t-1 should not depend on close at t."""
        df = make_price_df(n=50)
        df_t = add_target(df.copy(), horizon=1, task_type="classification")
        # Target at index i should reflect close[i+1] vs close[i]
        for i in range(1, len(df) - 1):
            expected_sign = 1 if df["close"].iloc[i + 1] > df["close"].iloc[i] else -1
            assert df_t["target"].iloc[i] == expected_sign
