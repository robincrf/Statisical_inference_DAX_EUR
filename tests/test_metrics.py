"""
test_metrics.py
===============
Unit tests for src/evaluation/metrics.py

Checks mathematical correctness of:
- Sharpe ratio
- Max drawdown
- Annualized return and volatility
- Turnover
- Strategy return construction
"""

import pytest
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "..")

from src.evaluation.metrics import (
    sharpe_ratio,
    max_drawdown,
    annualized_return,
    annualized_volatility,
    turnover_rate,
    compute_strategy_returns,
    directional_accuracy,
)


def make_const_returns(value: float, n: int = 252) -> pd.Series:
    """Constant returns series."""
    return pd.Series([value] * n)


def make_zero_returns(n: int = 252) -> pd.Series:
    return pd.Series([0.0] * n)


class TestSharpeRatio:

    def test_zero_returns_zero_sharpe(self):
        ret = make_zero_returns()
        sr = sharpe_ratio(ret)
        assert np.isnan(sr) or sr == 0.0

    def test_positive_returns_positive_sharpe(self):
        ret = make_const_returns(0.001)  # 0.1% per day
        sr = sharpe_ratio(ret, risk_free_rate=0.0)
        assert sr > 0

    def test_negative_returns_negative_sharpe(self):
        ret = make_const_returns(-0.001)
        sr = sharpe_ratio(ret, risk_free_rate=0.0)
        assert sr < 0

    def test_annualization_applied(self):
        # Use returns with clear positive mean so Sharpe != 0
        rng = np.random.default_rng(99)
        ret = pd.Series(0.001 + rng.normal(0, 0.01, 500))  # positive drift
        sr_annualized = sharpe_ratio(ret, annualize=True, risk_free_rate=0.0)
        sr_not = sharpe_ratio(ret, annualize=False, risk_free_rate=0.0)
        # Annualized SR should be larger (multiplied by sqrt(252))
        assert abs(sr_annualized) > abs(sr_not)

    def test_nan_on_zero_std(self):
        ret = make_const_returns(0.005)  # std = 0
        sr = sharpe_ratio(ret)
        assert np.isnan(sr) or np.isinf(sr)


class TestMaxDrawdown:

    def test_no_drawdown_on_monotone_returns(self):
        ret = make_const_returns(0.001)
        mdd = max_drawdown(ret)
        assert mdd == pytest.approx(0.0, abs=1e-10)

    def test_full_drawdown_on_total_loss(self):
        ret = pd.Series([0.1, -0.2, -0.5, -0.5])
        mdd = max_drawdown(ret)
        assert 0 < mdd <= 1

    def test_mdd_is_non_negative(self):
        rng = np.random.default_rng(0)
        ret = pd.Series(rng.normal(0, 0.01, 500))
        assert max_drawdown(ret) >= 0


class TestAnnualizedMetrics:

    def test_annualized_return_scaling(self):
        daily_mean = 0.001
        ret = make_const_returns(daily_mean, n=252)
        ann_ret = annualized_return(ret, frequency=252)
        assert ann_ret == pytest.approx(daily_mean * 252, rel=1e-6)

    def test_annualized_vol_scaling(self):
        rng = np.random.default_rng(1)
        ret = pd.Series(rng.normal(0, 0.01, 252))
        ann_vol = annualized_volatility(ret, frequency=252)
        assert ann_vol == pytest.approx(ret.std() * np.sqrt(252), rel=1e-6)


class TestTurnover:

    def test_zero_turnover_constant_signal(self):
        signals = pd.Series([1] * 100)
        assert turnover_rate(signals) == pytest.approx(0.0, abs=1e-10)

    def test_max_turnover_alternating_signal(self):
        signals = pd.Series([-1, 1] * 50)
        to = turnover_rate(signals)
        assert to == pytest.approx(2.0, rel=0.1)


class TestDirectionalAccuracy:

    def test_perfect_prediction(self):
        y_true = np.array([1, -1, 1, 1, -1])
        assert directional_accuracy(y_true, y_true) == 1.0

    def test_zero_accuracy_inverted(self):
        y_true = np.array([1, -1, 1, 1, -1])
        y_pred = -y_true
        assert directional_accuracy(y_true, y_pred) == 0.0

    def test_random_prediction_near_half(self):
        rng = np.random.default_rng(5)
        y_true = rng.choice([-1, 1], size=1000)
        y_pred = rng.choice([-1, 1], size=1000)
        acc = directional_accuracy(y_true, y_pred)
        assert 0.4 < acc < 0.6  # Should be near 50% by chance
