"""
metrics.py
==========
Performance evaluation metrics for the DAX/EURO STOXX Futures research project.

This module provides:
  - Standard predictive metrics (accuracy, precision, recall, F1)
  - Financial performance metrics (Sharpe ratio, Calmar ratio, max drawdown)
  - Trading-specific metrics (hit ratio, turnover, annualized return/volatility)

All financial metrics are computed from a PnL series, not individual predictions.
No data leakage risk here since metrics are pure functions of observed outcomes.

Usage:
    from src.evaluation.metrics import compute_all_metrics, sharpe_ratio
    metrics = compute_all_metrics(returns_series, signals_series, y_true, y_pred)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Predictive metrics
# ---------------------------------------------------------------------------

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraction of correctly predicted directions.

    Parameters
    ----------
    y_true : np.ndarray
        True labels ∈ {-1, 1}.
    y_pred : np.ndarray
        Predicted labels ∈ {-1, 1}.

    Returns
    -------
    float
        Hit ratio in [0, 1].
    """
    return float(np.mean(y_true == y_pred))


def compute_classification_report(y_true, y_pred) -> dict:
    """
    Compute standard classification metrics.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted labels.

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Return series construction
# ---------------------------------------------------------------------------

def compute_strategy_returns(
    asset_returns: pd.Series,
    signals: pd.Series,
    cost_per_trade: float = 0.0,
) -> pd.Series:
    """
    Compute strategy PnL as a series of daily returns.

    PnL_t = signal_{t-1} * return_t - |Δsignal_t| * cost_per_trade

    The signal must already be lagged by 1 period (enforced in predict.py).
    This function uses the signal as-is.

    Parameters
    ----------
    asset_returns : pd.Series
        Log returns of the underlying asset (aligned to same index as signals).
    signals : pd.Series
        Trading signals ∈ {-1, 0, 1}. Must be already lagged appropriately.
    cost_per_trade : float
        Cost in return units per unit change in position.
        Example: 0.0002 ≈ 2 basis points round-trip.

    Returns
    -------
    pd.Series
        Strategy log returns (approximate).
    """
    # Align
    aligned = pd.DataFrame({"ret": asset_returns, "sig": signals}).dropna()

    gross_pnl = aligned["sig"] * aligned["ret"]
    turnover = aligned["sig"].diff().abs()
    transaction_costs = turnover * cost_per_trade

    net_pnl = gross_pnl - transaction_costs
    net_pnl.name = "strategy_return"
    return net_pnl


# ---------------------------------------------------------------------------
# Financial metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
    frequency: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Compute the annualized Sharpe ratio of a return series.

    Sharpe = (mean(r) - rf) / std(r) * sqrt(frequency)

    Parameters
    ----------
    returns : pd.Series
        Daily (or periodic) strategy returns.
    risk_free_rate : float
        Annualized risk-free rate (e.g., 0.03 for 3%). Converted to per-period.
    annualize : bool
        If True, multiply by sqrt(frequency).
    frequency : int
        Number of periods per year for annualization.

    Returns
    -------
    float
        Sharpe ratio. Returns NaN if std is zero.
    """
    rf_per_period = risk_free_rate / frequency
    excess = returns - rf_per_period
    std = returns.std()
    if std == 0 or np.isnan(std):
        return np.nan
    sr = excess.mean() / std
    if annualize:
        sr *= np.sqrt(frequency)
    return float(sr)


def max_drawdown(returns: pd.Series) -> float:
    """
    Compute the maximum drawdown from peak equity.

    MDD = max over t of (peak_equity_t - trough_equity_t) / peak_equity_t

    Parameters
    ----------
    returns : pd.Series
        Periodic strategy returns (not cumulative).

    Returns
    -------
    float
        Maximum drawdown as a positive fraction (e.g., 0.15 = -15%).
    """
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    return float(-drawdown.min())


def annualized_return(returns: pd.Series, frequency: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compute annualized arithmetic mean return."""
    return float(returns.mean() * frequency)


def annualized_volatility(returns: pd.Series, frequency: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compute annualized volatility (standard deviation of returns)."""
    return float(returns.std() * np.sqrt(frequency))


def calmar_ratio(returns: pd.Series, frequency: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Compute Calmar ratio = Annualized Return / Max Drawdown.

    Returns NaN if max drawdown is zero.
    """
    mdd = max_drawdown(returns)
    ann_ret = annualized_return(returns, frequency)
    if mdd == 0:
        return np.nan
    return float(ann_ret / mdd)


def turnover_rate(signals: pd.Series) -> float:
    """
    Compute the average daily signal turnover.

    Turnover = mean(|signal_t - signal_{t-1}|)

    Values closer to 0 = rarely trades.
    Values closer to 2 = reverses position every period (max turnover).

    Parameters
    ----------
    signals : pd.Series
        Signal series ∈ {-1, 0, 1}.

    Returns
    -------
    float
        Mean absolute change in signal.
    """
    return float(signals.diff().abs().mean())


# ---------------------------------------------------------------------------
# Aggregate metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    asset_returns: pd.Series,
    signals: pd.Series,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    cost_per_trade: float = 0.0002,
    risk_free_rate: float = 0.03,
) -> dict:
    """
    Compute the full set of predictive and financial performance metrics.

    Parameters
    ----------
    asset_returns : pd.Series
        Daily log returns of the underlying.
    signals : pd.Series
        Trading signals (already lagged).
    y_true : np.ndarray, optional
        True labels for predictive accuracy metrics.
    y_pred : np.ndarray, optional
        Model predictions for predictive accuracy metrics.
    cost_per_trade : float
        Round-trip cost in return units.
    risk_free_rate : float
        Annualized risk-free rate.

    Returns
    -------
    dict
        All computed metrics in a flat dictionary.
    """
    strat_ret = compute_strategy_returns(asset_returns, signals, cost_per_trade)

    metrics = {
        "sharpe_ratio": sharpe_ratio(strat_ret, risk_free_rate=risk_free_rate),
        "annualized_return": annualized_return(strat_ret),
        "annualized_volatility": annualized_volatility(strat_ret),
        "max_drawdown": max_drawdown(strat_ret),
        "calmar_ratio": calmar_ratio(strat_ret),
        "turnover": turnover_rate(signals),
        "n_periods": len(strat_ret),
    }

    if y_true is not None and y_pred is not None:
        predictive = compute_classification_report(y_true, y_pred)
        metrics.update(predictive)
        metrics["hit_ratio"] = directional_accuracy(y_true, y_pred)

    logger.info(f"Computed metrics: {metrics}")
    return metrics
