"""
build_features.py
=================
Feature engineering module for the DAX/EURO STOXX Futures research project.

This module computes all predictive features from cleaned OHLCV price data.
All features must be fully look-ahead-free: at time t, only information
available at or before t-1 (after applying lags) should be used.

Feature categories:
  - Return-based features (lagged log returns)
  - Rolling volatility
  - Momentum (cumulative return over rolling window)
  - Mean reversion (z-score, distance from moving average)
  - Range-based features (High-Low range, ATR)
  - Calendar/time features
  - Cross-asset features (optional)
  - Target variable (forward return or direction)

Usage:
    from src.features.build_features import build_all_features
    df_features = build_all_features(df_clean, cfg)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def build_all_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Compute all features from a cleaned OHLCV DataFrame, driven by config.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV DataFrame with DatetimeIndex and columns
        [open, high, low, close, volume] (volume may be NaN).
    cfg : dict
        Features configuration dict (loaded from features_config.yaml).

    Returns
    -------
    pd.DataFrame
        DataFrame with all feature columns appended.
        Rows with NaN (due to rolling windows) are NOT dropped here;
        handle them before model training.
    """
    df = df.copy()

    if cfg.get("return_features", {}).get("enabled", True):
        df = add_lagged_returns(df, lags=cfg["return_features"].get("lags", [1, 2, 3, 5]))

    if cfg.get("rolling_volatility", {}).get("enabled", True):
        df = add_rolling_volatility(df, windows=cfg["rolling_volatility"].get("windows", [5, 21]))

    if cfg.get("momentum", {}).get("enabled", True):
        df = add_momentum(df, windows=cfg["momentum"].get("windows", [5, 21]))

    if cfg.get("mean_reversion", {}).get("enabled", True):
        df = add_mean_reversion(df, windows=cfg["mean_reversion"].get("zscore_windows", [10, 21]))

    if cfg.get("range_features", {}).get("enabled", True):
        df = add_range_features(df, atr_window=cfg["range_features"].get("atr_window", 14))

    if cfg.get("calendar_features", {}).get("enabled", True):
        df = add_calendar_features(df, cfg.get("calendar_features", {}))

    # --- Compute target variable LAST ---
    target_cfg = cfg.get("target", {})
    df = add_target(
        df,
        horizon=target_cfg.get("horizon", 1),
        task_type=target_cfg.get("type", "classification"),
        threshold=target_cfg.get("classification_threshold", 0.0),
    )

    logger.info(f"Feature matrix built. Shape: {df.shape}. Columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Return-based features
# ---------------------------------------------------------------------------

def compute_log_returns(df: pd.DataFrame, col: str = "close") -> pd.Series:
    """
    Compute log returns of a price series.

    r_t = log(P_t / P_{t-1})

    Parameters
    ----------
    df : pd.DataFrame
    col : str
        Column to compute returns on.

    Returns
    -------
    pd.Series
        Log return series aligned to the original index.
    """
    return np.log(df[col] / df[col].shift(1))


def add_lagged_returns(df: pd.DataFrame, lags: list = [1, 2, 3, 5]) -> pd.DataFrame:
    """
    Add lagged log return features.

    Feature names: ret_lag_{k} for each k in lags.
    ret_lag_1 = return from t-2 to t-1 (i.e., past return, look-ahead safe).

    NOTE: Computing ret = log(close_t / close_{t-1}) and then lagging by 1
    gives ret_lag_1 = ret_{t-1}, which is known at time t. This is CORRECT.

    Parameters
    ----------
    df : pd.DataFrame
    lags : list of int
        Lag orders to include.

    Returns
    -------
    pd.DataFrame
    """
    ret = compute_log_returns(df)
    for lag in lags:
        df[f"ret_lag_{lag}"] = ret.shift(lag)
    return df


# ---------------------------------------------------------------------------
# Rolling volatility
# ---------------------------------------------------------------------------

def add_rolling_volatility(
    df: pd.DataFrame,
    windows: list = [5, 21],
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Add rolling realized volatility features (std of log returns).

    Feature names: vol_{w}d for each window w.
    All values are shifted by 1 to ensure look-ahead safety.

    Parameters
    ----------
    df : pd.DataFrame
    windows : list of int
        Rolling window sizes in periods.
    annualize : bool
        If True, multiply by sqrt(trading_days).
    trading_days : int
        Number of trading days per year for annualization.

    Returns
    -------
    pd.DataFrame
    """
    ret = compute_log_returns(df)
    scale = np.sqrt(trading_days) if annualize else 1.0

    for w in windows:
        vol = ret.rolling(w).std() * scale
        df[f"vol_{w}d"] = vol.shift(1)  # lag 1 to prevent look-ahead

    return df


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def add_momentum(df: pd.DataFrame, windows: list = [5, 21]) -> pd.DataFrame:
    """
    Add rolling momentum features (cumulative log return over window).

    Feature names: mom_{w}d for each window w.
    Shifted by 1 to ensure look-ahead safety.

    Parameters
    ----------
    df : pd.DataFrame
    windows : list of int
        Rolling window sizes in periods.

    Returns
    -------
    pd.DataFrame
    """
    ret = compute_log_returns(df)

    for w in windows:
        mom = ret.rolling(w).sum()
        df[f"mom_{w}d"] = mom.shift(1)

    return df


# ---------------------------------------------------------------------------
# Mean reversion
# ---------------------------------------------------------------------------

def add_mean_reversion(df: pd.DataFrame, windows: list = [10, 21]) -> pd.DataFrame:
    """
    Add z-score-based mean reversion features.

    z_{t,w} = (close_t - mean(close, w)) / std(close, w)

    Shifted by 1 to be look-ahead safe.

    Parameters
    ----------
    df : pd.DataFrame
    windows : list of int
        Look-back windows for rolling mean and std.

    Returns
    -------
    pd.DataFrame
    """
    for w in windows:
        roll_mean = df["close"].rolling(w).mean()
        roll_std = df["close"].rolling(w).std()
        zscore = (df["close"] - roll_mean) / roll_std
        df[f"zscore_{w}d"] = zscore.shift(1)

    return df


# ---------------------------------------------------------------------------
# Range-based features
# ---------------------------------------------------------------------------

def add_range_features(df: pd.DataFrame, atr_window: int = 14) -> pd.DataFrame:
    """
    Add high-low range features and Average True Range (ATR).

    All features shifted by 1 for look-ahead safety.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain [high, low, close].
    atr_window : int
        Rolling window for ATR calculation.

    Returns
    -------
    pd.DataFrame
    """
    if not all(c in df.columns for c in ["high", "low", "close"]):
        logger.warning("Missing high/low/close for range features. Skipping.")
        return df

    # Normalized High-Low range
    hl_range = (df["high"] - df["low"]) / df["close"]
    df["hl_range"] = hl_range.shift(1)

    # True Range
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_window).mean()
    df[f"atr_{atr_window}d"] = atr.shift(1)

    return df


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Add time and calendar features derived from the DatetimeIndex.

    These features capture potential seasonality patterns (day-of-week effects,
    month effects, etc.). Their predictive value must be validated empirically.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex.
    cfg : dict
        Calendar features config section from features_config.yaml.

    Returns
    -------
    pd.DataFrame
    """
    idx = df.index

    if cfg.get("day_of_week", True):
        df["day_of_week"] = idx.dayofweek  # 0=Monday, 4=Friday

    if cfg.get("month", True):
        df["month"] = idx.month

    if cfg.get("is_month_end", True):
        df["is_month_end"] = idx.is_month_end.astype(int)

    if cfg.get("is_quarter_end", True):
        df["is_quarter_end"] = idx.is_quarter_end.astype(int)

    # TODO: Consider encoding cyclical features (day_of_week, month) using
    # sin/cos transforms to preserve their circular nature.

    return df


# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------

def add_target(
    df: pd.DataFrame,
    horizon: int = 1,
    task_type: str = "classification",
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Compute and append the target variable.

    For classification: target = sign(forward_return) ∈ {-1, 1}
    For regression: target = forward_log_return (continuous)

    The target is forward-looking by construction. It must NEVER be used as a
    feature — it represents what we are trying to predict.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with at least a 'close' column.
    horizon : int
        Number of periods ahead to predict. horizon=1 → next period's return.
    task_type : str
        "classification" or "regression".
    threshold : float
        For classification: classify as +1 if forward_return > threshold, else -1.
        Set to 0.0 for pure direction; increase for "meaningful move" classification.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'target' column appended. NaNs appear at the end
        (horizon rows) — these rows must be dropped before training.
    """
    fwd_return = np.log(df["close"].shift(-horizon) / df["close"])

    if task_type == "classification":
        target = np.where(fwd_return > threshold, 1, -1)
        df["target"] = pd.Series(target, index=df.index)
    elif task_type == "regression":
        df["target"] = fwd_return
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Use 'classification' or 'regression'.")

    # Mark last `horizon` rows as NaN (no forward return available)
    df.loc[df.index[-horizon:], "target"] = np.nan
    logger.info(f"Target '{task_type}' added with horizon={horizon}. "
                f"NaN rows at tail: {horizon}.")
    return df
