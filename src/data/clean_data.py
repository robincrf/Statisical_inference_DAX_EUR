"""
clean_data.py
=============
Data cleaning module for the DAX/EURO STOXX Futures research project.

Responsibilities:
- Removing duplicate timestamps
- Handling missing values (NaN) with explicit strategy per column
- Filtering to relevant trading sessions (optional, intraday only)
- Detecting and optionally removing price outliers
- Resampling to a target frequency if needed
- Alignment of multiple assets to a common index

All cleaning decisions should be documented in notebook 01_data_audit.ipynb.
No imputation is applied silently — every gap or anomaly must be a conscious choice.

Usage:
    from src.data.clean_data import clean_ohlcv, align_assets
    df_clean = clean_ohlcv(df_raw, ticker="FDAX")
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------------------------

def clean_ohlcv(
    df: pd.DataFrame,
    ticker: str,
    drop_missing_close: bool = True,
    outlier_method: str = "zscore",
    outlier_threshold: float = 5.0,
    fill_method: Optional[str] = None,
    session_start: Optional[str] = None,
    session_end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply a sequence of cleaning steps to a raw OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame with DatetimeIndex. Output of `load_raw_data`.
    ticker : str
        Asset identifier for logging.
    drop_missing_close : bool
        If True, drop rows where `close` is NaN after initial processing.
    outlier_method : str
        Method for outlier detection. Options: "zscore", "iqr", "none".
    outlier_threshold : float
        Threshold for outlier detection (z-score or IQR multiplier).
    fill_method : str, optional
        How to handle remaining NaNs. Options: "ffill", "bfill", None.
        If None, NaNs are left in place and must be handled downstream.
    session_start : str, optional
        Start time of trading session (e.g., "08:00"). For intraday data only.
    session_end : str, optional
        End time of trading session (e.g., "22:00"). For intraday data only.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame. Shape may differ from input due to dropped rows.
    """
    df = df.copy()
    n_initial = len(df)
    logger.info(f"[{ticker}] Starting cleaning — {n_initial} rows.")

    # Step 1: Remove duplicates
    df = remove_duplicates(df, ticker)

    # Step 2: Enforce OHLCV consistency
    df = enforce_ohlcv_consistency(df, ticker)

    # Step 3: Filter session hours (intraday only)
    if session_start and session_end:
        df = filter_session(df, ticker, session_start, session_end)

    # Step 4: Handle missing close prices
    if drop_missing_close and "close" in df.columns:
        n_before = len(df)
        df = df.dropna(subset=["close"])
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.warning(f"[{ticker}] Dropped {n_dropped} rows with missing close price.")

    # Step 5: Outlier detection and removal
    if outlier_method != "none":
        df = detect_outliers(df, ticker, method=outlier_method, threshold=outlier_threshold)

    # Step 6: Optional forward/backward fill for remaining NaNs
    if fill_method == "ffill":
        df = df.ffill()
        logger.info(f"[{ticker}] Applied forward fill for remaining NaNs.")
    elif fill_method == "bfill":
        df = df.bfill()
        logger.info(f"[{ticker}] Applied backward fill for remaining NaNs.")
    elif fill_method is None:
        n_missing = df.isnull().sum().sum()
        if n_missing > 0:
            logger.warning(
                f"[{ticker}] {n_missing} NaN values remain in the cleaned data. "
                "Consider providing a fill_method or handling them explicitly."
            )

    n_final = len(df)
    logger.info(f"[{ticker}] Cleaning complete — {n_final} rows remaining "
                f"({n_initial - n_final} removed).")
    return df


# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Remove duplicate timestamps from the DataFrame.

    Strategy: keep the last occurrence on duplicate timestamps.
    This may be adjusted depending on the data source.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.
    ticker : str
        Asset identifier.

    Returns
    -------
    pd.DataFrame
        DataFrame with unique DatetimeIndex.
    """
    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        logger.warning(f"[{ticker}] Found {n_dupes} duplicate timestamps. Keeping last.")
        df = df[~df.index.duplicated(keep="last")]
    return df


def enforce_ohlcv_consistency(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Flag or remove rows where OHLCV values violate basic price constraints:
    - high >= low
    - high >= open, close
    - low <= open, close
    - All price values > 0

    TODO: Decide whether to drop or flag inconsistent rows.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    ticker : str
        Asset identifier.

    Returns
    -------
    pd.DataFrame
        DataFrame with inconsistent rows flagged (column 'ohlcv_inconsistent').
    """
    price_cols = ["open", "high", "low", "close"]
    available = [c for c in price_cols if c in df.columns]

    if len(available) < 4:
        logger.warning(f"[{ticker}] Not all OHLCV columns are present; skipping consistency check.")
        return df

    mask_negative = (df[available] <= 0).any(axis=1)
    mask_hl = df["high"] < df["low"]
    mask_inconsistent = mask_negative | mask_hl

    n_bad = mask_inconsistent.sum()
    if n_bad > 0:
        logger.warning(f"[{ticker}] {n_bad} rows with OHLCV inconsistencies flagged.")
        df["ohlcv_inconsistent"] = mask_inconsistent
        # TODO: Decide whether to drop these rows or investigate them further
    else:
        df["ohlcv_inconsistent"] = False

    return df


def detect_outliers(
    df: pd.DataFrame,
    ticker: str,
    method: str = "zscore",
    threshold: float = 5.0,
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """
    Detect and flag (or remove) price outliers in close returns.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.
    ticker : str
        Asset identifier.
    method : str
        Detection method. Options: "zscore", "iqr".
    threshold : float
        Detection threshold. For z-score: std multiples. For IQR: IQR multiples.
    columns : list, optional
        Columns to check. Defaults to ["close"].

    Returns
    -------
    pd.DataFrame
        DataFrame with an 'outlier_flag' column added.

    Notes
    -----
    Outlier detection is applied on log returns, not raw prices, to account
    for non-stationarity. Rows are flagged but NOT automatically dropped.
    TODO: Review flagged rows manually and decide on treatment.
    """
    if columns is None:
        columns = ["close"]

    df["outlier_flag"] = False

    for col in columns:
        if col not in df.columns:
            continue
        returns = np.log(df[col]).diff()

        if method == "zscore":
            z = (returns - returns.mean()) / returns.std()
            mask = z.abs() > threshold
        elif method == "iqr":
            q1, q3 = returns.quantile(0.25), returns.quantile(0.75)
            iqr = q3 - q1
            mask = (returns < q1 - threshold * iqr) | (returns > q3 + threshold * iqr)
        else:
            raise ValueError(f"Unknown outlier method: {method}")

        n_outliers = mask.sum()
        if n_outliers > 0:
            logger.warning(f"[{ticker}] {n_outliers} outliers detected in '{col}' returns "
                           f"using {method} method (threshold={threshold}).")
            df.loc[mask, "outlier_flag"] = True

    # TODO: Review outlier_flag=True rows in notebook 01 before deciding to drop
    return df


def filter_session(
    df: pd.DataFrame,
    ticker: str,
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    """
    Filter intraday data to keep only rows within the specified trading session.

    Parameters
    ----------
    df : pd.DataFrame
        Intraday OHLCV DataFrame with DatetimeIndex (timezone-aware).
    ticker : str
        Asset identifier.
    start_time : str
        Session start in "HH:MM" format (local exchange time, e.g. "08:00").
    end_time : str
        Session end in "HH:MM" format (e.g. "22:00").

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only bars within session hours.
    """
    # TODO: Verify that timezone conversion is consistent before applying this filter
    time_index = df.index.time
    start = pd.Timestamp(f"1970-01-01 {start_time}").time()
    end = pd.Timestamp(f"1970-01-01 {end_time}").time()
    mask = (time_index >= start) & (time_index <= end)
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        logger.info(f"[{ticker}] Session filter: removed {n_dropped} bars outside {start_time}–{end_time}.")
    return df[mask]


# ---------------------------------------------------------------------------
# Multi-asset alignment
# ---------------------------------------------------------------------------

def align_assets(
    datasets: dict,
    method: str = "inner",
) -> pd.DataFrame:
    """
    Align multiple asset DataFrames to a common datetime index.

    Only the 'close' column is kept from each asset. The resulting DataFrame
    has one column per asset (named by ticker).

    Parameters
    ----------
    datasets : dict
        Dictionary mapping ticker → pd.DataFrame (output of clean_ohlcv).
    method : str
        Join method. Options: "inner" (common dates only), "outer" (all dates, NaN fill).

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with columns = tickers, index = common datetime.

    Notes
    -----
    Use "inner" join to avoid spurious NaN entries from non-overlapping sessions.
    TODO: Verify the overlap window is sufficient for your analysis.
    """
    closes = {ticker: df["close"].rename(ticker) for ticker, df in datasets.items()}
    combined = pd.concat(closes.values(), axis=1, join=method)
    logger.info(f"Aligned {len(datasets)} assets. Shape: {combined.shape}. Method: {method}.")
    return combined
