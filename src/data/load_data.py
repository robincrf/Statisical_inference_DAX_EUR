"""
load_data.py
============
Data ingestion module for the DAX/EURO STOXX Futures research project.

Responsibilities:
- Loading raw OHLCV data from flat files (CSV, Parquet)
- Standardizing column names and data types
- Parsing and validating timestamps
- Sorting chronologically and enforcing a unique datetime index
- Basic structural validation (shape, expected columns, date range)

Usage:
    from src.data.load_data import load_raw_data, validate_structure
    df = load_raw_data("data/raw/FDAX_1d_20150101_20241231.csv", ticker="FDAX")
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------

STANDARD_COLUMNS = ["open", "high", "low", "close", "volume"]

DEFAULT_COLUMN_MAP = {
    # Add your raw file column names here if they differ from standard
    # e.g. "Close": "close", "Date": "datetime"
}


# ---------------------------------------------------------------------------
# Core loading function
# ---------------------------------------------------------------------------

def load_raw_data(
    filepath: str,
    ticker: str,
    datetime_col: str = "datetime",
    column_map: Optional[dict] = None,
    tz_input: str = "Europe/Berlin",
    tz_output: str = "UTC",
    file_format: str = "csv",
    sep: str = ",",
    decimal: str = ".",
) -> pd.DataFrame:
    """
    Load raw OHLCV data from a flat file and return a standardized DataFrame.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the raw data file.
    ticker : str
        Asset identifier, e.g. "FDAX" or "FESX". Appended as a column.
    datetime_col : str
        Name of the datetime column in the raw file.
    column_map : dict, optional
        Mapping from raw column names to standard names. Defaults to DEFAULT_COLUMN_MAP.
    tz_input : str
        Timezone of the raw timestamps. Use "UTC" if already in UTC.
    tz_output : str
        Target timezone for the output DataFrame index (typically UTC).
    file_format : str
        Format of the raw file. Options: "csv", "parquet".
    sep : str
        CSV field separator.
    decimal : str
        Decimal separator.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with a DatetimeIndex and columns:
        [open, high, low, close, volume, ticker].
        Volume column may be NaN if not present in source.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at `filepath`.
    ValueError
        If required columns are missing after applying the column map.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    logger.info(f"Loading {ticker} data from {filepath}")

    # --- Load raw file ---
    if file_format == "csv":
        df = pd.read_csv(path, sep=sep, decimal=decimal, low_memory=False)
    elif file_format == "parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}. Use 'csv' or 'parquet'.")

    # --- Apply column mapping ---
    col_map = column_map if column_map is not None else DEFAULT_COLUMN_MAP
    if col_map:
        df.rename(columns=col_map, inplace=True)

    logger.debug(f"Raw columns after mapping: {list(df.columns)}")

    # --- Parse datetime ---
    if datetime_col not in df.columns:
        raise ValueError(
            f"Datetime column '{datetime_col}' not found. Available: {list(df.columns)}"
        )
    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=False)

    # --- Localize and convert timezone ---
    # TODO: Adjust tz_input if your raw data already contains tz-aware timestamps
    if df[datetime_col].dt.tz is None:
        df[datetime_col] = df[datetime_col].dt.tz_localize(tz_input, ambiguous="infer")
    df[datetime_col] = df[datetime_col].dt.tz_convert(tz_output)

    # --- Set datetime index ---
    df.set_index(datetime_col, inplace=True)
    df.index.name = "datetime"

    # --- Sort chronologically ---
    df.sort_index(inplace=True)

    # --- Standardize floats ---
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Volume: make optional ---
    if "volume" not in df.columns:
        logger.warning(f"No 'volume' column found for {ticker}. Filling with NaN.")
        df["volume"] = float("nan")
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # --- Add ticker identifier ---
    df["ticker"] = ticker

    logger.info(f"Loaded {len(df)} rows for {ticker}. Range: {df.index[0]} → {df.index[-1]}")
    return df


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------

def validate_structure(df: pd.DataFrame, ticker: str) -> None:
    """
    Perform basic structural validation on the loaded DataFrame.

    Checks:
    - DatetimeIndex is present and monotonically increasing
    - Required price columns are present
    - No entirely-null price columns

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by `load_raw_data`.
    ticker : str
        Asset identifier for logging purposes.

    Raises
    ------
    TypeError
        If the index is not a DatetimeIndex.
    ValueError
        If required columns are missing or data is entirely null.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"[{ticker}] Index must be a DatetimeIndex.")

    if not df.index.is_monotonic_increasing:
        raise ValueError(f"[{ticker}] DatetimeIndex is not monotonically increasing.")

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{ticker}] Missing required columns: {missing}")

    for col in required:
        if df[col].isna().all():
            raise ValueError(f"[{ticker}] Column '{col}' is entirely null.")

    logger.info(f"[{ticker}] Structural validation passed.")


# ---------------------------------------------------------------------------
# Multi-asset loader
# ---------------------------------------------------------------------------

def load_multiple_assets(asset_configs: list) -> dict:
    """
    Load and validate data for multiple assets from a list of config dicts.

    Parameters
    ----------
    asset_configs : list of dict
        Each dict must contain at minimum: {"ticker": str, "file": str, "format": str}.
        Additional keys are passed to `load_raw_data`.

    Returns
    -------
    dict
        Dictionary mapping ticker → pd.DataFrame.

    Example
    -------
    >>> configs = [
    ...     {"ticker": "FDAX", "file": "data/raw/fdax.csv", "format": "csv"},
    ...     {"ticker": "FESX", "file": "data/raw/fesx.csv", "format": "csv"},
    ... ]
    >>> data = load_multiple_assets(configs)
    """
    # TODO: Populate asset_configs from data_config.yaml at runtime
    datasets = {}
    for cfg in asset_configs:
        ticker = cfg.pop("ticker")
        filepath = cfg.pop("file")
        file_format = cfg.pop("format", "csv")
        df = load_raw_data(filepath, ticker=ticker, file_format=file_format, **cfg)
        validate_structure(df, ticker)
        datasets[ticker] = df
    return datasets
