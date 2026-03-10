"""
load_data.py
============
Data ingestion module for the DAX / EURO STOXX 50 research project.

Data source: Yahoo Finance via the `yfinance` library.
Tickers: ^GDAXI (DAX Performance Index) and ^STOXX50E (EURO STOXX 50 Index).

Responsibilities:
- Downloading OHLCV data from Yahoo Finance
- Caching raw downloads locally as Parquet to avoid repeated API calls
- Standardizing column names and data types
- Converting to UTC DatetimeIndex, sorting chronologically
- Basic structural validation

Usage:
    from src.data.load_data import download_index_data, load_all_assets
    df = download_index_data("^GDAXI", start="2010-01-01", end="2024-12-31")
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Yahoo Finance downloader
# ---------------------------------------------------------------------------

def download_index_data(
    ticker: str,
    start: str,
    end: str,
    cache_path: Optional[str] = None,
    force_download: bool = False,
    label: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for an index ticker from Yahoo Finance.

    Uses `Adj Close` as the close price (adjusted for dividends and splits,
    although index tickers are typically unadjusted).

    Data is cached locally as Parquet on first download to avoid repeated
    API calls. Use `force_download=True` to bypass the cache.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol, e.g. "^GDAXI" or "^STOXX50E".
    start : str
        Start date in "YYYY-MM-DD" format (inclusive).
    end : str
        End date in "YYYY-MM-DD" format (inclusive).
    cache_path : str, optional
        Directory to save/load cached Parquet files. If None, no caching.
    force_download : bool
        If True, re-download even if a local cache file exists.
    label : str, optional
        Human-readable label for logging. Defaults to ticker.

    Returns
    -------
    pd.DataFrame
        DataFrame with UTC DatetimeIndex and standardized columns:
        [open, high, low, close, volume, ticker].

    Raises
    ------
    ImportError
        If `yfinance` is not installed.
    ValueError
        If the downloaded DataFrame is empty (invalid ticker or date range).
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is not installed. Run: pip install yfinance"
        )

    label = label or ticker
    safe_name = ticker.replace("^", "").replace(".", "_")

    # --- Check cache ---
    cache_file = None
    if cache_path is not None:
        cache_dir = Path(cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{safe_name}_{start}_{end}.parquet"

        if cache_file.exists() and not force_download:
            logger.info(f"[{label}] Loading from cache: {cache_file}")
            df = pd.read_parquet(cache_file)
            return df

    # --- Download from Yahoo Finance ---
    logger.info(f"[{label}] Downloading {ticker} from Yahoo Finance ({start} → {end})...")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    if raw.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' between {start} and {end}. "
            "Check that the ticker is valid and the date range is correct."
        )

    # --- Flatten MultiIndex columns (yfinance ≥0.2 may return MultiIndex) ---
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    # --- Normalize column names ---
    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    raw.rename(columns=col_map, inplace=True)

    # Use Adj Close as the working close price for indices
    if "adj_close" in raw.columns:
        raw["close"] = raw["adj_close"]
        raw.drop(columns=["adj_close"], inplace=True)

    # --- Standardize index ---
    raw.index = pd.to_datetime(raw.index)
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    else:
        raw.index = raw.index.tz_convert("UTC")
    raw.index.name = "datetime"

    # --- Sort chronologically ---
    raw.sort_index(inplace=True)

    # --- Handle volume (meaningless for index tickers) ---
    if "volume" in raw.columns:
        if raw["volume"].sum() == 0 or raw["volume"].isna().all():
            logger.info(f"[{label}] Volume column is zero/NaN (expected for index). Setting to NaN.")
            raw["volume"] = np.nan

    # --- Tag with ticker ---
    raw["ticker"] = ticker

    # --- Drop completely empty rows ---
    raw.dropna(subset=["open", "high", "low", "close"], how="all", inplace=True)

    logger.info(
        f"[{label}] Downloaded {len(raw)} rows. "
        f"Range: {raw.index[0].date()} → {raw.index[-1].date()}"
    )

    # --- Cache to disk ---
    if cache_file is not None:
        raw.to_parquet(cache_file)
        logger.info(f"[{label}] Cached to {cache_file}")

    return raw


# ---------------------------------------------------------------------------
# Multi-asset loader
# ---------------------------------------------------------------------------

def load_all_assets(cfg: dict) -> Dict[str, pd.DataFrame]:
    """
    Load all assets defined in data_config.yaml from Yahoo Finance.

    Uses the caching mechanism in `download_index_data`.

    Parameters
    ----------
    cfg : dict
        Full data configuration dict loaded from `data_config.yaml`.

    Returns
    -------
    dict
        Mapping of ticker symbol → standardized pd.DataFrame.

    Example
    -------
    >>> from src.utils.config import load_config
    >>> from src.data.load_data import load_all_assets
    >>> cfg = load_config("configs/data_config.yaml")
    >>> datasets = load_all_assets(cfg)
    >>> df_dax = datasets["^GDAXI"]
    """
    start = cfg["date_range"]["start"]
    end = cfg["date_range"]["end"]
    cache_path = cfg.get("data_cache", {}).get("path", None) \
        if cfg.get("data_cache", {}).get("enabled", False) else None

    datasets = {}
    for asset in cfg["assets"]:
        ticker = asset["ticker"]
        label = asset.get("label", ticker)
        df = download_index_data(
            ticker=ticker,
            start=start,
            end=end,
            cache_path=cache_path,
            label=label,
        )
        validate_structure(df, label)
        datasets[ticker] = df

    return datasets


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------

def validate_structure(df: pd.DataFrame, label: str) -> None:
    """
    Perform basic structural validation on a loaded DataFrame.

    Checks:
    - DatetimeIndex present and monotonically increasing
    - Required columns present and not entirely null

    Parameters
    ----------
    df : pd.DataFrame
    label : str
        Asset label for logging.

    Raises
    ------
    TypeError / ValueError on validation failure.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"[{label}] Index must be a DatetimeIndex.")

    if not df.index.is_monotonic_increasing:
        raise ValueError(f"[{label}] DatetimeIndex is not monotonically increasing.")

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"[{label}] Missing required column: '{col}'.")
        if df[col].isna().all():
            raise ValueError(f"[{label}] Column '{col}' is entirely null.")

    logger.info(f"[{label}] Structural validation passed. {len(df)} rows.")
