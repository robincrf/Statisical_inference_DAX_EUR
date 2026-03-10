"""
helpers.py
==========
General-purpose utility functions for the DAX/EURO STOXX Futures research project.

These are small, stateless helper functions that do not belong to a specific
analytical module but are used across the pipeline.

Categories:
  - Date / time utilities
  - DataFrame inspection utilities
  - Path resolution utilities
  - Quick sanity check functions
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Date / time helpers
# ---------------------------------------------------------------------------

def to_datetime_index(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    """
    Set a column as a DatetimeIndex if not already done.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
        Name of the column to set as index.

    Returns
    -------
    pd.DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df[col]))
        df.drop(columns=[col], errors="ignore", inplace=True)
    return df


def enforce_business_day_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove weekend rows from a daily DataFrame (if any).

    Note: Futures markets may have holiday gaps. Use with care on intraday data.

    Parameters
    ----------
    df : pd.DataFrame with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
    """
    mask = df.index.dayofweek < 5  # Monday=0, Friday=4
    removed = (~mask).sum()
    if removed > 0:
        logger.warning(f"Removed {removed} non-business-day rows.")
    return df[mask]


def date_range_coverage(index: pd.DatetimeIndex, start: str, end: str) -> Tuple[int, float]:
    """
    Check how many periods within [start, end] are present in the index.

    Parameters
    ----------
    index : pd.DatetimeIndex
    start : str
    end : str

    Returns
    -------
    Tuple[int, float]
        (n_present, coverage_ratio) where coverage_ratio ∈ [0, 1].
    """
    period_mask = (index >= start) & (index <= end)
    n_present = period_mask.sum()
    expected = len(pd.bdate_range(start, end))
    ratio = n_present / expected if expected > 0 else 0.0
    return int(n_present), round(ratio, 4)


# ---------------------------------------------------------------------------
# DataFrame inspection
# ---------------------------------------------------------------------------

def check_no_lookahead(df: pd.DataFrame, target_col: str = "target") -> None:
    """
    Basic sanity check: verify the target column is not correlated with
    contemporaneous features in a suspicious way.

    This is a heuristic, not a proof. True look-ahead checks require code review.

    Parameters
    ----------
    df : pd.DataFrame
        Feature + target DataFrame.
    target_col : str
        Name of the target column.

    Raises
    ------
    Warning
        Logs a warning if any feature has suspiciously high correlation
        with the target (|r| > 0.5). This may indicate look-ahead leakage.
    """
    if target_col not in df.columns:
        logger.info("No target column found. Skipping look-ahead check.")
        return

    numeric_features = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    correlations = numeric_features.corrwith(df[target_col]).abs()
    suspicious = correlations[correlations > 0.5]

    if not suspicious.empty:
        logger.warning(
            f"Potential look-ahead bias: {len(suspicious)} features have |correlation| > 0.5 "
            f"with target. Inspect manually: {suspicious.to_dict()}"
        )
    else:
        logger.info("Look-ahead correlation check passed (no feature |r| > 0.5 with target).")


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a comprehensive summary of a DataFrame.

    Includes: dtype, missing count, missing %, min, max, mean, std.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Summary table, one row per column.
    """
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "n_missing": df.isnull().sum(),
        "pct_missing": (df.isnull().sum() / len(df) * 100).round(2),
        "min": df.min(numeric_only=True),
        "max": df.max(numeric_only=True),
        "mean": df.mean(numeric_only=True),
        "std": df.std(numeric_only=True),
    })
    return summary


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_directory(path: str) -> Path:
    """
    Create a directory (and all parents) if it does not exist.

    Parameters
    ----------
    path : str

    Returns
    -------
    Path
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_project_root() -> Path:
    """
    Resolve the project root directory.

    Searches upward from the current file until a directory containing
    'README.md' is found. Returns the first such directory.

    Returns
    -------
    Path
        Project root path.
    """
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "README.md").exists():
            return parent
    logger.warning("Could not locate project root. Defaulting to cwd.")
    return Path.cwd()


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

def flatten_multiindex_columns(df: pd.DataFrame, sep: str = "_") -> pd.DataFrame:
    """
    Flatten MultiIndex columns to single-level by joining levels with `sep`.

    Useful after pd.DataFrame.resample().agg() or similar operations.

    Parameters
    ----------
    df : pd.DataFrame
    sep : str

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [sep.join(str(c) for c in col).strip(sep) for col in df.columns.values]
    return df
