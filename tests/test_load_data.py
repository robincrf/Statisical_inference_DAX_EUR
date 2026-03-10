"""
test_load_data.py
=================
Unit tests for src/data/load_data.py

Tests focus on structure validation and cache behavior.
Actual Yahoo Finance downloads are mocked to avoid network dependency.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, "..")

from src.data.load_data import validate_structure


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_valid_df(n: int = 100, ticker: str = "^GDAXI") -> pd.DataFrame:
    """Create a minimal valid OHLCV DataFrame for testing."""
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    df = pd.DataFrame({
        "open":   np.random.uniform(10000, 15000, n),
        "high":   np.random.uniform(10100, 15100, n),
        "low":    np.random.uniform(9900,  14900, n),
        "close":  np.random.uniform(10000, 15000, n),
        "volume": np.nan,
        "ticker": ticker,
    }, index=idx)
    # Ensure high >= low
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1) + 10
    df["low"]  = df[["open", "high", "low", "close"]].min(axis=1) - 10
    return df


# ---------------------------------------------------------------------------
# Tests: validate_structure
# ---------------------------------------------------------------------------

class TestValidateStructure:

    def test_valid_dataframe_passes(self):
        df = make_valid_df()
        validate_structure(df, "^GDAXI")  # should not raise

    def test_raises_if_not_datetime_index(self):
        df = make_valid_df().reset_index()
        with pytest.raises(TypeError, match="DatetimeIndex"):
            validate_structure(df, "^GDAXI")

    def test_raises_if_index_not_monotonic(self):
        df = make_valid_df()
        df = pd.concat([df.iloc[50:], df.iloc[:50]])
        with pytest.raises(ValueError, match="monotonically"):
            validate_structure(df, "^GDAXI")

    def test_raises_if_close_missing(self):
        df = make_valid_df().drop(columns=["close"])
        with pytest.raises(ValueError, match="close"):
            validate_structure(df, "^GDAXI")

    def test_raises_if_close_all_nan(self):
        df = make_valid_df()
        df["close"] = np.nan
        with pytest.raises(ValueError):
            validate_structure(df, "^GDAXI")

    def test_passes_with_nan_volume(self):
        df = make_valid_df()
        df["volume"] = np.nan
        validate_structure(df, "^GDAXI")  # volume is optional


# ---------------------------------------------------------------------------
# Tests: download_index_data (mocked)
# ---------------------------------------------------------------------------

class TestDownloadIndexData:
    """
    Tests for download_index_data.
    yfinance is imported lazily inside the function; we mock the module-level
    import by patching the name as seen inside the src.data.load_data module.
    """

    def _make_mock_raw(self, n: int = 50) -> pd.DataFrame:
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.DataFrame({
            "Open":      np.random.uniform(13000, 14000, n),
            "High":      np.random.uniform(14000, 15000, n),
            "Low":       np.random.uniform(12000, 13000, n),
            "Close":     np.random.uniform(13000, 14000, n),
            "Adj Close": np.random.uniform(13000, 14000, n),
            "Volume":    np.zeros(n),
        }, index=idx)

    def test_returns_dataframe(self):
        """Mocked download returns a valid-structure DataFrame."""
        import types
        mock_yf = types.ModuleType("yfinance")
        n = 50
        mock_raw = self._make_mock_raw(n)
        mock_yf.download = MagicMock(return_value=mock_raw)

        import src.data.load_data as load_mod
        original_import = load_mod.__builtins__

        # Patch by temporarily replacing yfinance inside the module
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            # Re-import to pick up the mock
            import importlib
            importlib.reload(load_mod)
            df = load_mod.download_index_data("^GDAXI", start="2020-01-01", end="2020-03-31")

        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "close" in df.columns

    def test_raises_on_empty_download(self):
        import types, importlib
        mock_yf = types.ModuleType("yfinance")
        mock_yf.download = MagicMock(return_value=pd.DataFrame())

        import src.data.load_data as load_mod
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            importlib.reload(load_mod)
            with pytest.raises(ValueError, match="No data returned"):
                load_mod.download_index_data("INVALID_TICKER", start="2020-01-01", end="2020-12-31")
