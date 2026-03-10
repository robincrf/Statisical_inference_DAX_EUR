"""
costs.py
========
Transaction cost models for the DAX/EURO STOXX Futures research project.

This module centralizes all cost-related logic to make assumptions explicit
and easy to adjust during sensitivity analysis.

Cost components:
  - Commission: exchange + broker fees per contract per side
  - Slippage: market impact / execution slippage estimation
  - Bid-ask spread: half-spread paid on each transaction

All costs are expressed in:
  - Basis points (bps): 1 bps = 0.01% of notional
  - Return units: a decimal fraction (e.g., 0.0002 = 2 bps)

Usage:
    from src.backtest.costs import TransactionCostModel
    model = TransactionCostModel(commission_bps=0.5, slippage_bps=0.5, spread_bps=1.0)
    cost_per_unit = model.total_round_trip()  # in return units
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BPS_TO_DECIMAL = 1e-4  # 1 bps = 0.0001


# ---------------------------------------------------------------------------
# Transaction Cost Model
# ---------------------------------------------------------------------------

class TransactionCostModel:
    """
    Simple, transparent transaction cost model.

    Parameters
    ----------
    commission_bps : float
        Exchange + broker commission per side in basis points.
        Typical range: 0.2–1.0 bps per side for exchange-traded futures.
    slippage_bps : float
        Estimated execution slippage per side in basis points.
        Highly dependent on contract size, liquidity, and strategy frequency.
    spread_bps : float
        Half bid-ask spread per side in basis points.
        For FDAX/FESX, typically 0.5–1.5 bps per side in liquid sessions.
    """

    def __init__(
        self,
        commission_bps: float = 0.5,
        slippage_bps: float = 0.5,
        spread_bps: float = 1.0,
    ):
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.spread_bps = spread_bps

    def commission_per_side(self) -> float:
        """Commission in return units, per side."""
        return self.commission_bps * BPS_TO_DECIMAL

    def slippage_per_side(self) -> float:
        """Slippage in return units, per side."""
        return self.slippage_bps * BPS_TO_DECIMAL

    def spread_per_side(self) -> float:
        """Spread cost in return units, per side."""
        return self.spread_bps * BPS_TO_DECIMAL

    def total_per_side(self) -> float:
        """Total cost in return units, per side of a transaction."""
        return self.commission_per_side() + self.slippage_per_side() + self.spread_per_side()

    def total_round_trip(self) -> float:
        """Total round-trip cost in return units (entry + exit)."""
        return 2 * self.total_per_side()

    def total_round_trip_bps(self) -> float:
        """Total round-trip cost in basis points."""
        return self.total_round_trip() / BPS_TO_DECIMAL

    def apply_costs(
        self,
        strategy_returns: pd.Series,
        signals: pd.Series,
    ) -> pd.Series:
        """
        Deduct transaction costs from a strategy return series.

        Cost is applied whenever the signal changes (entry, exit, or reversal).

        Parameters
        ----------
        strategy_returns : pd.Series
            Gross strategy returns (before costs).
        signals : pd.Series
            Signal series ∈ {-1, 0, 1}.

        Returns
        -------
        pd.Series
            Net strategy returns after transaction costs.
        """
        turnover = signals.diff().abs().reindex(strategy_returns.index).fillna(0)
        cost_drag = turnover * self.total_round_trip()
        net_returns = strategy_returns - cost_drag

        logger.info(f"Cost model applied. Total drag: {cost_drag.sum():.6f} "
                    f"({cost_drag.sum() * 10000:.2f} bps total). "
                    f"Average per period: {cost_drag.mean() * 10000:.4f} bps.")
        return net_returns

    def breakeven_hits_required(self, avg_return_per_trade: float) -> int:
        """
        Estimate the minimum hit ratio required to break even after costs.

        Given an average expected return on a winning trade, how many wins
        per 100 trades are needed to cover round-trip transaction costs?

        Parameters
        ----------
        avg_return_per_trade : float
            Expected absolute return on a correctly predicted period.

        Returns
        -------
        int
            Approximate minimum hit ratio (percentage) to break even.

        TODO: Use this as a calibration tool after initial EDA on return distributions.
        """
        if avg_return_per_trade <= 0:
            return 100
        cost = self.total_round_trip()
        breakeven_hr = 0.5 + cost / (2 * avg_return_per_trade)
        return min(int(np.ceil(breakeven_hr * 100)), 100)

    def summary(self) -> dict:
        """Return a readable cost breakdown dictionary."""
        return {
            "commission_per_side_bps": self.commission_bps,
            "slippage_per_side_bps": self.slippage_bps,
            "spread_per_side_bps": self.spread_bps,
            "total_per_side_bps": self.total_per_side() / BPS_TO_DECIMAL,
            "total_round_trip_bps": self.total_round_trip_bps(),
            "total_round_trip_decimal": self.total_round_trip(),
        }
