"""
engine.py
=========
Vectorized backtest engine for the DAX/EURO STOXX Futures research project.

This engine simulates a simple systematic strategy: at each time step,
a signal ∈ {-1, 0, 1} is converted to a position, applied to the next
period's return, and adjusted for transaction costs.

Design principles:
  - Vectorized (pandas/numpy): no loops over rows
  - Fully transparent: every component of PnL is decomposable
  - Conservative: positions are lagged by 1 period by default (see predict.py)
  - Auditable: full equity curve, drawdown, and turnover series are returned

This engine does NOT model order book dynamics, partial fills, or intraday
execution. It assumes end-of-bar execution at close price plus costs.

Usage:
    from src.backtest.engine import BacktestEngine
    engine = BacktestEngine(cfg)
    result = engine.run(signals, asset_returns)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Container for backtest outputs."""
    equity_curve: pd.Series
    strategy_returns: pd.Series
    gross_pnl: pd.Series
    transaction_costs_series: pd.Series
    position: pd.Series
    turnover: pd.Series
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Vectorized backtest engine.

    Parameters
    ----------
    cfg : dict
        Backtest configuration from backtest_config.yaml. Expected keys:
        - position.signal_lag (int)
        - transaction_costs.total_round_trip_bps (float)
        - performance.risk_free_rate (float)
        - performance.trading_days_per_year (int)
        - equity_curve.initial_capital (float)
        - equity_curve.compound_returns (bool)
    """

    def __init__(self, cfg: dict):
        pos_cfg = cfg.get("position", {})
        cost_cfg = cfg.get("transaction_costs", {})
        perf_cfg = cfg.get("performance", {})
        eq_cfg = cfg.get("equity_curve", {})

        self.signal_lag = int(pos_cfg.get("signal_lag", 1))
        self.max_position = float(pos_cfg.get("max_position", 1.0))

        # Convert bps to decimal return units
        bps = float(cost_cfg.get("total_round_trip_bps", 2.0))
        self.cost_per_trade = bps / 10_000  # 2 bps → 0.0002

        self.risk_free_rate = float(perf_cfg.get("risk_free_rate", 0.03))
        self.trading_days = int(perf_cfg.get("trading_days_per_year", 252))
        self.initial_capital = float(eq_cfg.get("initial_capital", 1.0))
        self.compound_returns = bool(eq_cfg.get("compound_returns", False))

    def run(
        self,
        signals: pd.Series,
        asset_returns: pd.Series,
        label: str = "strategy",
    ) -> BacktestResult:
        """
        Execute the backtest given a signal series and asset returns.

        Parameters
        ----------
        signals : pd.Series
            Trading signals ∈ {-1, 0, 1} with DatetimeIndex.
            MUST already be lagged by at least 1 period (from predict.py).
        asset_returns : pd.Series
            Log or simple returns of the underlying asset.
        label : str
            Label for logging purposes.

        Returns
        -------
        BacktestResult
            Full decomposition of PnL and equity curve.

        Notes
        -----
        The signal is additionally lagged by self.signal_lag for safety.
        If signal is already lagged in predict.py, set signal_lag=0 in config
        to avoid double-lagging.

        TODO: Verify that lag is applied exactly once across the full pipeline.
        """
        # Align
        df = pd.DataFrame({"signal": signals, "ret": asset_returns}).dropna()

        # Apply additional lag (config-driven; typically 0 if already lagged in predict.py)
        df["position"] = df["signal"].shift(self.signal_lag).clip(-self.max_position, self.max_position)
        df = df.dropna(subset=["position"])

        # Compute gross PnL (position at t * return at t+1)
        df["gross_pnl"] = df["position"] * df["ret"]

        # Compute turnover (absolute change in position)
        df["turnover"] = df["position"].diff().abs()

        # Transaction costs applied on every change in position
        df["tc"] = df["turnover"] * self.cost_per_trade

        # Net strategy returns
        df["strategy_return"] = df["gross_pnl"] - df["tc"]

        # Equity curve
        if self.compound_returns:
            df["equity"] = self.initial_capital * (1 + df["strategy_return"]).cumprod()
        else:
            df["equity"] = self.initial_capital + df["strategy_return"].cumsum()

        # Compute summary metrics
        from src.evaluation.metrics import (
            sharpe_ratio, max_drawdown, annualized_return,
            annualized_volatility, calmar_ratio, turnover_rate
        )

        metrics = {
            "sharpe_ratio": sharpe_ratio(df["strategy_return"], self.risk_free_rate,
                                         frequency=self.trading_days),
            "annualized_return": annualized_return(df["strategy_return"], self.trading_days),
            "annualized_volatility": annualized_volatility(df["strategy_return"], self.trading_days),
            "max_drawdown": max_drawdown(df["strategy_return"]),
            "calmar_ratio": calmar_ratio(df["strategy_return"], self.trading_days),
            "turnover_mean": float(df["turnover"].mean()),
            "total_tc_drag": float(df["tc"].sum()),
            "n_periods": len(df),
        }

        logger.info(f"[{label}] Backtest complete. Sharpe={metrics['sharpe_ratio']:.3f}, "
                    f"MDD={metrics['max_drawdown']:.3f}, Turnover={metrics['turnover_mean']:.3f}")

        return BacktestResult(
            equity_curve=df["equity"],
            strategy_returns=df["strategy_return"],
            gross_pnl=df["gross_pnl"],
            transaction_costs_series=df["tc"],
            position=df["position"],
            turnover=df["turnover"],
            metrics=metrics,
        )

    def compare_gross_vs_net(self, result: BacktestResult) -> pd.DataFrame:
        """
        Compare gross (pre-cost) and net (post-cost) performance.

        This decomposition quantifies how much of the signal's raw predictive
        value is eroded by transaction costs.

        Parameters
        ----------
        result : BacktestResult

        Returns
        -------
        pd.DataFrame
            Side-by-side gross vs. net metrics.
        """
        from src.evaluation.metrics import sharpe_ratio, annualized_return, max_drawdown

        gross_ret = result.gross_pnl
        net_ret = result.strategy_returns

        comparison = pd.DataFrame({
            "gross": {
                "sharpe_ratio": sharpe_ratio(gross_ret, frequency=self.trading_days),
                "annualized_return": annualized_return(gross_ret, self.trading_days),
                "max_drawdown": max_drawdown(gross_ret),
            },
            "net": {
                "sharpe_ratio": sharpe_ratio(net_ret, frequency=self.trading_days),
                "annualized_return": annualized_return(net_ret, self.trading_days),
                "max_drawdown": max_drawdown(net_ret),
            },
        })

        comparison["cost_drag"] = comparison["gross"] - comparison["net"]
        return comparison
