"""
plots.py
========
Visualization utilities for the DAX/EURO STOXX Futures research project.

All plots are designed to be:
- Publication-quality: clean axes, proper labels, consistent styling
- Saveable: all functions accept an optional output path
- Composable: each function is standalone and returns a matplotlib figure

Usage:
    from src.visualization.plots import plot_equity_curve, plot_drawdown
    fig = plot_equity_curve(result.equity_curve, label="XGBoost")
    fig.savefig("reports/figures/equity_curve.png", dpi=150)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Consistent style across all plots
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "figure.figsize": (12, 5),
})


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

def plot_equity_curve(
    equity_curves: Dict[str, pd.Series],
    title: str = "Equity Curves",
    risk_free_line: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot multiple equity curves on the same axes for comparison.

    Parameters
    ----------
    equity_curves : dict
        Mapping of label → equity curve (pd.Series with DatetimeIndex).
    title : str
        Figure title.
    risk_free_line : bool
        If True, overlay a flat line at 1.0 (no growth benchmark).
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    for label, curve in equity_curves.items():
        ax.plot(curve.index, curve.values, label=label, linewidth=1.5)

    if risk_free_line:
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Break-even")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (normalized)")
    ax.legend(frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Equity curve saved to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

def plot_drawdown(
    strategy_returns: Dict[str, pd.Series],
    title: str = "Underwater Equity (Drawdown)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the drawdown (underwater equity) curves for one or more strategies.

    Parameters
    ----------
    strategy_returns : dict
        Mapping of label → strategy return series.
    title : str
    save_path : str, optional

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(13, 4))

    for label, ret in strategy_returns.items():
        cum = (1 + ret).cumprod()
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.35, label=label)
        ax.plot(drawdown.index, drawdown.values, linewidth=0.8)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(frameon=True)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Return distribution
# ---------------------------------------------------------------------------

def plot_return_distribution(
    returns: pd.Series,
    label: str = "Returns",
    bins: int = 60,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the empirical return distribution with a normal overlay.

    Parameters
    ----------
    returns : pd.Series
    label : str
    bins : int
    save_path : str, optional

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(returns.dropna(), bins=bins, density=True, color="steelblue",
            alpha=0.7, label="Empirical")

    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    normal_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, normal_pdf, "r--", linewidth=1.5, label="Normal fit")

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Return Distribution — {label}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importances: pd.Series,
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of feature importances.

    Parameters
    ----------
    importances : pd.Series
        Feature importance scores indexed by feature name.
    top_n : int
        Number of top features to display.
    title : str
    save_path : str, optional

    Returns
    -------
    plt.Figure
    """
    top = importances.nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    ax.barh(top.index, top.values, color="steelblue", alpha=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Regime comparison
# ---------------------------------------------------------------------------

def plot_regime_sharpe(
    sharpe_by_regime: Dict[str, Dict[str, float]],
    title: str = "Sharpe Ratio by Market Regime",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grouped bar chart of Sharpe ratios per model, grouped by regime.

    Parameters
    ----------
    sharpe_by_regime : dict
        {model_name: {regime_name: sharpe_value}}.
    title : str
    save_path : str, optional

    Returns
    -------
    plt.Figure

    TODO: Populate this from regime_analysis results in notebook 08.
    """
    # TODO: Implement once regime labels and model results are available
    df = pd.DataFrame(sharpe_by_regime).T
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(kind="bar", ax=ax, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(title="Regime")
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Walk-forward performance over time
# ---------------------------------------------------------------------------

def plot_walk_forward_sharpe(
    fold_results: list,
    title: str = "Walk-Forward Sharpe Ratio per Fold",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of Sharpe ratio evolution across walk-forward folds.

    Parameters
    ----------
    fold_results : list of FoldResult
        Output of WalkForwardValidator.evaluate().
    title : str
    save_path : str, optional

    Returns
    -------
    plt.Figure
    """
    sharpes = [r.metrics.get("sharpe_ratio", np.nan) for r in fold_results]
    fold_labels = [f"Fold {r.fold_idx + 1}\n{r.test_start[:7]}" for r in fold_results]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["steelblue" if s >= 0 else "salmon" for s in sharpes]
    ax.bar(fold_labels, sharpes, color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
