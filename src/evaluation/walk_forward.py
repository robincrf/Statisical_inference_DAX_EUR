"""
walk_forward.py
===============
Walk-forward validation engine for the DAX/EURO STOXX Futures research project.

Walk-forward validation is the only temporally valid cross-validation strategy
for financial time series. Standard k-fold CV is NOT appropriate here because
it violates temporal ordering and induces look-ahead bias.

Two modes are supported:
  - Rolling window: fixed-size training window that moves forward in time
  - Expanding window: training window grows as time progresses (more common)

Key invariants enforced:
  - No training data from after the test period is ever used
  - Each test fold is strictly out-of-sample relative to training data
  - Gap between train_end and test_start is configurable to avoid data leakage
    at frequency junctions (e.g., weekly features on daily data)

Usage:
    from src.evaluation.walk_forward import WalkForwardValidator
    wfv = WalkForwardValidator(n_splits=5, window_type="expanding", gap=0)
    results = wfv.evaluate(pipeline_factory, X, y, asset_returns)
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fold data class
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""
    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    metrics: Dict = field(default_factory=dict)
    predictions: Optional[pd.Series] = None
    signals: Optional[pd.Series] = None


# ---------------------------------------------------------------------------
# Walk-forward validator
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """
    Implements walk-forward validation for time series classification/regression.

    Parameters
    ----------
    n_splits : int
        Number of out-of-sample test folds.
    window_type : str
        "expanding": training window grows from origin to current fold.
        "rolling": training window has fixed size, shifts forward.
    train_window : int, optional
        Number of periods per training window. Required if window_type="rolling".
    gap : int
        Number of periods between training end and test start.
        Set > 0 if features have look-ahead contamination risk at boundaries.
    min_train_size : int
        Minimum number of training samples required to fit a model.
    """

    def __init__(
        self,
        n_splits: int = 5,
        window_type: str = "expanding",
        train_window: Optional[int] = None,
        gap: int = 0,
        min_train_size: int = 252,
    ):
        self.n_splits = n_splits
        self.window_type = window_type
        self.train_window = train_window
        self.gap = gap
        self.min_train_size = min_train_size

        if window_type == "rolling" and train_window is None:
            raise ValueError("train_window must be specified when window_type='rolling'.")

    def generate_splits(
        self, index: pd.DatetimeIndex
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate (train_indices, test_indices) pairs for each fold.

        Parameters
        ----------
        index : pd.DatetimeIndex
            DatetimeIndex of the full dataset.

        Yields
        ------
        Tuple of (train_idx, test_idx) as integer arrays.
        """
        n = len(index)
        test_size = n // (self.n_splits + 1)

        for fold in range(self.n_splits):
            test_start_idx = (fold + 1) * test_size
            test_end_idx = test_start_idx + test_size

            if self.window_type == "expanding":
                train_start_idx = 0
                # train ends just before the test period (exclusive upper bound for arange)
                train_end_idx = test_start_idx - self.gap
            else:  # rolling
                train_end_idx = test_start_idx - self.gap
                train_start_idx = max(0, train_end_idx - self.train_window)

            if (train_end_idx - train_start_idx) < self.min_train_size:
                logger.warning(f"Fold {fold}: insufficient training samples. Skipping.")
                continue

            test_end_idx = min(test_end_idx, n)
            yield (
                np.arange(train_start_idx, train_end_idx),
                np.arange(test_start_idx, test_end_idx),
            )

    def evaluate(
        self,
        pipeline_factory: Callable[[], Pipeline],
        X: pd.DataFrame,
        y: pd.Series,
        asset_returns: pd.Series,
        cost_per_trade: float = 0.0002,
        risk_free_rate: float = 0.03,
    ) -> List[FoldResult]:
        """
        Run the full walk-forward evaluation across all folds.

        A fresh pipeline is instantiated via `pipeline_factory` for each fold,
        ensuring zero information leakage between folds.

        Parameters
        ----------
        pipeline_factory : Callable[[], Pipeline]
            A zero-argument callable that returns an unfitted sklearn Pipeline.
            Example: lambda: build_ml_pipeline("random_forest", cfg=cfg)
        X : pd.DataFrame
            Full feature matrix (all periods).
        y : pd.Series
            Full target series.
        asset_returns : pd.Series
            Full asset return series (for computing financial metrics).
        cost_per_trade : float
            Transaction cost in return units.
        risk_free_rate : float
            Annualized risk-free rate for Sharpe computation.

        Returns
        -------
        List[FoldResult]
            One FoldResult per fold.
        """
        results = []
        index = X.index

        for fold_idx, (train_idx, test_idx) in enumerate(self.generate_splits(index)):
            logger.info(f"Fold {fold_idx + 1}/{self.n_splits}: "
                        f"train [{index[train_idx[0]]} → {index[train_idx[-1]]}], "
                        f"test [{index[test_idx[0]]} → {index[test_idx[-1]]}]")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            ret_test = asset_returns.iloc[test_idx]

            # Drop NaN rows before fitting
            valid_train = X_train.notna().all(axis=1) & y_train.notna()
            if valid_train.sum() < self.min_train_size:
                logger.warning(f"Fold {fold_idx}: too few valid training rows. Skipping.")
                continue

            # Instantiate a fresh model for each fold
            pipeline = pipeline_factory()
            pipeline.fit(X_train[valid_train], y_train[valid_train])

            # Generate predictions on test set
            valid_test = X_test.notna().all(axis=1)
            raw_preds = pd.Series(index=X_test.index, dtype=float)
            raw_preds[valid_test] = pipeline.predict(X_test[valid_test])

            # Map to signals (lag already applied in signal generation)
            # TODO: call predictions_to_signals from predict.py here
            signals = raw_preds.apply(lambda x: int(np.sign(x)) if not np.isnan(x) else 0)
            signals = signals.shift(1).fillna(0)  # safety lag

            # Evaluate
            y_pred_arr = raw_preds.dropna().values
            y_true_arr = y_test[raw_preds.notna()].values

            metrics = compute_all_metrics(
                ret_test, signals, y_true_arr, y_pred_arr,
                cost_per_trade=cost_per_trade,
                risk_free_rate=risk_free_rate,
            )

            result = FoldResult(
                fold_idx=fold_idx,
                train_start=str(index[train_idx[0]]),
                train_end=str(index[train_idx[-1]]),
                test_start=str(index[test_idx[0]]),
                test_end=str(index[test_idx[-1]]),
                n_train=valid_train.sum(),
                n_test=valid_test.sum(),
                metrics=metrics,
                predictions=raw_preds,
                signals=signals,
            )
            results.append(result)

        logger.info(f"Walk-forward evaluation complete. {len(results)} folds processed.")
        return results

    def aggregate_results(self, results: List[FoldResult]) -> pd.DataFrame:
        """
        Aggregate fold-level metrics into a summary DataFrame.

        Parameters
        ----------
        results : List[FoldResult]

        Returns
        -------
        pd.DataFrame
            One row per fold, columns = metric names.
        """
        rows = []
        for r in results:
            row = {
                "fold": r.fold_idx + 1,
                "train_start": r.train_start,
                "train_end": r.train_end,
                "test_start": r.test_start,
                "test_end": r.test_end,
                "n_train": r.n_train,
                "n_test": r.n_test,
            }
            row.update(r.metrics)
            rows.append(row)

        return pd.DataFrame(rows)
