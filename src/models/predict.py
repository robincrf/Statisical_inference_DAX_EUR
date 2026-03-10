"""
predict.py
==========
Prediction and signal generation module for the DAX/EURO STOXX Futures project.

This module handles the final step between model output and tradeable signals:
- Generating raw predictions (class labels or regression scores)
- Generating probability scores from classifiers
- Mapping predictions to discrete trading signals: +1 (long), -1 (short), 0 (flat)
- Applying a lag to prevent look-ahead bias in live simulation

Usage:
    from src.models.predict import generate_predictions, predictions_to_signals
    preds = generate_predictions(pipeline, X_test)
    signals = predictions_to_signals(preds, method="threshold", threshold=0.55)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prediction generation
# ---------------------------------------------------------------------------

def generate_predictions(
    pipeline: Pipeline,
    X: pd.DataFrame,
    task_type: str = "classification",
    return_proba: bool = False,
) -> pd.DataFrame:
    """
    Generate predictions from a fitted pipeline on feature matrix X.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn pipeline (output of train_model).
    X : pd.DataFrame
        Feature matrix. Must have the same columns used during training.
        Rows with NaN are excluded and filled with NaN in output.
    task_type : str
        "classification" or "regression". Controls whether probabilities are returned.
    return_proba : bool
        If True and task_type="classification", also return predicted probabilities.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and columns:
        - "prediction": raw model prediction (class or regression value)
        - "proba_pos" (optional): probability of positive class (if return_proba=True)
    """
    mask = X.notna().all(axis=1)
    X_valid = X[mask]

    preds = pd.Series(index=X.index, dtype=float, name="prediction")
    preds[mask] = pipeline.predict(X_valid)

    result = pd.DataFrame({"prediction": preds})

    if return_proba and task_type == "classification":
        try:
            proba = pipeline.predict_proba(X_valid)
            classes = pipeline.classes_ if hasattr(pipeline, "classes_") \
                else pipeline.named_steps["model"].classes_
            pos_class_idx = list(classes).index(1) if 1 in classes else -1
            proba_series = pd.Series(index=X.index, dtype=float)
            proba_series[mask] = proba[:, pos_class_idx]
            result["proba_pos"] = proba_series
        except AttributeError:
            logger.warning("Pipeline does not support predict_proba. Skipping probability output.")

    logger.info(f"Predictions generated for {mask.sum()} valid rows out of {len(X)}.")
    return result


# ---------------------------------------------------------------------------
# Signal mapping
# ---------------------------------------------------------------------------

def predictions_to_signals(
    predictions: pd.DataFrame,
    method: str = "sign",
    threshold: float = 0.5,
    use_proba: bool = False,
    signal_lag: int = 1,
) -> pd.Series:
    """
    Map model predictions or probabilities to discrete trading signals.

    Signal convention:
        +1 = long
        -1 = short
         0 = flat (no position)

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of `generate_predictions`. Must contain "prediction" column.
        If use_proba=True, must also contain "proba_pos".
    method : str
        Mapping method:
        - "sign": signal = sign(prediction). For {-1, 1} class labels.
        - "threshold": long if proba_pos > threshold, short if < (1 - threshold), else flat.
        - "regression_sign": signal = np.sign(regression prediction).
    threshold : float
        Probability threshold for the "threshold" method (default: 0.5).
        Setting threshold=0.6 means we go long only when P(up) > 0.6.
    use_proba : bool
        If True, use "proba_pos" column instead of "prediction".
    signal_lag : int
        Number of periods to lag the signal to ensure no look-ahead.
        CRITICAL: Must be >= 1 in all real-time or backtesting contexts.

    Returns
    -------
    pd.Series
        Series of signals ∈ {-1, 0, 1} with same DatetimeIndex as input.

    Notes
    -----
    The lag is applied AFTER computing the signal. A lag of 1 means that
    the signal computed from features up to time t is only acted on at t+1.
    This is the correct and look-ahead-safe convention.
    """
    if method == "sign":
        raw = predictions["prediction"]
        signal = raw.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    elif method == "threshold":
        if use_proba and "proba_pos" in predictions.columns:
            prob = predictions["proba_pos"]
        else:
            logger.warning("use_proba=True but 'proba_pos' not found. Falling back to 'prediction'.")
            prob = predictions["prediction"]

        signal = pd.Series(0, index=predictions.index, dtype=int)
        signal[prob > threshold] = 1
        signal[prob < (1 - threshold)] = -1

    elif method == "regression_sign":
        raw = predictions["prediction"]
        signal = np.sign(raw).astype(int)

    else:
        raise ValueError(f"Unknown method: '{method}'. Options: sign, threshold, regression_sign.")

    # Apply lag (CRITICAL for look-ahead safety)
    if signal_lag > 0:
        signal = signal.shift(signal_lag)
        logger.info(f"Signal lagged by {signal_lag} period(s) (look-ahead safe).")

    signal.name = "signal"
    return signal
