"""
baselines.py
============
Baseline model definitions for the DAX/EURO STOXX Futures research project.

Baselines serve two critical roles:
1. Establishing a minimum performance benchmark that any ML model must beat.
2. Quantifying whether signals are "naive" (e.g., always predicting up) or
   statistically structured.

Baselines included:
  - NaiveDirectional: always predicts the most frequent class (or last observed direction)
  - RandomBaseline: random predictions (calibrated to class distribution)
  - LinearRegression: OLS regression on feature matrix
  - LogisticRegression: linear classification baseline (sklearn-compatible)

All models are sklearn-compatible (fit/predict interface). This allows them
to be plugged directly into the walk-forward validation pipeline.

Usage:
    from src.models.baselines import NaiveDirectionalClassifier, build_baseline_pipeline
    model = build_baseline_pipeline("logistic")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Naive Directional Classifier
# ---------------------------------------------------------------------------

class NaiveDirectionalClassifier(BaseEstimator, ClassifierMixin):
    """
    Naive baseline: always predicts the majority class.

    This is the minimum bar any classifier must beat. If your ML model cannot
    outperform this, it has no directional skill.

    Parameters
    ----------
    strategy : str
        "majority": always predict the most frequent class.
        "prior": sample from the empirical class distribution.

    Attributes
    ----------
    majority_class_ : int
        The most frequent class observed during training.
    class_distribution_ : dict
        Empirical class proportions from training data.
    """

    def __init__(self, strategy: str = "majority"):
        self.strategy = strategy

    def fit(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        self.majority_class_ = int(classes[np.argmax(counts)])
        self.class_distribution_ = dict(zip(classes, counts / counts.sum()))
        logger.info(f"NaiveDirectional fitted. Majority class: {self.majority_class_}. "
                    f"Distribution: {self.class_distribution_}")
        return self

    def predict(self, X):
        n = len(X)
        if self.strategy == "majority":
            return np.full(n, self.majority_class_)
        elif self.strategy == "prior":
            classes = list(self.class_distribution_.keys())
            probs = list(self.class_distribution_.values())
            return np.random.choice(classes, size=n, p=probs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def predict_proba(self, X):
        # Returns constant probability equal to training proportions
        n = len(X)
        classes = sorted(self.class_distribution_.keys())
        probs = [[self.class_distribution_.get(c, 0.0) for c in classes]] * n
        return np.array(probs)


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def build_baseline_pipeline(
    model_type: str = "logistic",
    scale: bool = True,
    **model_kwargs,
) -> Pipeline:
    """
    Build a sklearn Pipeline for a given baseline model type.

    Optionally applies StandardScaler before the estimator.
    All models expose a consistent fit/predict interface.

    Parameters
    ----------
    model_type : str
        Type of baseline model. Options:
        - "naive": NaiveDirectionalClassifier (majority)
        - "logistic": LogisticRegression
        - "linear": LinearRegression (for regression tasks)
    scale : bool
        If True, apply StandardScaler before the estimator.
    **model_kwargs
        Additional keyword arguments passed to the estimator constructor.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A fitted-ready pipeline.

    Raises
    ------
    ValueError
        If `model_type` is not recognized.

    Example
    -------
    >>> pipeline = build_baseline_pipeline("logistic", C=0.1)
    >>> pipeline.fit(X_train, y_train)
    >>> preds = pipeline.predict(X_test)
    """
    if model_type == "naive":
        estimator = NaiveDirectionalClassifier(**model_kwargs)
        steps = [("model", estimator)]

    elif model_type == "logistic":
        defaults = dict(penalty="l2", C=1.0, max_iter=1000, solver="lbfgs",
                        class_weight="balanced", random_state=42)
        defaults.update(model_kwargs)
        estimator = LogisticRegression(**defaults)
        steps = ([("scaler", StandardScaler())] if scale else []) + [("model", estimator)]

    elif model_type == "linear":
        estimator = LinearRegression(**model_kwargs)
        steps = ([("scaler", StandardScaler())] if scale else []) + [("model", estimator)]

    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. "
                         "Choose from: 'naive', 'logistic', 'linear'.")

    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Hypothesis testing utility
# ---------------------------------------------------------------------------

def test_directional_significance(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Test whether the hit ratio of a directional predictor is significantly
    greater than chance (50%) using a one-sided binomial test.

    Parameters
    ----------
    y_true : np.ndarray
        True labels ∈ {-1, 1}.
    y_pred : np.ndarray
        Predicted labels ∈ {-1, 1}.

    Returns
    -------
    dict
        Dictionary with keys: hit_ratio, n_correct, n_total, p_value.

    Notes
    -----
    A low p-value does NOT imply economic significance. Statistical significance
    alone is insufficient — see backtest metrics for financial evaluation.
    TODO: Apply multiple testing corrections if testing many feature subsets.
    """
    from scipy.stats import binom_test  # type: ignore

    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    hit_ratio = correct / total

    # One-sided: p(hit ratio > 0.5 by chance)
    p_value = binom_test(correct, total, p=0.5, alternative="greater")

    result = {
        "hit_ratio": hit_ratio,
        "n_correct": int(correct),
        "n_total": int(total),
        "p_value": float(p_value),
    }
    logger.info(f"Directional significance test: {result}")
    return result
