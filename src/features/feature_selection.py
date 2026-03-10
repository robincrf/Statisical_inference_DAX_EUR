"""
feature_selection.py
====================
Feature selection module for the DAX/EURO STOXX Futures research project.

This module provides utilities to rank and filter the feature set before
model training, reducing noise and improving generalizability.

Feature selection in financial modeling is particularly delicate:
- Avoid any selection criterion that uses future data
- Do not select features based on in-sample performance on the test set
- Prefer stability-aware methods over pure in-sample correlation

Methods available:
  - Correlation filter (remove highly collinear features)
  - Information gain / mutual information ranking
  - Variance threshold (remove near-constant features)
  - Model-based importance (post-hoc, from a fitted baseline)

Usage:
    from src.features.feature_selection import select_features
    selected = select_features(X_train, y_train, cfg)
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Master selection pipeline
# ---------------------------------------------------------------------------

def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict,
    task_type: str = "classification",
) -> List[str]:
    """
    Select a subset of features based on the configuration.

    All selection criteria are computed exclusively on training data.
    No test-set leakage must occur.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix (rows = samples, columns = feature names).
    y_train : pd.Series
        Training target variable.
    cfg : dict
        Feature selection config from features_config.yaml.
    task_type : str
        "classification" or "regression". Determines mutual info function.

    Returns
    -------
    List[str]
        Ordered list of selected feature names.
    """
    features = list(X_train.columns)
    logger.info(f"Feature selection started with {len(features)} candidate features.")

    # Step 1: Remove near-constant features
    features = variance_threshold_filter(X_train[features], min_variance=1e-6)

    # Step 2: Remove highly correlated features
    method = cfg.get("method", "correlation")
    max_feats = cfg.get("max_features", len(features))

    if method == "correlation":
        features = correlation_filter(X_train[features], threshold=0.90)

    elif method == "mutual_info":
        scores = mutual_info_scores(X_train[features], y_train, task_type)
        features = scores.nlargest(max_feats).index.tolist()

    elif method == "model_based":
        features = model_based_importance(X_train[features], y_train, task_type, top_n=max_feats)

    else:
        logger.warning(f"Unknown feature selection method: {method}. Returning all features.")

    logger.info(f"Feature selection complete. {len(features)} features selected.")
    # TODO: Log or plot the feature importances/scores for inspection

    return features


# ---------------------------------------------------------------------------
# Variance threshold
# ---------------------------------------------------------------------------

def variance_threshold_filter(
    X: pd.DataFrame,
    min_variance: float = 1e-6,
) -> List[str]:
    """
    Remove features with near-zero variance (quasi-constant).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    min_variance : float
        Minimum variance threshold. Features below this are removed.

    Returns
    -------
    List[str]
        Remaining feature names.
    """
    variances = X.var(ddof=1)
    selected = variances[variances > min_variance].index.tolist()
    n_removed = len(X.columns) - len(selected)
    if n_removed > 0:
        logger.info(f"Variance filter removed {n_removed} near-constant features.")
    return selected


# ---------------------------------------------------------------------------
# Correlation filter
# ---------------------------------------------------------------------------

def correlation_filter(
    X: pd.DataFrame,
    threshold: float = 0.90,
    method: str = "pearson",
) -> List[str]:
    """
    Remove features that are highly correlated with another feature.

    Uses a greedy approach: given a pair of features with |correlation| > threshold,
    remove the second one (assuming they are ordered by some prior ranking).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    threshold : float
        Absolute correlation threshold (e.g., 0.90 removes features with r > 0.90).
    method : str
        Correlation method. Options: "pearson", "spearman".

    Returns
    -------
    List[str]
        Remaining feature names after removing highly correlated pairs.

    TODO: Consider ordering features by predictive score before applying this filter.
    """
    corr = X.corr(method=method).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Identify columns to drop
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    selected = [col for col in X.columns if col not in to_drop]

    logger.info(f"Correlation filter (threshold={threshold}): "
                f"{len(to_drop)} features removed, {len(selected)} retained.")
    return selected


# ---------------------------------------------------------------------------
# Mutual information
# ---------------------------------------------------------------------------

def mutual_info_scores(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = "classification",
    random_state: int = 42,
) -> pd.Series:
    """
    Compute mutual information scores between each feature and the target.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (must be free of NaN).
    y : pd.Series
        Target variable.
    task_type : str
        "classification" or "regression".
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.Series
        Series of MI scores indexed by feature name, sorted descending.
    """
    # Drop rows with NaN
    mask = X.notna().all(axis=1) & y.notna()
    X_clean = X[mask]
    y_clean = y[mask]

    if task_type == "classification":
        scores = mutual_info_classif(X_clean, y_clean, random_state=random_state)
    else:
        scores = mutual_info_regression(X_clean, y_clean, random_state=random_state)

    return pd.Series(scores, index=X.columns).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Model-based importance
# ---------------------------------------------------------------------------

def model_based_importance(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = "classification",
    top_n: int = 20,
    random_state: int = 42,
) -> List[str]:
    """
    Estimate feature importance using a shallow Random Forest.

    NOTE: This should only be applied to training data. Using it on the full
    dataset (including test) constitutes look-ahead bias.

    The fitted model here is used solely for feature ranking — it is NOT
    the final production model.

    Parameters
    ----------
    X : pd.DataFrame
        Training feature matrix.
    y : pd.Series
        Training target variable.
    task_type : str
        "classification" or "regression".
    top_n : int
        Number of top features to return.
    random_state : int
        Random seed.

    Returns
    -------
    List[str]
        Top `top_n` feature names ranked by importance.

    TODO: Validate stability of importance across walk-forward folds.
    """
    mask = X.notna().all(axis=1) & y.notna()
    X_clean = X[mask]
    y_clean = y[mask]

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, max_depth=5,
                                       random_state=random_state, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=5,
                                      random_state=random_state, n_jobs=-1)

    model.fit(X_clean, y_clean)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    logger.info(f"Top {top_n} features by model importance:\n{importances.head(top_n)}")
    return importances.head(top_n).index.tolist()
