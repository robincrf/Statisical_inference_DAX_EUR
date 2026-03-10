"""
train_ml.py
===========
Machine learning training pipeline for the DAX/EURO STOXX Futures research project.

This module provides a generic, sklearn-compatible training interface that:
- Accepts any ML estimator via a unified interface
- Applies optional feature scaling
- Records training metadata and fit timestamps
- Does NOT fit on the test set (enforced by design)

Model hyperparameter tuning must be done on the validation set only.
Final model evaluation is reserved for the held-out test period.

Usage:
    from src.models.train_ml import build_ml_pipeline, train_model
    pipeline = build_ml_pipeline("random_forest", task_type="classification", cfg=cfg)
    result = train_model(pipeline, X_train, y_train, model_name="random_forest")
"""

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def build_ml_pipeline(
    model_type: str,
    task_type: str = "classification",
    cfg: Optional[dict] = None,
    scale: bool = True,
) -> Pipeline:
    """
    Build a sklearn Pipeline for a given ML model type.

    Model hyperparameters are read from `cfg` (models_config.yaml section)
    if provided, otherwise defaults are used. All defaults are deliberately
    conservative to avoid overfitting out-of-the-box.

    Parameters
    ----------
    model_type : str
        Model identifier. Options: "random_forest", "gradient_boosting",
        "xgboost", "svm".
    task_type : str
        "classification" or "regression".
    cfg : dict, optional
        Sub-configuration dict for this model from models_config.yaml.
        If None, default hyperparameters are applied.
    scale : bool
        If True, prepend a StandardScaler. Recommended for SVM, optional for trees.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Untrained pipeline ready for fitting.

    Raises
    ------
    ValueError
        If model_type or task_type is unrecognized.

    TODO: Add calibration step (CalibratedClassifierCV) for probability outputs.
    """
    cfg = cfg or {}
    estimator = _instantiate_estimator(model_type, task_type, cfg)

    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))

    logger.info(f"Built pipeline: [{' → '.join([s[0] for s in steps])}]")
    return Pipeline(steps)


def _instantiate_estimator(model_type: str, task_type: str, cfg: dict) -> Any:
    """
    Instantiate a scikit-learn-compatible estimator from config.

    Parameters
    ----------
    model_type : str
    task_type : str
    cfg : dict
        Hyperparameter dict for this model.

    Returns
    -------
    Estimator object.
    """
    rs = cfg.get("random_state", 42)

    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        params = dict(
            n_estimators=cfg.get("n_estimators", 100),
            max_depth=cfg.get("max_depth", None),
            min_samples_leaf=cfg.get("min_samples_leaf", 20),
            max_features=cfg.get("max_features", "sqrt"),
            class_weight=cfg.get("class_weight", "balanced") if task_type == "classification" else None,
            n_jobs=-1,
            random_state=rs,
        )
        return RandomForestClassifier(**params) if task_type == "classification" \
            else RandomForestRegressor(**{k: v for k, v in params.items() if k != "class_weight"})

    elif model_type == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        params = dict(
            n_estimators=cfg.get("n_estimators", 200),
            learning_rate=cfg.get("learning_rate", 0.05),
            max_depth=cfg.get("max_depth", 3),
            subsample=cfg.get("subsample", 0.8),
            random_state=rs,
        )
        return GradientBoostingClassifier(**params) if task_type == "classification" \
            else GradientBoostingRegressor(**params)

    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        params = dict(
            n_estimators=cfg.get("n_estimators", 200),
            learning_rate=cfg.get("learning_rate", 0.05),
            max_depth=cfg.get("max_depth", 3),
            colsample_bytree=cfg.get("colsample_bytree", 0.8),
            subsample=cfg.get("subsample", 0.8),
            random_state=rs,
            eval_metric=cfg.get("eval_metric", "logloss"),
            verbosity=0,
        )
        return XGBClassifier(**params) if task_type == "classification" \
            else XGBRegressor(**params)

    elif model_type == "svm":
        from sklearn.svm import SVC, SVR
        params = dict(
            kernel=cfg.get("kernel", "rbf"),
            C=cfg.get("C", 1.0),
            gamma=cfg.get("gamma", "scale"),
            probability=True if task_type == "classification" else False,
        )
        return SVC(**params) if task_type == "classification" else SVR(**params)

    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. "
                         "Options: random_forest, gradient_boosting, xgboost, svm.")


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------

def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "model",
) -> Dict[str, Any]:
    """
    Fit a sklearn pipeline and return metadata about the training run.

    Parameters
    ----------
    pipeline : Pipeline
        Untrained sklearn-compatible pipeline.
    X_train : pd.DataFrame
        Training feature matrix (NaN-free; rows with NaN must be dropped before).
    y_train : pd.Series
        Training labels.
    model_name : str
        Human-readable name for logging.

    Returns
    -------
    dict
        Training result dict with keys:
        - ``model_name``: str
        - ``pipeline``: fitted Pipeline
        - ``n_train_samples``: int
        - ``feature_names``: list of str
        - ``train_start``: str (index start)
        - ``train_end``: str (index end)
        - ``training_time_sec``: float

    Notes
    -----
    This function does NOT perform any validation or test evaluation.
    Use walk_forward.py for proper temporal evaluation.
    """
    # Drop NaNs
    mask = X_train.notna().all(axis=1) & y_train.notna()
    X_ft = X_train[mask]
    y_ft = y_train[mask]

    logger.info(f"Training '{model_name}' on {len(X_ft)} samples "
                f"({mask.sum()} / {len(X_train)} valid rows).")

    t0 = time.time()
    pipeline.fit(X_ft, y_ft)
    elapsed = time.time() - t0

    result = {
        "model_name": model_name,
        "pipeline": pipeline,
        "n_train_samples": len(X_ft),
        "feature_names": list(X_ft.columns),
        "train_start": str(X_ft.index[0]) if len(X_ft) > 0 else None,
        "train_end": str(X_ft.index[-1]) if len(X_ft) > 0 else None,
        "training_time_sec": round(elapsed, 3),
    }

    logger.info(f"'{model_name}' trained in {elapsed:.2f}s.")
    return result
