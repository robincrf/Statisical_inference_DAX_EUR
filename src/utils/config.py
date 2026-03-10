"""
config.py
=========
Configuration loader utility for the DAX/EURO STOXX Futures research project.

Provides a simple, consistent interface to load and access YAML configuration
files across all project modules.

Usage:
    from src.utils.config import load_config
    cfg = load_config("configs/data_config.yaml")
    tickers = cfg["assets"]
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(filepath: str, root: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return it as a Python dictionary.

    Parameters
    ----------
    filepath : str
        Relative or absolute path to the YAML config file.
    root : str, optional
        Optional root directory to prepend to `filepath` if it is relative.
        Defaults to the current working directory.

    Returns
    -------
    dict
        Parsed YAML configuration.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist at the resolved path.
    yaml.YAMLError
        If the YAML file cannot be parsed.
    """
    if root is None:
        root = os.getcwd()

    path = Path(root) / filepath if not Path(filepath).is_absolute() else Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML config at {path}: {e}")

    logger.debug(f"Loaded config: {path}")
    return cfg


def get(cfg: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Safely access a nested config value using dot-notation.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    key_path : str
        Dot-separated key path, e.g. "position.signal_lag".
    default : Any
        Default value if the key path is not found.

    Returns
    -------
    Any
        The config value, or `default` if not found.

    Example
    -------
    >>> lag = get(cfg, "position.signal_lag", default=1)
    """
    keys = key_path.split(".")
    value = cfg
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default
