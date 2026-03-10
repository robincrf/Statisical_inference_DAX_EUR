"""
logging_utils.py
================
Logging configuration for the DAX/EURO STOXX Futures research project.

Provides a centralized setup function that configures both:
- Console output (colored by log level if `colorlog` is available)
- File output (rotating log file)

Usage:
    from src.utils.logging_utils import setup_logging
    setup_logging(level="INFO", log_file="logs/pipeline.log")
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """
    Configure project-wide logging.

    Parameters
    ----------
    level : str
        Logging level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    log_file : str, optional
        Path to log file. If None, only console logging is used.
        Parent directory is created automatically.
    format_str : str, optional
        Custom logging format string. If None, a sensible default is used.

    Returns
    -------
    logging.Logger
        Configured root logger.
    """
    if format_str is None:
        format_str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicate output in notebooks
    root_logger.handlers.clear()

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S"))
    root_logger.addHandler(console_handler)

    # --- File handler (rotating) ---
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S"))
        root_logger.addHandler(file_handler)

    root_logger.info(f"Logging initialized at level={level}.")
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger.

    Use this in module-level code to get a logger specific to that module:
        logger = get_logger(__name__)

    Parameters
    ----------
    name : str
        Logger name, typically `__name__` from the calling module.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)
