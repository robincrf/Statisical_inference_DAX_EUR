# src/

This directory contains all **production-quality Python source code** for the project. Functions and classes here are intended to be reusable, testable, and importable from notebooks and scripts.

The philosophy: **notebooks explore, `src/` consolidates**.

---

## Module Structure

```
src/
├── data/               # Data ingestion and cleaning
│   ├── load_data.py    # Loading raw files, parsing, standardizing columns
│   └── clean_data.py   # Handling missing values, duplicates, frequency alignment
│
├── features/           # Feature engineering and selection
│   ├── build_features.py    # Compute features: returns, momentum, volatility, etc.
│   └── feature_selection.py # Filter, rank, and select features for modeling
│
├── models/             # Model definitions and training
│   ├── baselines.py    # Naive, linear, and logistic regression baselines
│   ├── train_ml.py     # ML model training pipeline (sklearn-compatible)
│   └── predict.py      # Generate predictions and map to trading signals
│
├── evaluation/         # Validation and metrics
│   ├── metrics.py      # Financial and predictive performance metrics
│   └── walk_forward.py # Walk-forward / expanding window validation engine
│
├── backtest/           # Backtesting infrastructure
│   ├── engine.py       # Vectorized backtest engine
│   └── costs.py        # Transaction cost and slippage models
│
├── visualization/      # Plotting utilities
│   └── plots.py        # Reusable figures: equity curves, drawdowns, feature importance
│
└── utils/              # General utilities
    ├── config.py        # YAML configuration loader
    ├── logging_utils.py # Standardized logging setup
    └── helpers.py       # General-purpose utility functions
```

---

## Conventions

- All functions must have **type-annotated signatures** and **NumPy-style docstrings**.
- No hard-coded file paths. All paths must come from configuration files.
- All time-series operations must be **look-ahead safe** — verify every `shift()` and `rolling()` call.
- Functions should be **pure** where possible (no side effects; return values explicitly).
- Do not use `print()` in production code; use the logging utilities from `utils/logging_utils.py`.

---

## Adding a New Module

1. Create the `.py` file in the appropriate sub-directory.
2. Add a docstring at the module level describing its purpose.
3. Add corresponding unit tests in `tests/`.
4. Import it from notebooks via `from src.submodule.module import function`.
