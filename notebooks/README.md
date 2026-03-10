# notebooks/

This directory contains the analysis notebooks for the project, organized in a **logical sequential order** that mirrors the research pipeline. Each notebook corresponds to a distinct analytical stage, from raw data auditing to final results synthesis.

---

## Notebook Index

| # | Notebook | Purpose |
|---|---|---|
| 01 | `01_data_audit.ipynb` | Data quality, completeness, frequency checks |
| 02 | `02_eda_and_market_structure.ipynb` | Exploratory analysis, return distributions, autocorrelations |
| 03 | `03_feature_engineering.ipynb` | Feature construction and preliminary inspection |
| 04 | `04_statistical_baselines.ipynb` | Naive baselines, linear models, hypothesis tests |
| 05 | `05_ml_models.ipynb` | Machine learning model training and evaluation |
| 06 | `06_walk_forward_validation.ipynb` | Temporal cross-validation framework |
| 07 | `07_backtest_and_costs.ipynb` | Signal-to-position mapping, PnL, Sharpe, drawdown |
| 08 | `08_regime_dependence.ipynb` | Conditional performance by market regime |
| 09 | `09_final_results.ipynb` | Final comparison table, figures, synthesis |

---

## Usage Guidelines

1. **Run notebooks in order.** Each notebook depends on outputs from the previous ones (processed data, features, models).
2. **Do not commit large output cells.** Clear cell outputs before committing; use `nbstripout` or equivalent.
3. **Do not fill in conclusions before the empirical work is done.** All `TODO` markers in notebooks indicate sections that require user input based on actual results.
4. **Notebooks are exploratory.** Production logic should be refactored into `src/` before being considered final.

---

## Dependencies

All notebooks import from `src/` using either relative paths or a properly configured Python path. Ensure the project root is in `sys.path`:

```python
import sys
sys.path.insert(0, "..") # from the notebooks/ directory
```

Or configure `pyproject.toml` / `setup.py` for editable install.

---

## Naming Convention

Notebooks are prefixed with a two-digit number to enforce ordering. Do not rename or reorder without updating this README.
