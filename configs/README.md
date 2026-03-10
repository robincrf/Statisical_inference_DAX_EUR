# configs/

This directory contains all **YAML configuration files** that control the behavior of the pipeline. The goal is a fully configurable project where no analytical decision is hard-coded in source files.

---

## Config Files

| File | Purpose |
|---|---|
| `project_config.yaml` | Global project metadata and paths |
| `data_config.yaml` | Data sources, tickers, frequency, date ranges |
| `features_config.yaml` | Feature list, rolling windows, lag parameters |
| `models_config.yaml` | Model types, hyperparameter grids, classification vs. regression |
| `backtest_config.yaml` | Cost structure, position sizing, slippage, hold periods |

---

## How Configs Are Loaded

```python
from src.utils.config import load_config

cfg = load_config("configs/project_config.yaml")
data_cfg = load_config("configs/data_config.yaml")
```

---

## Guidelines

- All values in YAML files marked with `# PLACEHOLDER` must be filled in before running the corresponding pipeline step.
- Do not commit API keys, credentials, or data paths specific to your local machine. Use environment variables for sensitive values.
- When a parameter has multiple valid options, they are listed as YAML comments.
