# Statistical Inference & Machine Learning on DAX and EURO STOXX Futures

> **Research status:** In progress — framework prepared, empirical results pending.

---

## Overview

This project investigates whether statistically grounded and machine-learning-derived signals can generate **exploitable predictive power** on **DAX (FDAX)** and **EURO STOXX 50 (FESX)** futures, after accounting for transaction costs, temporal dependence, and out-of-sample robustness.

The work follows a rigorous **quantitative research methodology**: baseline statistical models are established first, then progressively more complex ML estimators are introduced and compared under a strict **walk-forward validation** framework. All performance claims must withstand empirical scrutiny across multiple market regimes.

---

## Research Questions

1. Do lagged returns, volatility, and momentum features contain statistically significant predictive information on DAX/EURO STOXX futures at daily (and/or intraday) frequencies?
2. How do ML classifiers and regressors compare against naive and linear baselines after controlling for data-snooping and look-ahead bias?
3. Does predictive performance degrade materially once realistic transaction costs and slippage are incorporated?
4. To what extent is performance regime-dependent (trending vs. mean-reverting, low vs. high volatility, stress vs. calm)?
5. Is there a meaningful performance differential between DAX and EURO STOXX signals, and can cross-asset features add value?

---

## Objectives

- Build a **reproducible, auditable pipeline** from raw data to evaluated trading signals.
- Establish statistically sound **baselines** before introducing any ML complexity.
- Apply **proper temporal cross-validation** (walk-forward) to avoid look-ahead contamination.
- Measure performance using both **predictive** (accuracy, hit ratio) and **financial** (Sharpe ratio, max drawdown, turnover) metrics.
- Conduct a **regime-conditional analysis** to assess robustness and generalizability.
- Produce a report suitable for **academic submission** or **portfolio presentation**.

---

## Assets Studied

| Ticker | Full Name | Source | Frequency |
|---|---|---|---|
| `^GDAXI` | DAX Performance Index | Yahoo Finance (`yfinance`) | Daily |
| `^STOXX50E` | EURO STOXX 50 Index | Yahoo Finance (`yfinance`) | Daily |

Both series are **cash equity indices** (not derivatives) downloaded via the `yfinance` Python library. They serve as proxies for systematic signal research; real-world execution would require a replication vehicle (index ETF, CFD, or futures). Volume data for these tickers is not meaningful and is treated as unavailable.

---

## Working Hypotheses

> These are **research hypotheses to be tested**, not conclusions.

- **H1 (Predictability):** Short-horizon returns of FDAX/FESX are not fully random; some statistical structure is detectable.
- **H2 (ML vs. Linear):** Non-linear models (tree-based ensembles) capture patterns not captured by linear baselines.
- **H3 (Costs):** After accounting for bid-ask spread and commission, alpha is significantly reduced or eliminated for high-frequency signals.
- **H4 (Regimes):** Model performance varies significantly across volatility regimes and trend/mean-reversion periods.
- **H5 (Cross-asset):** DAX and EURO STOXX signals exhibit partial co-movement, and cross-asset features may enhance prediction.

---

## Methodology

```
Raw Data
   │
   ▼
Data Audit & Cleaning
   │
   ▼
Feature Engineering
   │
   ▼
Statistical Baselines  ──►  Significance Tests
   │
   ▼
ML Model Training (Walk-Forward)
   │
   ▼
Signal Generation → Position Sizing
   │
   ▼
Backtest with Transaction Costs
   │
   ▼
Regime-Conditional Evaluation
   │
   ▼
Final Model Comparison & Reporting
```

All steps are implemented with **no look-ahead bias**. Validation is strictly temporal. No hyperparameter is tuned on the test set.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── configs/                    # YAML configuration files
│   ├── project_config.yaml
│   ├── data_config.yaml
│   ├── features_config.yaml
│   ├── models_config.yaml
│   └── backtest_config.yaml
│
├── data/                       # Data (never committed if proprietary)
│   ├── raw/                    # Original, immutable data
│   ├── interim/                # Intermediate transformations
│   ├── processed/              # Final clean datasets ready for modeling
│   └── external/               # External data (economic calendars, etc.)
│
├── notebooks/                  # Analysis notebooks (numbered, ordered)
│   ├── 01_data_audit.ipynb
│   ├── 02_eda_and_market_structure.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_statistical_baselines.ipynb
│   ├── 05_ml_models.ipynb
│   ├── 06_walk_forward_validation.ipynb
│   ├── 07_backtest_and_costs.ipynb
│   ├── 08_regime_dependence.ipynb
│   └── 09_final_results.ipynb
│
├── src/                        # All production Python source code
│   ├── data/
│   │   ├── load_data.py
│   │   └── clean_data.py
│   ├── features/
│   │   ├── build_features.py
│   │   └── feature_selection.py
│   ├── models/
│   │   ├── baselines.py
│   │   ├── train_ml.py
│   │   └── predict.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── walk_forward.py
│   ├── backtest/
│   │   ├── engine.py
│   │   └── costs.py
│   ├── visualization/
│   │   └── plots.py
│   └── utils/
│       ├── config.py
│       ├── logging_utils.py
│       └── helpers.py
│
├── reports/                    # Output artifacts
│   ├── figures/
│   ├── tables/
│   ├── results_template.md
│   └── presentation_notes.md
│
├── docs/                       # Research documentation
│   └── report_outline.md
│
├── experiments/                # Tracked experiment runs (MLflow / manual)
│
└── tests/                      # Unit tests
    ├── test_load_data.py
    ├── test_build_features.py
    ├── test_walk_forward.py
    └── test_metrics.py
```

---

## Recommended Workflow

```
1. Configure  →  Edit configs/*.yaml to match your data and research choices
2. Ingest     →  Run src/data/load_data.py to standardize raw files
3. Clean      →  Run src/data/clean_data.py to handle gaps and outliers
4. Features   →  Run src/features/build_features.py to generate feature matrix
5. Explore    →  Work through notebooks 01 → 03
6. Baseline   →  Notebook 04 + src/models/baselines.py
7. ML         →  Notebook 05 + src/models/train_ml.py
8. Validate   →  Notebook 06 + src/evaluation/walk_forward.py
9. Backtest   →  Notebook 07 + src/backtest/engine.py
10. Regimes   →  Notebook 08
11. Report    →  Notebook 09 + docs/report_outline.md
```

---

## Getting Started

```bash
# Clone the repository
git clone <repo_url>
cd Statistical_inference_analysis_DAX_EUR

# Create and activate environment
conda env create -f environment.yml
conda activate dax_research

# OR using pip
pip install -r requirements.txt

# Launch Jupyter
jupyter lab

# Run tests
pytest tests/ -v
```

---

## Dependencies

See `requirements.txt` and `environment.yml` for the full list.  
Core packages: `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `xgboost`, `pyyaml`, `pytest`, `jupyter`.

---

## Limitations

- All conclusions are contingent on the quality and completeness of the input data.
- In-sample model selection introduces implicit bias; walk-forward validation mitigates but does not eliminate this.
- Transaction cost models are simplified; actual execution costs may differ significantly.
- The project does not model market impact, order book dynamics, or intraday liquidity constraints.
- Past statistical regularities may not persist in future market regimes.
- No guarantee of out-of-sample performance is implied by any in-sample result.

---

## Research Roadmap

- [ ] Ingest and validate raw FDAX / FESX data
- [ ] Complete feature engineering pipeline
- [ ] Establish and document statistical baselines
- [ ] Run walk-forward ML experiments
- [ ] Implement full backtest with cost model
- [ ] Regime analysis
- [ ] Write final report / paper skeleton
- [ ] Package experiments for reproducibility

---

## Next Steps

After completing empirical analysis:
- Compare results against benchmark (buy-and-hold, index return)
- Sensitivity analysis on key hyperparameters
- Extend to intraday data if available
- Consider alternative assets (FGBL, FSMI) for robustness checks

---

*This project was prepared as a rigorous research framework. No empirical results are presented in this document. All analytical conclusions are to be derived from data.*
