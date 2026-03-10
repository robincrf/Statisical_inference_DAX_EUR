# report_outline.md
# Research Report / Paper Skeleton
# Project: Statistical Inference and ML on DAX and EURO STOXX 50 Indices


---

# Statistical Inference and Machine Learning on DAX and EURO STOXX 50 Indices

**Author:** PLACEHOLDER  
**Institution:** PLACEHOLDER  
**Date:** PLACEHOLDER  
**Status:** In progress — empirical results pending

---

## Abstract

> **TODO:** Write a 150–200 word abstract after all empirical results are complete.  
> Should cover: research question, data, methods, key findings (if any), and main limitation.  
> Do NOT write speculative conclusions.

---

## 1. Introduction

**Objective:** Contextualize the problem and motivate the research.

**To cover:**
- Why European equity indices (DAX, EURO STOXX 50)?
- Why ML-based signals vs. simple technical rules?
- What is the specific predictive question?
- What gap in the existing literature does this address?

**Tables / Figures:**
- Summary statistics table of both indices (mean return, vol, SR of buy-and-hold)

**Pitfalls to avoid:**
- Do not overstate the novelty — remain empirically grounded
- Do not claim results before empirical analysis is complete

---

## 2. Literature Review and Theoretical Motivation

**Objective:** Ground the research in prior work. Not exhaustive — 10–20 key references.

**To cover:**
- Weak-form market efficiency and its critics
- Statistical persistence in equity returns (momentum, reversal)
- ML applications in asset return prediction (brief survey)
- Transaction cost impact on systematic strategies
- Regime-conditional predictability

**Key references to consider:**
- Jegadeesh & Titman (1993) — Momentum
- Lo & MacKinlay (1988) — Return predictability
- Gu, Kelly, Xiu (2020) — ML in asset pricing
- Frazzini & Pedersen (2012) — Trading costs and alpha

**Pitfalls to avoid:**
- Do not present literature as supporting your (yet unobserved) results
- Do not claim that "prior literature shows ML works here" prematurely

---

## 3. Data

**Objective:** Describe the data precisely so results are reproducible.

**To cover:**
- Data source: Yahoo Finance (`yfinance`)
- Tickers: `^GDAXI` (DAX Performance Index) and `^STOXX50E` (EURO STOXX 50 Index)
- Period: PLACEHOLDER (from notebook 01)
- Frequency: daily
- Variables used: Adj Close (treated as close), Open, High, Low
- Note on volume: not meaningful for these tickers
- Cleaning decisions: PLACEHOLDER (from notebook 01)
- Temporal splits: train / validation / test (from data_config.yaml)

**Tables / Figures:**
- Table 1: Descriptive statistics (mean, std, skew, kurtosis, min, max) per asset
- Figure 1: Price series and rolling volatility
- Figure 2: Return distribution vs. Normal

**Pitfalls to avoid:**
- Do not conflate the index price with a tradable instrument
- Document that volume data is excluded and why

---

## 4. Methodology

**Objective:** Precisely describe the modeling and validation framework.

### 4.1 Feature Engineering
- Feature categories used (from features_config.yaml)
- Rolling window choices and rationale
- Look-ahead bias prevention measures

### 4.2 Prediction Targets
- Classification vs. regression choice (from models_config.yaml)
- Horizon: PLACEHOLDER
- Class balance: PLACEHOLDER (from notebook 03)

### 4.3 Validation Framework
- Walk-forward validation: expanding vs. rolling window
- Number of folds, fold sizes
- Gap parameter justification
- No data snooping post-test-set

**Figures:**
- Figure 3: Walk-forward fold diagram (from notebook 06)

**Pitfalls to avoid:**
- Must explicitly state that all hyperparameter choices were made on validation set
- Must confirm test set was touched only once

---

## 5. Statistical Baselines

**Objective:** Establish minimum performance benchmarks.

**To cover:**
- Naive directional classifier
- Logistic regression (features used, regularization)
- Binomial significance test on hit ratio
- Multiple testing correction (if applicable)

**Tables / Figures:**
- Table 2: Baseline results (hit ratio, p-value, Sharpe gross/net, MDD)

**Pitfalls to avoid:**
- Do not report only in-sample results
- Statistical significance ≠ economic significance — address both

---

## 6. Machine Learning Models

**Objective:** Present ML model results with honest comparison to baselines.

**To cover:**
- Model architectures used (Random Forest, XGBoost, etc.)
- Hyperparameter tuning procedure (validation set only)
- Feature importance analysis
- Calibration of probability outputs (if used)

**Tables / Figures:**
- Table 3: Validation-set comparison (all models)
- Figure 4: Feature importance chart

**Pitfalls to avoid:**
- Do not present validation results as final — test results are final
- Do not overfit to feature importance — use walk-forward importance

---

## 7. Walk-Forward Validation Results

**Objective:** Present unbiased out-of-sample performance across multiple folds.

**To cover:**
- Sharpe per fold for each model
- Consistency of performance vs. variability
- Evidence of regime sensitivity

**Tables / Figures:**
- Table 4: Walk-forward metrics per fold
- Figure 5: Fold-by-fold Sharpe bar chart

---

## 8. Backtest Design and Results

**Objective:** Translate signals into PnL under realistic cost assumptions.

**To cover:**
- Replication vehicle assumption (ETF, CFD, or futures)
- Cost model parameters (commission, slippage, spread)
- Position sizing rule
- Signal lag convention (confirm 1-period lag)
- Gross vs. net alpha decomposition
- Cost sensitivity analysis

**Tables / Figures:**
- Table 5: Final backtest metrics (test period)
- Figure 6: Equity curves — all models vs. buy-and-hold
- Figure 7: Drawdown curves

**Pitfalls to avoid:**
- Cost model is approximate — clearly state this limitation
- Do not present gross Sharpe as the primary result

---

## 9. Regime Analysis

**Objective:** Characterize under which market conditions the signals work.

**To cover:**
- Regime definition methodology (vol-based, trend-based, manual)
- Conditional performance per regime per model
- Implication for real-world deployment timing

**Tables / Figures:**
- Table 6: Regime conditional metrics
- Figure 8: Sharpe by regime per model

---

## 10. Discussion and Limitations

**Objective:** Honest, balanced assessment of the results.

**To cover:**
- What do results show vs. what they do NOT show?
- Transaction cost sensitivity
- Data limitations (daily only, no volume, index ≠ tradable instrument)
- Feature set limitations (price-only)
- Overfitting risk despite walk-forward methodology
- Non-stationarity of return distributions

**Pitfalls to avoid:**
- Do not overclaim — every result is an empirical finding, not a proof
- Limitations section must be specific, not generic

---

## 11. Conclusion

**Objective:** Concise summary of the research contribution.

**To cover:**
- Restate research question
- Summarize main empirical findings (factual, not interpreted)
- Identify the most promising direction for future research

> **TODO:** Write after all sections are complete. Do not pre-write conclusions.

---

## 12. References

> **TODO:** Add references in your preferred citation format (APA, IEEE, etc.)
> Use Zotero, Mendeley, or BibTeX for citation management.

---

## Appendix A — Configuration Details

Reproduce model and backtest parameters from YAML configs for full reproducibility.

## Appendix B — Additional Figures

Any supplementary charts not fitting in the main text.

## Appendix C — Code Availability

Link to GitHub repository.
