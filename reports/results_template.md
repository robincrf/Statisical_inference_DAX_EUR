# results_template.md
# Template for reporting final empirical results.
# Fill in each section after completing all analysis notebooks.
# Do NOT write conclusions before the data has been analyzed.

---

# Final Results Summary
**Project:** Statistical Inference and ML on DAX and EURO STOXX 50 Indices  
**Data:** `^GDAXI` (DAX) and `^STOXX50E` (EURO STOXX 50) — Yahoo Finance  
**Test Period:** PLACEHOLDER (from data_config.yaml)  
**Prepared by:** PLACEHOLDER  
**Date:** PLACEHOLDER  

---

## 1. Data Overview

| Metric | DAX (`^GDAXI`) | EURO STOXX 50 (`^STOXX50E`) |
|---|---|---|
| Period | PLACEHOLDER | PLACEHOLDER |
| N observations (train) | PLACEHOLDER | PLACEHOLDER |
| Ann. return (full period) | PLACEHOLDER | PLACEHOLDER |
| Ann. volatility (full period) | PLACEHOLDER | PLACEHOLDER |
| Sharpe (buy & hold) | PLACEHOLDER | PLACEHOLDER |
| Max drawdown (buy & hold) | PLACEHOLDER | PLACEHOLDER |

---

## 2. Feature Set

**Features used in final model:**

| Feature | Type | Window |
|---|---|---|
| PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |

**Features discarded (and reason):**

| Feature | Reason for exclusion |
|---|---|
| PLACEHOLDER | PLACEHOLDER |

---

## 3. Statistical Baselines — Validation Set

| Model | Hit Ratio | p-value (Binomial) | Sharpe (gross) | Sharpe (net) | MDD |
|---|---|---|---|---|---|
| Naive (majority) | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| Logistic Regression | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |

---

## 4. ML Models — Validation Set

| Model | Hit Ratio | Sharpe (net) | MDD | Turnover | Notes |
|---|---|---|---|---|---|
| Random Forest | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| XGBoost | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |

**Best model selected:** PLACEHOLDER  
**Selection criterion:** PLACEHOLDER (e.g., validation Sharpe net of costs)

---

## 5. Walk-Forward Results — All Folds

| Fold | Train Period | Test Period | Sharpe | MDD | Hit Ratio |
|---|---|---|---|---|---|
| 1 | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| 2 | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| 3 | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| 4 | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| 5 | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| **Avg** | — | — | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |

---

## 6. Final Backtest — Test Set

**Cost assumptions:**
- Total round-trip cost: PLACEHOLDER bps
- Replication vehicle: PLACEHOLDER (ETF / CFD / futures)
- Signal lag: 1 period

| Model | Sharpe | Ann. Return | Ann. Vol. | MDD | Turnover | Hit Ratio |
|---|---|---|---|---|---|---|
| Naive | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| Logistic | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| Best ML | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| Buy & Hold | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | N/A | N/A |

---

## 7. Regime Analysis

| Model | Regime | Sharpe | MDD | Ann. Return |
|---|---|---|---|---|
| Best ML | High Vol | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| Best ML | Low Vol | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |

---

## 8. Cost Sensitivity (Test Set, Best ML)

| Round-Trip Cost (bps) | Sharpe | Ann. Return |
|---|---|---|
| 0 bps | PLACEHOLDER | PLACEHOLDER |
| 1 bps | PLACEHOLDER | PLACEHOLDER |
| 2 bps | PLACEHOLDER | PLACEHOLDER |
| 5 bps | PLACEHOLDER | PLACEHOLDER |
| 10 bps | PLACEHOLDER | PLACEHOLDER |

---

## 9. Key Findings

> **TODO:** Write 3–5 factual observations from the data, not conclusions.

1. PLACEHOLDER
2. PLACEHOLDER
3. PLACEHOLDER

---

## 10. Main Limitations

> **TODO:** Document limitations specific to your actual results.

1. PLACEHOLDER
2. PLACEHOLDER
3. PLACEHOLDER
