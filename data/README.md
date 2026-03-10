# data/

This directory contains all data files used in the project. It is organized to enforce a strict separation between raw, intermediate, and final datasets, following a **data immutability principle**: raw data is never modified in-place.

---

## Directory Structure

```
data/
├── raw/          # Original data as received — NEVER MODIFY
├── interim/      # Intermediate files produced during cleaning/processing
├── processed/    # Final, model-ready datasets
└── external/     # External reference data (economic calendars, holidays, VIX, etc.)
```

---

## Data Sources

> **TODO:** Fill in the actual data sources once acquired.

| Variable | Source | Format | Frequency | Coverage |
|---|---|---|---|---|
| FDAX (DAX Futures) | *TBD (e.g., Refinitiv, Bloomberg, Eurex)* | CSV / Parquet | Daily / Intraday | *TBD* |
| FESX (EURO STOXX 50 Futures) | *TBD* | CSV / Parquet | Daily / Intraday | *TBD* |
| Economic Calendar | *TBD (e.g., Investing.com, Bloomberg)* | CSV | Event-based | *TBD* |

---

## Data Conventions

- **Timestamps:** All timestamps must be in **UTC** or explicitly labelled with timezone. Naive timestamps are not accepted.
- **Columns:** Standardized to `open`, `high`, `low`, `close`, `volume` (where applicable), `datetime` (index).
- **Returns:** Computed as **log returns** unless stated otherwise: `r_t = log(P_t / P_{t-1})`.
- **Contracts:** Front-month continuous contracts are preferred; rollover methodology must be documented.
- **Gaps:** Session gaps (weekends, holidays) are expected and must not be filled with zeros or interpolated without explicit justification.

---

## Data Access Policy

- **Raw data is not committed to the repository** if it is proprietary or under a data licensing agreement.
- Add raw files to `.gitignore` if necessary.
- Document the acquisition procedure in a `data/raw/SOURCES.md` file.

---

## File Naming Convention

```
{ticker}_{frequency}_{start_date}_{end_date}.{extension}
```

Examples:
- `FDAX_1d_20150101_20241231.csv`
- `FESX_5min_20200101_20231231.parquet`

---

## Processing Pipeline

```
raw/ → [load_data.py] → interim/ → [clean_data.py] → processed/
```

Refer to `src/data/load_data.py` and `src/data/clean_data.py` for implementation details.
