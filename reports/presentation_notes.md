# presentation_notes.md
# Notes for oral presentation / slide deck
# Fill in after completing the final results notebook.

---

## Narrative Arc (suggested 15-minute structure)

1. **Context & Motivation (2 min)**
   - Why DAX and EURO STOXX 50?
   - Why a statistical / ML approach?
   - What question are we answering?

2. **Data & Pipeline (2 min)**
   - Source: Yahoo Finance — daily prices, 2010–2024
   - Features: price-based (momentum, vol, z-score, calendar)
   - Validation: walk-forward, strictly temporal

3. **Baseline results (3 min)**
   - Naive: hit ratio PLACEHOLDER, Sharpe PLACEHOLDER
   - Logistic: hit ratio PLACEHOLDER, Sharpe PLACEHOLDER
   - Key insight: PLACEHOLDER

4. **ML results (3 min)**
   - Best model: PLACEHOLDER
   - Sharpe net of costs: PLACEHOLDER
   - Key features: PLACEHOLDER

5. **Regime analysis (2 min)**
   - Chart: Sharpe by regime
   - Insight: PLACEHOLDER

6. **Limitations & next steps (3 min)**
   - PLACEHOLDER

---

## Key Talking Points for Interviews

- "I enforced strict temporal separation using walk-forward validation to prevent any look-ahead bias."
- "I decomposed gross vs. net alpha to understand how much of the signal is eroded by transaction costs."
- "I ran a regime-conditional analysis to understand when the model works and when it doesn't."
- "All results on the test set were generated once, after finalizing all analytical choices."
- "The project framework is fully reproducible: configs, seeds, and data caching are all documented."

---

## TODO

- [ ] Build slide deck after final results are ready
- [ ] Include equity curve chart on title slide
- [ ] Prepare a 2-minute elevator pitch version
