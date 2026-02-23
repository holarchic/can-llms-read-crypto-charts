# Backtesting Philosophy

## Backtesting for validation = valid. Backtesting for parameter tuning = noise-fitting.

Quantopian studied 888 strategies. Backtest Sharpe had **near-zero predictive power** for live Sharpe (R² < 0.025). Backtest performance explains less than 2.5% of live variance.

This means backtest *optimization* is useless — but backtesting *itself* is not.

### What backtesting can answer

- **Does this idea work at all?** A setup type with negative expectancy across 200+ signals and multiple regimes is probably bad.
- **Is the LLM filter adding value?** Compare CONFIRM vs WAIT vs all-signals expectancy.
- **Which setups survive regime changes?** Cross-tab setup_type × regime to find what works everywhere, not just now.
- **Is a metric statistically significant?** Bootstrap CI — if the 95% interval includes zero, you have no evidence of edge.

### What backtesting cannot answer

- **What's the optimal ATR stop multiplier?** Sweep results on historical data fit noise. The "best" parameters are the ones that got lucky on the sample you have.
- **Which parameter set will perform best forward?** Walk-forward efficiency studies show most optimized parameters degrade out-of-sample. WFE < 0.3 = overfitting.
- **Should we disable this setup type?** Not with < 30 samples. Not without regime diversity. A type that fails in bear markets may be your best performer in trending markets.

### The parameter snooping trap

Running `sweep` on the full dataset and picking the best result is the single most common backtesting error. It guarantees overfitting.

The fix is walk-forward sweep (train on window N, test on N+1, roll forward) — but this requires enough regime-diverse history that each window contains meaningful variety. **Until we have 6+ months across different market conditions, sweep results are decoration.**

Baseline v0 parameters, chosen from first principles and the ruleset, may be as good as anything we'd tune to. Resist the urge to optimize early.

### Minimum evidence thresholds

| What | Minimum | Why |
|------|---------|-----|
| Samples per slice | 30 | Below this, statistics are meaningless |
| Bootstrap CI | Excludes zero | Otherwise no evidence of edge |
| Regime diversity | Bull + bear + ranging | Single-regime conclusions don't generalize |
| Walk-forward efficiency | > 0.5 | Below 0.3 = overfitting |

### What we use backtesting for today

1. **Signal quality validation** — expectancy by setup type, direction, structure, confidence
2. **LLM filter assessment** — does CONFIRM outperform the base rate?
3. **Regime decay tracking** — rolling window expectancy with trend arrows
4. **Auto-disable suggestions** — but only with n >= 30 and bootstrap CI excluding zero

We do NOT use backtesting for parameter optimization. The sweep tool exists for future use once we have regime-diverse data and walk-forward methodology.
