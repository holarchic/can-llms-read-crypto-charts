# Hypotheses

Testable hypotheses queued for validation. Each needs sufficient signal data before running.

## H1: BTC Regime as Alt Filter

**Hypothesis**: BTC's market structure (trending up / down / ranging) predicts alt signal direction performance. Long signals on alts perform better when BTC is trending up; short signals perform better when BTC is trending down.

**Test**: Tag each resolved signal with BTC's structure classification at signal time. Slice backtest metrics by `(direction, btc_regime)` — compare win rate and expectancy across cells.

**Data needed**: ~50 resolved signals per cell (direction x regime). Estimate 2-3 weeks of live scanning.

**If validated**: Add "Market Context" step to ruleset — inject BTC regime into LLM prompt. Could filter (suppress longs in BTC downtrend) or adjust confidence.
