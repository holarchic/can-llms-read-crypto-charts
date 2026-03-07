# Hypotheses

Testable hypotheses queued for validation. Each needs sufficient signal data before running.

## H1: BTC Regime as Alt Filter

**Hypothesis**: BTC's market structure (trending up / down / ranging) predicts alt signal direction performance. Long signals on alts perform better when BTC is trending up; short signals perform better when BTC is trending down.

**Test**: Tag each resolved signal with BTC's structure classification at signal time. Slice backtest metrics by `(direction, btc_regime)` — compare win rate and expectancy across cells.

**Data needed**: ~50 resolved signals per cell (direction x regime). Estimate 2-3 weeks of live scanning.

**If validated**: Add "Market Context" step to ruleset — inject BTC regime into LLM prompt. Could filter (suppress longs in BTC downtrend) or adjust confidence.

## H2: Reinforced Signals Have Higher Win Rate

**Hypothesis**: When the LLM CONFIRMs the same coin in the same direction on consecutive scans (within 48h), the trade has a higher win rate and expectancy than single-CONFIRM trades.

**Background** (2026-03-07 analysis, v2/v3 prompts only, n=42 CONFIRMs):
- 10 coins had 2+ CONFIRMs. 9/10 fully direction-consistent (only BTC mixed).
- Within 48h: 12 reinforcement pairs vs 2 reversal pairs (86% reinforcement rate).
- Both reversals are BTC only (~24h gap). At single-scan-cycle gap (≤5h): zero reversals.
- The LLM frequently overrides the screener direction to maintain its view — screener flips while LLM stays consistent.
- Confidence scores clustered 5-7 across both reinforced and reversed signals — no discriminating power yet.
- **Important**: LLM direction lives in `machine_block.direction`, NOT `scans.direction` (which is screener direction).

**Test**: Once sufficient resolved signals exist, compare:
1. Win rate of reinforced CONFIRMs (2+ same-direction within 48h) vs single CONFIRMs
2. Expectancy (avg R-multiple) of each group
3. Whether confidence score of the reinforcing signal adds information

**Data needed**: ~30 resolved reinforced trades + ~30 resolved single trades. At current rate (~2 reinforcement pairs/week on v2/v3), need ~15 weeks of scanning — or retroactively tag existing resolved signals.

**If validated**: Gate first entry behind 2+ consecutive same-direction CONFIRMs, or increase position size on reinforced signals (e.g. 1.5x base risk%). If invalidated, the current behavior (log and skip duplicates) is correct.

**Related rejected ideas**:
- *Opposite-direction CONFIRM = close position*: Only 2 reversals in dataset (both BTC). Too rare to build around, and LLM reads charts blind — a reversal is an independent read, not an updated opinion.
- *Filter open-position coins from LLM*: Saves ~$0.10-0.50/day but loses the signal intelligence that feeds this hypothesis.
