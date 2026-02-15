# Scanner Roadmap

## Vision

Phase 1 → 2 → 3 → 4: from signal detection to fully autonomous trade management.

| Phase | What | Status |
|-------|------|--------|
| **1** | Signal detection + alerts | **Live** — scanning, LLM analysis, Discord alerts, backtesting, strategy system, deployed to GCE VM |
| **2** | Auto-execution | Not started — place trades on HyperLiquid with SL/TP |
| **3** | Dynamic position management | Not started — manage open positions, early invalidation |
| **4** | WAIT signal re-evaluation | Not started — revisit WAITs on shorter timeframe |

---

## Phase 1: Signal Detection + Alerts (LIVE)

Screener detects setups A-G on 4H candles, LLM confirms/rejects via blind chart analysis, CONFIRMs alert to Discord with entry/stop/target/leverage. Human places trades manually.

Deployed on GCE e2-small, scanning every 4H close. Daily backtest + weekly validation reports to Discord.

### Signal Ranking Evolution

#### Tier 1: Composite scoring (SHIPPED)
Weighted composite replaces static per-setup-type confidence.
Factors: base prior (0.30), structure alignment (0.25), level strength (0.15), proximity (0.15), confluence (0.15).

#### Tier 2: Diminishing returns penalty
**When**: If clustering appears in production (e.g. 15/20 signals are the same setup type).
**What**: Greedy selection with decay — `effective_score = composite * 0.85^n` where `n` = count of same setup type already selected. Replaces the simple sort+slice in `pipeline.py:stage1_screen`.
**Scope**: ~20 lines in `pipeline.py`. No model/screener changes.

#### Tier 3: Feedback loop from historical results
**When**: 50+ LLM results per setup type accumulated in `scanner.db`.
**What**: Query historical CONFIRM rate by setup type from storage. Blend empirical hit rate as a 6th scoring factor in `score_signal()`. The LLM's own judgment feeds back into which signals get prioritized — the system learns which setups are worth spending tokens on.

**Why wait**: Premature optimization. Need real data to validate. Bad early data would create a self-reinforcing bias (low-sample setups get deprioritized → never get sampled → stay deprioritized).

---

## Phase 2: Auto-Execution

Remove the human from trade entry. Bot places HyperLiquid perp trades automatically when LLM confirms a signal.

**What**:
- On CONFIRM, place limit order at LLM's entry zone with SL and TP orders
- Position sizing from account balance + risk percentage (reuse `calculate_position_size()` logic)
- Leverage set per LLM recommendation, capped by exchange max
- Order state tracking in DB (placed, filled, cancelled, expired)
- Kill switch: global flag to halt all new entries
- Discord alerts shift from "signal for you to trade" to "trade placed on your behalf"

**Key decisions** (to resolve when starting):
- Limit vs market entry (limit at entry zone vs market on CONFIRM?)
- Max concurrent positions / max exposure
- Per-trade and total portfolio risk limits
- HyperLiquid API auth (API key + secret, separate from Anthropic key)

**Dependencies**: Phase 1 running with positive expectancy. Need enough backtest data to trust the system before letting it trade real capital.

---

## Phase 3: Dynamic Position Management

Actively manage open positions — don't just set-and-forget SL/TP.

**What**:
- Monitor open positions against incoming price data
- Early invalidation: close positions when the setup thesis breaks (e.g. structure shift, key level lost) before SL is hit
- Trailing stops: move SL to breakeven after partial TP, trail behind structure
- Partial TP: scale out at multiple targets (aligned with ruleset's multi-target TP structure)
- Re-evaluate on each 4H candle: generate fresh chart, ask LLM "is this position still valid?"
- Log management decisions to DB for review

**Key decisions**:
- LLM-driven vs rule-based management (or hybrid)
- Re-evaluation frequency (every 4H candle? every 1H?)
- Cost control — each re-evaluation is an LLM call

---

## Phase 4: WAIT Signal Re-Evaluation

WAIT signals are "not yet, but watch this" — revisit them sooner than the next 4H candle.

**What**:
- Track WAIT signals separately with their conditions (what needs to happen for entry)
- Re-check on shorter timeframe (1H candles) for developing setups
- Generate updated charts and re-evaluate: "has this WAIT become a CONFIRM?"
- If promoted to CONFIRM, route through Phase 2 execution
- Expire WAITs after N candles if conditions don't materialize

**Key decisions**:
- Re-check frequency (every 1H? every 2H?)
- Cost budget for WAIT re-evaluation (could be significant if many WAITs accumulate)
- Max WAIT lifetime before auto-expire
