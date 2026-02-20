# Scanner Roadmap

## Vision

Phase 1 → 2 → 3 → 4: from signal detection to fully autonomous trade management.

| Phase | What | Status |
|-------|------|--------|
| **1** | Signal detection + alerts | **Live** — scanning, LLM analysis, Discord alerts, backtesting, strategy system, deployed to GCE VM |
| **2** | Auto-execution | Not started — place trades on HyperLiquid with SL/TP |
| **3** | Dynamic position management | Not started — manage open positions, early invalidation |
| **4** | WAIT signal re-evaluation | **On hold** — early data shows WAITs have negative utility, REJECT steering deployed |

---

## Phase 1: Signal Detection + Alerts (LIVE)

Screener detects setups A-G on 4H candles, LLM confirms/rejects via blind chart analysis, CONFIRMs alert to Discord with entry/stop/target/leverage. Human places trades manually.

Deployed on GCE e2-small, scanning every 4H close. Daily backtest + weekly validation reports to Discord.

### Regime-Aware Monitoring

#### Current: Time-window decay (SHIPPED — Loop 3)
Rolling N-week window expectancy with trend arrows. Detects whether edge is growing/stable/decaying over time.

#### Current: Market regime classification (SHIPPED)
Every signal tagged with vol_percentile (ATR trailing percentile), efficiency_ratio (Kaufman ER), and regime label (trending_up, trending_down, volatile, ranging). Regime slicer in backtest metrics. Setup_type x regime cross-tab via `--regime-crosstab` flag on report command.

#### Next: Concave allocation lifecycle
**When**: Phase 2 auto-execution is live.
**What**: Scale position sizing by setup-type edge lifecycle — small allocation for new/unproven setups, grow as edge is confirmed (positive expectancy, CI excludes zero), shrink as edge decays (regime decay trend). Requires Phase 2 auto-execution to act on allocations.
**Research basis**: Lopez de Prado's concave strategy lifecycle. See `reference/RESEARCH_TRADING_BOT_ROBUSTNESS.md` — Stream 1.

### LLM Decision Consistency

#### Current: Single-pass analysis (SHIPPED)
One LLM call per signal, structured decision block output, ~$0.12/signal.

#### Next: Majority-vote ensemble (3-5 passes)
**When**: Once we can measure run-to-run variance on the same chart. Need ~50 CONFIRMs to test agreement rate.
**What**: Run the same chart through the LLM 3-5 times. Take majority vote on CONFIRM/REJECT/WAIT. Only alert if >=3/5 agree on CONFIRM. Optionally average the entry/stop/target across agreeing runs.
**Cost**: $0.36-0.60/signal (3-5x current). At 20 signals/scan, $7.20-12.00/scan vs current $2.40. Still within $15/day budget if scans stay at 1-2/day, tight at 6/day.
**Why**: Research shows ~40% decision determinism for large models (arXiv:2601.15322). Simple aggregation across 3-5 runs "dramatically improves consistency." Our structured output format helps, but the underlying analysis still varies.
**Research basis**: See `reference/RESEARCH_TRADING_BOT_ROBUSTNESS.md` — Stream 3 (LLM trading), non-determinism section.

### Signal Ranking Evolution

#### Tier 1: Composite scoring (SHIPPED)
Weighted composite replaces static per-setup-type confidence.
Factors: base prior (0.30), structure alignment (0.25), level strength (0.15), proximity (0.15), confluence (0.15).

#### Tier 1.5: Increase LLM cap (cost-gated)
**When**: Scanner is profitable (positive expectancy on live trades, covering its own costs).
**What**: Raise `max_llm_calls_per_scan` from 20 to 40. Currently dropping ~26 coins per scan (56%) that never get LLM eyes. At $0.16/call, doubling adds ~$10/day (~$20/day total).
**Why wait**: At $10/day the scanner needs to be paying for itself first. The LLM filter is where the edge lives — more coverage = more CONFIRMs = more trades.

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

### Prerequisite: Secret Management

Before adding trade execution credentials, migrate from `.env` to GCP Secret Manager.

**Why**: HyperLiquid API keys can move real funds — different threat class from Anthropic/Discord keys. Secrets must not sit as plaintext on disk.

**Steps**:
1. Store all secrets in GCP Secret Manager (free tier: 6 secrets, 10K accesses/month)
2. Update `config.py` to fetch from Secret Manager at runtime, fall back to `os.getenv()` for local dev
3. Scope VM service account to `secretmanager.secretAccessor` only (read, no create/delete)
4. Delete `.env` from VM disk
5. Dependency: `google-cloud-secret-manager`

**HyperLiquid-specific hardening**:
- IP-whitelist API key to VM's external IP
- Create key without withdrawal permission (trading only)
- Trade from a sub-account with capped balance (limits blast radius)

**Kill switch**: Secret Manager key or GCS object (`KILL_SWITCH`). Pipeline checks every scan cycle. Flippable from phone.

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

## Phase 4: WAIT Signal Re-Evaluation — ON HOLD

> **2026-02-18 finding**: Early data (203 scans, bear market) shows conditional WAITs have negative utility. 86% never reach entry; those that do have negative expectancy. 97/177 WAITs were soft-rejects with no trade plan at all. REJECT steering prompt now deployed — expect most former WAITs to become REJECTs. Re-evaluate this phase once REJECT steering has accumulated 2+ weeks of data across varied market conditions.

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
- Whether WAITs are worth re-evaluating at all (pending more data)

---

## Phase 5: Multi-Exchange Support

Expand beyond HyperLiquid to scan additional exchanges for setups.

**Why**: More pairs = more opportunities. Different exchanges list different tokens. Cross-exchange signals add confluence.

### Exchange Assessment

| Exchange | Candle Data | Trading API | ccxt Support | AU Access | Verdict |
|----------|-------------|-------------|--------------|-----------|---------|
| **HyperLiquid** | Working | Phase 2 | Good | Yes | **Live** |
| **Bybit** | Easy | Easy | Tier 1 | Yes | **Best next** |
| **Binance** | Easy | Easy | Tier 1 | **Futures blocked for AU since 2021** | Geo-risk |
| **MEXC** | Institutional only | Institutional only | Good (but locked) | N/A | **Dead end** |

### Bybit (recommended first)
- 350+ perpetual pairs (vs HL's ~50), V5 unified API
- Symbol format: `{coin}/USDT:USDT` (USDT not USDC)
- Rate limits more generous than HL (10 req/s trading, 50 req/s data)
- Demo trading with 50K USDT virtual funds, no KYC needed
- Live trading: Level 1 KYC (gov ID)
- No Australian restrictions

### Binance (if AU access confirmed)
- Most mature ccxt integration, largest liquidity
- Same symbol format as Bybit: `{coin}/USDT:USDT`
- **Blocker**: ASIC forced AU futures block in 2021. Existing accounts may be grandfathered — verify before building.
- Level 2 KYC required (ID + proof of address)
- Gotcha: `set_leverage()` must be called separately, not in `create_order()` params

### MEXC (not viable)
- Futures API locked to institutional users since July 2022 (KYB required)
- No retail API access for perps — not even read-only market data

### Implementation approach
Current `data.py` is mostly exchange-agnostic (retry logic, batch parallelism, DataFrame shape). Exchange-specific parts: symbol format, pair filtering, quote currency. Start with a Bybit copy of `data.py`, refactor to shared interface if adding a third exchange.

---

## Future: Prediction Market Arbitrage

Structurally identical to crypto perps thesis: fragmented, retail-dominated, API-accessible, too small/weird for institutional desks. **Window: 12-24 months** before institutions dominate (Jump Trading took stakes in Polymarket + Kalshi Feb 2026, Susquehanna already market-making on Kalshi).

### Why prediction markets

- **Proven alpha**: Arb bots extracted $40M from Polymarket in 12 months. One bot turned $313 → $414K in a month trading BTC/ETH 15-min markets. Academic research confirms persistent mispricing (Polymarket 67% accurate, Kalshi 78%).
- **Fragmented**: Polymarket, Kalshi, Limitless, Opinion, PredictIt, DraftKings — each prices the same events differently.
- **Low capital**: Start with $100-$1K on Polymarket (USDC on Polygon).
- **Fee-free or low-fee**: Polymarket is mostly fee-free with maker rebates. Kalshi 0.7-3.5%.

### Platforms & APIs

| Platform | API | Protocol | Access |
|----------|-----|----------|--------|
| **Polymarket** | REST + WebSocket + CLOB | EIP712 signed orders on Polygon | Crypto wallet. US: invite-only + KYC since Dec 2025. AU: grey area, not explicitly blocked |
| **Kalshi** | REST + FIX 4.4 | Traditional exchange | US-only, CFTC-regulated. Full KYC. $25K max/contract retail |
| **Limitless** | API available | Onchain | Crypto wallet |
| **Opinion** | API available | Onchain | Crypto wallet |

### Tooling (open-source)

- **dr-manhattan** (github.com/uncanny_guzus): ccxt-equivalent for prediction markets. Supports Polymarket, Kalshi, Limitless, Opinion. Arbitrage + market-making bots.
- **jack_cs prediction market toolkit**: Polymarket + Kalshi clients, WebSocket reconnection, 15-min rollover edge case handling. Optimized for Claude Code dev.
- **Polymarket official Python CLOB client**: github.com/Polymarket/py-clob-client
- **Polymarket AI agents framework**: github.com/Polymarket/agents

### Strategy hierarchy

1. **Cross-platform rebalancing arb** (lowest risk): Buy YES + NO across platforms when combined < $1.00. Documented 4-6 cent spreads on identical events.
2. **Temporal arbitrage** (medium risk): Polymarket prices lag confirmed spot momentum on Binance/Coinbase for crypto-linked markets. 98% win rate documented.
3. **Market making** (medium risk): Provide liquidity on thin markets with wide spreads. One user reported $700-800/day at peak.
4. **Overreaction fading** (higher risk): Negative serial correlation means price spikes tend to reverse next day. Systematic mean reversion.
5. **AI probability models** (highest edge, highest complexity): Ensemble models on news/social data. One bot generated $2.2M in two months.

### Key risks
- Regulatory whiplash (state-level bans, country blocks expanding)
- Event resolution disputes (Polymarket has had controversial resolutions)
- Smart contract / bridge risk (Polygon)
- Institutions compressing margins rapidly (Jump, Susquehanna)
- AU access is regulatory grey area for Polymarket

---

## Future: Alternative Market Landscape

Markets ranked by structural similarity to the crypto perps thesis (too small for institutions, fragmented, volatile, API-accessible).

### Tier 1 — Strong structural match

| Market | Inefficiency | API | Capital | Why it works |
|--------|-------------|-----|---------|-------------|
| **Prediction markets** | HIGH | Good | $100+ | See section above |
| **Funding rate arb (multi-venue)** | HIGH | Existing infra | $5K+ | Delta-neutral, natural extension of current scanner. Long spot + short perp to collect funding. Cross-venue: HL vs Drift vs dYdX vs GMX |
| **Esports betting** | MODERATE-HIGH | Betfair, PandaScore, Oddin.gg | $500+ | Bookmakers systematically under-model niche titles (Valorant T2-3, CS2 minor leagues). EUR 1.3M documented profit over 5 years on RebelBetting esports |
| **Horse racing quant** | MODERATE-HIGH | Betfair Exchange API | $1K+ | Parimutuel markets are measurably inefficient. Massive structured data (jockey, trainer, surface, distance). Someone reportedly made a billion dollars doing this |

### Tier 2 — Real but harder

| Market | Inefficiency | API | Capital | Notes |
|--------|-------------|-----|---------|-------|
| **CS2/Steam skin arbitrage** | MODERATE | Steam API, Tradeit.gg, Buff163 | $500+ | Price inconsistencies across 60M+ trades. Zero institutional competition. $100-300/day documented |
| **Meme coin sniping (Pump.fun)** | HIGH (but adversarial) | Solana RPC | $500+ | High variance, rug pull risk. 70-80% of individual trades lose. Survivors win big |
| **DEX market making** | MODERATE | Hummingbot, DEX APIs | $5K+ | "Shitcoin market making" — big firms pass due to insufficient scale. Individuals can make "a few million/year" per Quant Arb |
| **0DTE options** | MODERATE | IBKR TWS API | $5K+ | 51% of S&P 500 options volume. Systematic theta harvesting. Retail was first mover |
| **Domain name sniping** | MODERATE | Registrar APIs | $1K+ | 50K domains expire daily. AI analysis of traffic/backlinks/brandability is the edge. No institutional competition |

### Tier 3 — Emerging / watch list

| Market | Status | When to revisit |
|--------|--------|----------------|
| **Tokenized RWAs** (real estate, royalties) | $30B+ tokenized but thin secondary markets | When secondary market liquidity materializes |
| **Music royalties** (ANote, Royalty Exchange) | 117K transactions/year, 12% avg returns | When API trading is available (currently auction-based) |
| **Weather derivatives** (Kalshi parametric contracts) | Thin liquidity, NOAA data is free, clean settlement | When volume grows enough to deploy capital |
| **Carbon credits** (tokenized: Thallo, TakeCarbon) | $2.5B voluntary market, projected $100B+ by 2030 | When electronic orderbooks exist for retail |
| **Electricity spot arb** (battery arbitrage) | 66% of US utility batteries used for arb in 2024 | Requires physical infrastructure or virtual power trading access |
| **Freight futures** (Baltic Exchange FFAs) | Growing algo participation, still niche | When retail API access improves |

### The meta-filter

Markets where retail algo traders find alpha share these traits:

| Filter | Why |
|--------|-----|
| Too small for institutional capital | Funds need $10M+ capacity; sub-$1M/day markets are ignored |
| Fragmented venues | Multiple platforms with inconsistent pricing |
| 24/7 or non-standard hours | Institutions staffed 9-5; overnight/weekend = less competition |
| Regulatory grey zone | Compliance costs keep institutions out |
| Domain expertise moat | Niche knowledge institutions lack (esports, weather, horse racing) |
| API-accessible but unsophisticated participants | Markets where most participants trade manually/emotionally |
| Young/growing market | Inefficiencies haven't been competed away yet |
