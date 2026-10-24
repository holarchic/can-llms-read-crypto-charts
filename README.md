# can-llms-read-crypto-charts

Can LLMs read crypto charts and trade well enough to have an edge? We built a full AI scanner and Hyperliquid execution testing harness to find out.

> **Status:** complete, June 2026. Not running live. The edge-test harness and the scanner engine are reusable; the example strategy is not.

## What we found

| Hypothesis | Verdict |
|---|---|
| An LLM reading charts finds setups | **Falsified**. Deep historical replay: −0.25R net, negative in *every* half-year; the LLM's directional calls were anti-predictive and low-conviction. |
| Horizontal support/resistance levels have edge | **Dead**. Hold at chance once distance- and cluster-controlled (mechanical constructions *and* expert hand-drawn levels). |
| Discrete momentum (breakout / pullback) | **Dead** standalone. Momentum is a real *portfolio* factor (long-short Sharpe ≈ 0.86), but per-trade variance is too large to harvest one bet at a time. |
| Social text leads price | **No**. Social-attention volume *lags* price (marginal IC over momentum ≈ 0; lead-vs-confirm differential robustly negative). |

A controlled null (confounds named, harness self-validated) is harder to fake than a curve-fit equity curve. The contribution is the method and the dead ends it found.

## How we tested it

The scanner ran the LLM test end to end:

```
scheduled scan → multi-timeframe market data → LLM chart analysis →
signal generation → execution / shadow-capture → historical-replay backtest
```

[`reference/momentum/`](reference/momentum/) is the edge-existence harness, built so it can't fool itself:

- **Matched same-day controls.** Signals are measured against control trades from the same day, so market beta drops out (crypto rose through 2023-26, which makes naive long R positive for free).
- **Confound corrections.** Naive tests here faked edges through a *distance* confound (signals near spot look predictive because price mean-reverts at short range) and a *cluster* confound (N hand-labelled charts are about N *correlated* observations, not N independent levels).
- **Point-in-time discipline.** Decide on data available at decision time, enter at the next open, and lag external data by its real publication latency (1 day for social volume).
- **Block-bootstrap inference.** Circular moving-block bootstrap clustered by time, never IID resampling.
- **Anti-vacuity fixtures.** The harness must recover a *planted* signal, return zero on a *null*, and strip a *clone* of a known factor to zero after controlling for it.

## Reproduce

Python 3.13 and [`uv`](https://github.com/astral-sh/uv). [`text_lead.py`](reference/momentum/text_lead.py) runs the fixtures on synthetic panels in about a second, with no data or keys:

```bash
uv run python reference/momentum/text_lead.py
```

```text
ALL FIXTURES PASS — engine validated
```

```bash
uv run --extra dev pytest    # 4,851 passed, 42 skipped (macOS, Python 3.13.16)
```

- The momentum scripts read a local Binance 15m candle database at `data/binance-backtest.db`, relative to the repo root (SQLite, table `candles` with `coin`, `interval`, `timestamp`, `open`, `high`, `low`, `close`, `volume`). This repo doesn't build it.
- `text_panel.py` also needs `sanpy` (not installed by `pyproject.toml`) and a `SANTIMENT_API_KEY`.
- The live scanner needs an Anthropic API key and exchange access (HyperLiquid by default). Variables are listed in [`.env.example`](.env.example).

## Limits

- The LLM deep-replay result came from CLIs this repo doesn't ship (`scripts/shadow-replay.py`, `scripts/shadow-resolve.py`). The replay library they drive is in `scanner/shadow/`.
- The support/resistance experiments aren't included either. The momentum and text-lead tests rerun from `reference/momentum/` given the candle database.
- The text-lead test covers about 40 weekly rebalances (June 2025 to March 2026, Santiment free tier), all in one regime.
- The `reference/momentum/` scripts are exploratory (compact, run-and-read), not a packaged library.

## Reuse it

### Test your own signal

1. Build it from the point-in-time panel in [`panel.py`](reference/momentum/panel.py).
2. Reuse an existing gate: the trade-level gate in [`discrete_trade.py`](reference/momentum/discrete_trade.py) (as [`ribbon_trade.py`](reference/momentum/ribbon_trade.py) does), or the factor gate in `text_lead.py` (as [`text_panel.py`](reference/momentum/text_panel.py) does with real data).
3. Run the fixtures before you trust a result.

### Plug in your own strategy

The **engine** (data plane, LLM pipeline, execution, shadow-capture, point-in-time replay harness) is generic. The **strategy** (what counts as a setup, what the LLM is asked, how signals are gated) is a swappable layer, version-gated by `SETUP_VERSION` in [`scanner/config.py`](scanner/config.py) so yours can run alongside the example:

- **Setup detection:** `scanner/setups_v2.py`, `scanner/sr_levels.py`, `scanner/trend_state.py`, `scanner/regime.py`
- **What the LLM sees and is asked:** `scanner/llm.py` (system prompt, scanner-mode prompt)
- **Signal gating:** `scanner/screener_v2.py`, `scanner/setup_validator.py`

The example strategy's ruleset file is not included. The LLM step exits with an error until you add your own at `strategy/unified-ta-ruleset.md` (`SETUP_VERSION=v1`, the default) or `strategy/unified-ta-ruleset-v2.md` (`v2`).

```
scanner/             the scanner engine + the example strategy (swap this)
reference/momentum/  the edge-existence test harness
```

## Disclaimer

Nothing here is financial advice. You use this code at your own risk.

> [!CAUTION]
> The execution module can place real orders. It is off and pointed at testnet by default (`EXECUTION_ENABLED=false`, `EXECUTION_MODE=testnet`). Leave it that way unless you mean to trade.

## Contributing

Issues and replications are welcome. If you find an edge the harness missed, open an issue with the code and data to rerun it.

## Attribution

For those who care about such things, the majority of the analysis and code in this repo was produced by Claude Opus 4.6 (Anthropic), with david guiding Claude, providing insights, direction and critical review throughout the process.

## License

[MIT](LICENSE).
