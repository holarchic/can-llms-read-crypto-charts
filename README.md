# llm-trader

A scaffold for an **LLM-driven crypto perpetual-futures scanner** — clone it, plug in your own strategy, and use the built-in harness to test whether that strategy actually has edge. Plus an honest account of the strategy *we* tried, which doesn't.

## What this is

A working skeleton for an LLM-in-the-loop trading scanner:

```
scheduled scan → multi-timeframe market data → LLM chart analysis →
signal generation → execution / shadow-capture → historical-replay backtest
```

The **engine** (data plane, LLM pipeline, execution, shadow-capture, point-in-time replay harness) is generic. The **strategy** — what counts as a setup, what the LLM is asked, how signals are gated — is a swappable layer. Our strategy is included as the worked example; the point is to replace it with yours.

## Replace the strategy with your own

The strategy surface is version-gated (`SETUP_VERSION` in `scanner/config.py`), so you can add your own alongside the example:

- **Setup detection** — `scanner/setups_v2.py`, `scanner/sr_levels.py`, `scanner/trend_state.py`, `scanner/regime.py`
- **What the LLM sees / is asked** — `scanner/llm.py` (system prompt, scanner-mode prompt)
- **Signal gating** — `scanner/screener_v2.py`, `scanner/setup_validator.py`

Swap those for your own logic; keep the engine.

## Test it for edge before you trust it

The part that matters. `reference/momentum/` is an **edge-existence test harness** built to *not* fool itself:

- **Matched same-day controls** — measure the signal, not market beta.
- **Confound corrections** — naive tests here faked edges via a *distance* confound (signals near spot look predictive because price mean-reverts at short range) and a *cluster* confound (N hand-labelled charts are ~N *correlated* observations, not N independent levels).
- **Point-in-time discipline** — decide on data available at decision time, enter at the next open, lag external data by its real publication latency.
- **Block-bootstrap inference** — circular moving-block bootstrap clustered by time, not IID resampling.
- **Anti-vacuity fixtures** — feed the harness a *planted* signal (must recover it), a *null* (must return zero), and a *clone* of a known factor (must strip to zero after controlling for it). A harness that can't pass these isn't measuring what it claims.

## What we found (why we open-sourced it)

We ran our own strategy through this harness. Every entry edge we tested came back empty:

| Hypothesis | Verdict |
|---|---|
| An LLM reading charts finds setups | **Falsified** — deep historical replay: −0.25R net, negative in *every* half-year; the LLM's directional calls were anti-predictive and low-conviction. |
| Horizontal support/resistance levels have edge | **Dead** — hold at chance once distance- and cluster-controlled (mechanical constructions *and* expert hand-drawn levels). |
| Discrete momentum (breakout / pullback) | **Dead** standalone — momentum is a real *portfolio* factor (long-short Sharpe ≈ 0.86), but per-trade variance is too large to harvest one bet at a time. |
| Social text leads price | **No** — social-attention volume *lags* price (marginal IC over momentum ≈ 0; lead-vs-confirm differential robustly negative). |

A rigorously-controlled null — confounds named, harness self-validated — is harder to fake than a curve-fit equity curve, and more honest about how markets behave. That's the contribution: the method, and the dead ends it found.

## Layout

```
scanner/             the scanner engine + the example strategy (swap this)
reference/momentum/  the edge-existence test harness
```

## Running it

Python 3.13 and [`uv`](https://github.com/astral-sh/uv). The live scanner needs an Anthropic API key and exchange access; the offline research harness needs a local Binance 15m candle database at `data/binance-backtest.db`. See `.env.example`. The `reference/momentum/` scripts are exploratory (compact, run-and-read), not a packaged library.

## Status

The example strategy was tested and **found to have no edge**; this bot is not run live. The engine and the test harness are the reusable parts.

## License

[MIT](LICENSE).
