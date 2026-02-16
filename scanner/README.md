# Scanner

Automated HyperLiquid perpetuals market scanner. Screens top pairs by volume for 7 setup types (A-G), generates multi-TF charts, runs Claude deep analysis, and posts CONFIRM signals to Discord.

## Usage

```bash
# Full scan (all pairs above volume threshold)
python -m scanner.cli

# Dry run — screener only, no LLM calls
python -m scanner.cli --dry-run

# Single coin
python -m scanner.cli --coin BTC

# Skip Discord alerts
python -m scanner.cli --no-discord

# Scan history stats
python -m scanner.cli --stats
```

## Pipeline

| Stage | What | Cost |
|-------|------|------|
| 1. Screen | Fetch pairs → 4H candles → detect setups A-G → dedup | Free |
| 2. Analyze | Multi-TF charts → Claude vision analysis → save to DB | ~$0.10/signal |
| 3. Alert | Filter CONFIRMs → Discord webhook → cleanup charts | Free |

## Setup Types

| Type | Name | Trigger |
|------|------|---------|
| A | Same-Way Level | Ranging + price at S/R boundary |
| B | Breakout/Breakdown | Close crosses respected S/R |
| C | S/R Flip Continuation | MSB + pullback to broken level |
| D | Deeper Retrace | MSB + price in OTE zone (0.618-0.786) |
| E | Reversal | 2+ opposing MSBs |
| F | Deviation | Wick beyond S/R + close back inside |
| G | Spike/SFP | Wick beyond swing H/L + close inside + large wick |

## Config

Key parameters in `scanner/config.py`:

| Param | Default | Purpose |
|-------|---------|---------|
| `min_volume_usd` | 5M | Minimum 24h volume to scan |
| `max_pairs` | 80 | Cap on pairs to screen |
| `max_llm_calls_per_scan` | 20 | Cost control |
| `daily_cost_cap` | $15 | Hard stop on daily LLM spend |
| `dedup_hours` | 12 | Don't re-alert same setup within window |


## Tests

```bash
pytest scanner/tests/ -v
```
