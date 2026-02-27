"""Backtest CLI entry point: python -m scanner.backtest"""

import argparse
import sys
from datetime import datetime, timezone, timedelta

from ..config import get_exchange_meta
from ..storage import get_connection
from .candles import fetch_and_cache
from .resolver import (
    TradeSetup,
    TradeOutcome,
    resolve_signal,
    build_screener_setup,
    build_llm_setup,
    estimate_slippage,
)
from .metrics import compute_metrics, slice_metrics
from .report import format_full_report, build_backtest_embeds


def _load_screener_signals(conn, days, coin, setup_type, strategy_id=None):
    """Load signals from the signals table."""
    clauses = ["datetime(timestamp) > datetime('now', ? || ' days')"]
    params = [f"-{int(days)}"]

    if coin:
        clauses.append("coin = ?")
        params.append(coin.upper())
    if setup_type:
        clauses.append("setup_type = ?")
        params.append(setup_type.upper())
    if strategy_id:
        clauses.append("strategy_id = ?")
        params.append(strategy_id)

    where = " AND ".join(clauses)
    rows = conn.execute(
        f"SELECT * FROM signals WHERE {where} ORDER BY timestamp",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def _load_llm_signals(conn, days, coin, setup_type, strategy_id=None, prompt_version=None):
    """Load CONFIRM scans from the scans table."""
    clauses = [
        "datetime(timestamp) > datetime('now', ? || ' days')",
        "decision = 'CONFIRM'",
    ]
    params = [f"-{int(days)}"]

    if coin:
        clauses.append("coin = ?")
        params.append(coin.upper())
    if setup_type:
        clauses.append("setup_type = ?")
        params.append(setup_type.upper())
    if strategy_id:
        clauses.append("strategy_id = ?")
        params.append(strategy_id)
    if prompt_version:
        clauses.append("prompt_version = ?")
        params.append(prompt_version)

    where = " AND ".join(clauses)
    rows = conn.execute(
        f"SELECT * FROM scans WHERE {where} ORDER BY timestamp",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def main():
    parser = argparse.ArgumentParser(
        description="Backtest scanner signals against forward price data.",
    )
    parser.add_argument(
        "--mode", choices=["screener", "llm"], default="screener",
        help="Resolution mode: screener (signals table) or llm (scans table, CONFIRM only)",
    )
    parser.add_argument("--exchange", choices=["hyperliquid", "bybit", "binance"],
                        default="hyperliquid", help="Exchange for candle data (default: hyperliquid)")
    parser.add_argument("--days", type=int, default=30, help="Lookback days for signals (default: 30)")
    parser.add_argument("--coin", type=str, default=None, help="Filter by coin (e.g. BTC)")
    parser.add_argument("--type", type=str, default=None, help="Filter by setup type (e.g. C)")
    parser.add_argument("--timeout", type=int, default=72, help="Timeout bars before EXPIRED (default: 72 = 3d)")
    parser.add_argument("--atr-stop", type=float, default=1.5, help="ATR multiplier for stop (screener mode)")
    parser.add_argument("--atr-target", type=float, default=3.0, help="ATR multiplier for target (screener mode)")
    parser.add_argument("--fee", type=float, default=None, help="Taker fee per side (default: exchange-specific)")
    parser.add_argument("--funding-rate", type=float, default=0.0001, help="Funding rate per 8h (default: 0.01%%)")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage rate per side (default: 0.05%%)")
    parser.add_argument("--no-slippage", action="store_true", help="Disable slippage estimation")
    parser.add_argument("--strategy", type=str, default=None, help="Filter signals by strategy version")
    parser.add_argument("--prompt-version", type=str, default=None, help="Filter LLM scans by prompt version (e.g. 2)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print per-signal details")
    parser.add_argument("--discord", action="store_true", help="Post results to Discord reports channel")
    parser.add_argument("--permutation", action="store_true", help="Run Monte Carlo permutation test on expectancy")
    args = parser.parse_args()

    # Resolve fee rate: explicit --fee > exchange default
    meta = get_exchange_meta(args.exchange)
    fee_rate = args.fee if args.fee is not None else meta["fee_rate"]

    conn = get_connection()

    # Load signals
    if args.mode == "screener":
        rows = _load_screener_signals(conn, args.days, args.coin, args.type, args.strategy)
        builder = lambda row: build_screener_setup(row, args.atr_stop, args.atr_target)
    else:
        rows = _load_llm_signals(conn, args.days, args.coin, args.type, args.strategy,
                                  prompt_version=args.prompt_version)
        builder = build_llm_setup

    if not rows:
        print(f"No signals found for --mode {args.mode} --days {args.days}")
        sys.exit(0)

    # Build setups
    setups = []
    skipped = 0
    for row in rows:
        setup = builder(row)
        if setup:
            setups.append(setup)
        else:
            skipped += 1

    if not setups:
        print(f"No valid setups built ({skipped} skipped due to missing/unparseable data)")
        sys.exit(0)

    print(f"Loaded {len(setups)} setups ({skipped} skipped) from {len(rows)} {args.mode} signals")

    # Resolve each setup
    outcomes: list[TradeOutcome] = []
    coins_seen = set()
    for i, setup in enumerate(setups):
        coins_seen.add(setup.coin)
        # Fetch candles: from signal time to signal + timeout
        signal_dt = datetime.fromisoformat(setup.timestamp)
        until_dt = signal_dt + timedelta(hours=args.timeout + 1)

        try:
            candles = fetch_and_cache(
                setup.coin, "1h",
                signal_dt.isoformat(),
                until_dt.isoformat(),
                conn,
                exchange=args.exchange,
            )
        except Exception as e:
            if args.verbose:
                print(f"  [{i+1}/{len(setups)}] {setup.coin} {setup.setup_type} — fetch failed: {e}")
            continue

        slippage = 0.0 if args.no_slippage else args.slippage
        outcome = resolve_signal(
            setup, candles, args.timeout, fee_rate,
            funding_rate_8h=args.funding_rate,
            slippage_rate=slippage,
        )
        outcomes.append(outcome)

        if args.verbose:
            print(
                f"  [{i+1}/{len(setups)}] {setup.coin:<6} {setup.setup_type} "
                f"{setup.direction:<5} → {outcome.result:<7} "
                f"{outcome.r_achieved:+.2f}R ({outcome.bars_to_exit} bars)"
            )

    conn.close()

    # Separate unresolved (insufficient forward data) from resolved
    unresolved = [o for o in outcomes if o.result == "UNRESOLVED"]
    resolved = [o for o in outcomes if o.result != "UNRESOLVED"]

    if not resolved:
        print(f"No outcomes resolved ({len(unresolved)} unresolved — insufficient forward data)")
        sys.exit(0)

    # Compute metrics
    overall = compute_metrics(resolved, "Overall", run_permutation=args.permutation)
    sliced = slice_metrics(resolved)

    report_text = format_full_report(overall, sliced, args.mode)
    unresolved_note = f" | Unresolved: {len(unresolved)}" if unresolved else ""
    footer = f"\n  Coins: {len(coins_seen)} | Signals: {len(setups)} | Resolved: {len(resolved)}{unresolved_note}"

    print()
    print(report_text)
    print(footer)

    if args.discord:
        from ..alerts import send_report_embeds
        embeds = build_backtest_embeds(overall, sliced, args.mode, args.days)
        ok = send_report_embeds(embeds)
        if ok:
            print("\n  Sent to Discord.")
        else:
            print("\n  Discord send failed (no webhook configured?).")


if __name__ == "__main__":
    main()
