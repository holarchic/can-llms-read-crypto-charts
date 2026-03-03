"""CLI entry point for the scanner. Run with: python -m scanner.cli"""

import argparse
import logging
import sys
import traceback

from .alerts import COLOR_ERROR, send_ops_log
from .config import ScannerConfig
from .pipeline import run_scan
from .storage import cleanup_old_candles, get_connection, signal_history, signal_stats, stats


def main():
    parser = argparse.ArgumentParser(
        description="Crypto Perpetuals Scanner (HyperLiquid, Bybit, Binance)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scanner.cli                                  # Scan HL, collect candles from all 3
  python -m scanner.cli --exchange bybit                 # Scan Bybit, collect candles from all 3
  python -m scanner.cli --candle-exchanges hyperliquid   # Only collect HL candles (skip Bybit/Binance)
  python -m scanner.cli --dry-run                        # Stage 1 only, print signals
  python -m scanner.cli --coin BTC                       # Single coin
  python -m scanner.cli --no-discord                     # Skip Discord alerts
  python -m scanner.cli --stats                          # Scan history summary
        """,
    )
    parser.add_argument("--exchange", choices=["hyperliquid", "bybit", "binance"],
                        default="hyperliquid", help="Exchange to scan (default: hyperliquid)")
    parser.add_argument("--candle-exchanges", nargs="+",
                        help="Exchanges to collect candles from (default: all three)")
    parser.add_argument("--dry-run", action="store_true", help="Run screener only, skip LLM + alerts")
    parser.add_argument("--coin", help="Scan a single coin (e.g. BTC, ETH, SOL)")
    parser.add_argument("--no-discord", action="store_true", help="Skip Discord alerts")
    parser.add_argument("--stats", action="store_true", help="Show scan history stats")
    parser.add_argument("--signals", action="store_true", help="View screener signal history")
    parser.add_argument("--signal-stats", action="store_true", help="Screener performance summary")
    parser.add_argument("--days", type=int, default=7, help="Days of history (default: 7)")
    parser.add_argument("--type", dest="setup_type", help="Filter by setup type (A-G)")
    parser.add_argument("--sent-only", action="store_true", help="Only show signals sent to LLM")
    parser.add_argument("--cleanup-candles", action="store_true", help="Delete old candles per retention policy (15m: 180 days)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    # Execution flags
    parser.add_argument("--execute", action="store_true", help="Enable trade execution for this run")
    parser.add_argument("--no-execute", action="store_true", help="Disable trade execution for this run")
    parser.add_argument("--execution-mode", choices=["testnet", "live"], help="Override execution mode")
    parser.add_argument("--dry-execute", action="store_true",
                        help="Run execution pipeline (validate, sanity, sizing) without placing orders")

    args = parser.parse_args()

    # Execution logging always visible (captured by cron 2>&1).
    # Handler at INFO, root logger at WARNING (blocks noise from other modules),
    # scanner.execution at INFO (so skip reasons + outcomes are visible).
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    logging.getLogger("scanner.execution").setLevel(logging.INFO)

    if args.cleanup_candles:
        deleted = cleanup_old_candles()
        if not args.quiet:
            print(f"Candle cleanup: {deleted} rows deleted")
        return

    if args.signal_stats:
        conn = get_connection()
        s = signal_stats(args.days, conn)
        conn.close()

        print(f"\nSignal Stats (last {s['days']} days)")
        print(f"{'=' * 50}")
        print(f"  Total signals:  {s['total_signals']}")
        print(f"  Sent to LLM:    {s['sent_to_llm']}")
        print(f"  Filtered:       {s['filtered']}")
        print(f"  Scan batches:   {s['batches']}")

        if s["by_type"]:
            print("\nBy setup type:")
            for t in s["by_type"]:
                print(f"  {t['setup_type']}  {t['setup_name']:<28}  "
                      f"total={t['cnt']:>3}  sent={t['sent']:>3}  "
                      f"avg_conf={t['avg_conf']:.2f}")

        if s["top_coins"]:
            print("\nTop coins:")
            for c in s["top_coins"]:
                print(f"  {c['coin']:>8}  total={c['cnt']:>3}  sent={c['sent']:>3}  "
                      f"avg_conf={c['avg_conf']:.2f}")
        return

    if args.signals:
        conn = get_connection()
        rows = signal_history(
            coin=args.coin,
            setup_type=args.setup_type,
            days=args.days,
            sent_only=args.sent_only,
            conn=conn,
        )
        conn.close()

        if not rows:
            print("No signals found.")
            return

        print(f"\nSignal History ({len(rows)} signals, last {args.days} days)")
        print(f"{'=' * 80}")
        for r in rows:
            sent_flag = "*" if r["sent_to_llm"] else " "
            print(f" {sent_flag} {r['coin']:>8}  {r['setup_type']}  {r['direction']:>5}  "
                  f"conf={r['confidence']:.2f}  lvl=${r['key_level']:,.2f}  "
                  f"#{r['rank_in_batch']}/{r['total_in_batch']}  "
                  f"{r['timestamp'][:16]}")
        print("\n  * = sent to LLM")
        return

    if args.stats:
        conn = get_connection()
        s = stats(conn)
        conn.close()

        print("\nScanner Stats")
        print(f"{'=' * 40}")
        print(f"  Total scans:    {s['total_scans']}")
        print(f"  CONFIRMs:       {s['confirms']}")
        print(f"  REJECTs:        {s['rejects']}")
        print(f"  WAITs:          {s['waits']}")
        print(f"  Alerted:        {s['alerted']}")
        print(f"  Total cost:     ${s['total_cost_usd']:.4f}")
        print(f"  Today's cost:   ${s['today_cost_usd']:.4f}")

        if s["recent"]:
            print("\nRecent scans:")
            for r in s["recent"]:
                print(f"  {r['coin']:>6}  {r['setup_type']}  {r['direction']:>5}  "
                      f"{r['decision'] or 'N/A':>7}  ${r['cost_usd']:.4f}  {r['timestamp'][:16]}")
        return

    candle_exchanges = args.candle_exchanges if args.candle_exchanges is not None else ["hyperliquid", "bybit", "binance"]
    config = ScannerConfig(exchange=args.exchange, candle_exchanges=candle_exchanges)

    # Build execution config with CLI overrides
    exec_config = None
    if args.execute or args.dry_execute:
        from .execution.config import load_execution_config
        exec_config = load_execution_config()
        if args.execute:
            exec_config.enabled = True
        if args.no_execute:
            exec_config.enabled = False
        if args.execution_mode:
            exec_config.mode = args.execution_mode

    try:
        run_scan(
            config=config,
            coin=args.coin,
            dry_run=args.dry_run,
            skip_discord=args.no_discord,
            verbose=not args.quiet,
            exec_config=exec_config,
            dry_execute=args.dry_execute,
        )
    except Exception as e:
        traceback.print_exc()
        send_ops_log(
            "Scan FAILED",
            {"Error": str(e)[:200], "Type": type(e).__name__},
            COLOR_ERROR,
            config,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
