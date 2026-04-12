"""CLI for execution module. Run with: python -m scanner.execution"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from .config import load_execution_config

_ENV_PATH = Path(__file__).parent.parent.parent / ".env"


def main():
    # Load .env before execution config — otherwise credentials are empty
    # when running standalone (python -m scanner.execution) vs via scanner.cli
    load_dotenv(_ENV_PATH)

    parser = argparse.ArgumentParser(
        description="Execution module CLI — monitor, positions, orders, kill switch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scanner.execution monitor         # Sync order status + health check
  python -m scanner.execution positions       # List open positions
  python -m scanner.execution orders          # Recent orders (7 days)
  python -m scanner.execution orders --days 3 # Recent orders (3 days)
  python -m scanner.execution kill            # Activate kill switch
  python -m scanner.execution unkill          # Deactivate kill switch
        """,
    )
    parser.add_argument("command", choices=["monitor", "positions", "orders", "kill", "unkill", "report", "open-positions", "smoke"],
                        help="Subcommand to run")
    parser.add_argument("--days", type=int, default=7, help="Days of history (default: 7)")
    parser.add_argument("--mode", choices=["testnet", "live"], help="Filter by execution mode")
    parser.add_argument("--discord", action="store_true", help="Send report to Discord")

    args = parser.parse_args()
    config = load_execution_config()

    if args.command == "monitor":
        from .discord_logging import configure_monitor_logging

        discord_handler = configure_monitor_logging(config.discord_errors_webhook_url)
        _cmd_monitor(config, discord_handler)
    elif args.command == "positions":
        _cmd_positions(config)
    elif args.command == "orders":
        _cmd_orders(config, args.days)
    elif args.command == "kill":
        _cmd_kill(config, active=True)
    elif args.command == "unkill":
        _cmd_kill(config, active=False)
    elif args.command == "report":
        _cmd_report(config, args.days, args.mode, args.discord)
    elif args.command == "open-positions":
        _cmd_open_positions(config, args.discord)
    elif args.command == "smoke":
        _cmd_smoke(config)


def _cmd_monitor(config, discord_handler=None):
    import traceback

    from ccxt.base.errors import ExchangeNotAvailable, NetworkError, RequestTimeout

    from scanner.storage import get_connection

    from .executor import _send_errors_alert
    from .monitor import check_positions, enforce_position_timeouts, sync_order_status

    _TRANSIENT = (ExchangeNotAvailable, NetworkError, RequestTimeout, ConnectionError, TimeoutError)

    try:
        conn = get_connection()

        # Fetch exchange open orders once — shared by sync and health check
        # to reduce API calls (HL counts per-user info requests against rate limit)
        exchange_open_orders = None
        try:
            from .exchange import get_exchange_open_orders
            exchange_open_orders = get_exchange_open_orders(config)
        except Exception as exc:
            print(f"Exchange open orders fetch failed: {exc}")

        # Each step isolated so a transient API blip doesn't skip the rest
        try:
            updated = sync_order_status(config, conn,
                                        exchange_open_orders=exchange_open_orders)
            print(f"Orders updated: {updated}")
            # Re-fetch after sync if orders changed (e.g. SL replaced after
            # TP1 fill) so check_positions sees the current exchange state
            if updated > 0:
                try:
                    exchange_open_orders = get_exchange_open_orders(config)
                except Exception as exc:
                    print(f"Exchange re-fetch after sync failed: {exc}")
        except _TRANSIENT as exc:
            print(f"sync_order_status skipped (transient): {exc}")

        try:
            if config.drift_enabled:
                from .drift import check_drift_positions
                drift_acted = check_drift_positions(config, conn)
                if drift_acted:
                    print(f"Drift invalidation: {drift_acted} position(s) acted on")
                    try:
                        exchange_open_orders = get_exchange_open_orders(config)
                    except Exception as exc:
                        print(f"Exchange re-fetch after drift failed: {exc}")
        except _TRANSIENT as exc:
            print(f"check_drift_positions skipped (transient): {exc}")

        try:
            timed_out = enforce_position_timeouts(config, conn)
            if timed_out:
                print(f"Positions timed out: {timed_out}")
        except _TRANSIENT as exc:
            print(f"enforce_position_timeouts skipped (transient): {exc}")

        try:
            issues = check_positions(config, conn,
                                     exchange_open_orders=exchange_open_orders)
        except _TRANSIENT as exc:
            print(f"check_positions skipped (transient): {exc}")
            issues = []

        if issues:
            print(f"\nHealth check issues ({len(issues)}):")
            for issue in issues:
                print(f"  [{issue['type']}] {issue['message']}")

            # Attempt SL recovery, then alert if still unprotected
            critical = [i for i in issues if i["type"] in ("missing_sl", "phantom_sl")]
            if critical:
                from .executor import _send_urgent_alert
                from .monitor import attempt_sl_recovery

                for issue in critical:
                    if issue["type"] == "missing_sl":
                        recovered = attempt_sl_recovery(issue, config, conn)
                        if recovered:
                            print(f"  SL recovery SUCCESS: {issue['coin']} {issue['direction']}")
                            continue
                    _send_urgent_alert(issue["message"], config)
        else:
            print("Health check: OK")

        conn.close()
    except Exception as exc:
        traceback.print_exc()
        _send_errors_alert(f"Monitor CRASHED: {type(exc).__name__}: {exc}", config)
    finally:
        if discord_handler:
            discord_handler.flush_to_discord()


def _cmd_positions(config):
    from .exchange import get_open_positions

    positions = get_open_positions(config)
    if not positions:
        print("No open positions.")
        return

    print(f"\nOpen Positions ({len(positions)}):")
    print(f"{'Coin':>8} {'Dir':>5} {'Size':>10} {'Entry':>10} {'uPnL':>10} {'Lev':>4}")
    print("-" * 55)
    for p in positions:
        print(f"{p['coin']:>8} {p['direction']:>5} {p['size']:>10.6f} "
              f"{p['entry_price']:>10.2f} {p['unrealized_pnl']:>10.2f} {p['leverage']:>4}x")


def _cmd_orders(config, days):
    from scanner.storage import get_connection

    conn = get_connection()
    rows = conn.execute(
        """SELECT * FROM orders
        WHERE datetime(created_at) > datetime('now', ?)
        ORDER BY created_at DESC""",
        (f"-{days} days",),
    ).fetchall()
    conn.close()

    if not rows:
        print(f"No orders in the last {days} days.")
        return

    print(f"\nRecent Orders ({len(rows)} in last {days} days):")
    print(f"{'ID':>8} {'Coin':>8} {'Dir':>5} {'Type':>10} {'Status':>10} {'Price':>10} {'Mode':>8}")
    print("-" * 70)
    for r in rows:
        price = r["fill_price"] or r["entry_price"] or r["trigger_price"] or 0
        print(f"{r['order_id']:>8} {r['coin']:>8} {r['direction']:>5} "
              f"{r['order_type']:>10} {r['status']:>10} "
              f"{price:>10.2f} {r['execution_mode']:>8}")


def _cmd_kill(config, active):
    from .kill_switch import is_killed, set_kill_switch

    set_kill_switch(active, config)
    status = is_killed(config)
    print(f"Kill switch: {'ACTIVE' if status else 'INACTIVE'}")


def _cmd_report(config, days, mode, discord):
    import requests as req

    from scanner.storage import get_connection

    from .analytics import compare_sim_vs_live, compute_live_metrics, get_closed_trades
    from .report import build_execution_embeds, format_execution_report

    conn = get_connection()
    trades = get_closed_trades(conn, execution_mode=mode, days=days)
    metrics = compute_live_metrics(trades)
    sim_vs_live = compare_sim_vs_live(trades, conn)
    conn.close()

    report_mode = mode or "all"
    report = format_execution_report(metrics, sim_vs_live, report_mode)
    print(report)

    if discord and config.discord_ops_webhook_url:
        embeds = build_execution_embeds(metrics, sim_vs_live, report_mode)
        try:
            resp = req.post(
                config.discord_ops_webhook_url,
                json={"embeds": embeds},
                timeout=10,
            )
            resp.raise_for_status()
            print("\nReport sent to Discord.")
        except Exception as exc:
            print(f"\nDiscord send failed: {exc}")


def _cmd_open_positions(config, discord):
    from datetime import datetime, timezone

    from scanner.storage import get_connection

    from .monitor import send_open_positions_report
    from .storage import get_open_entry_orders

    conn = get_connection()
    entries = get_open_entry_orders(conn, config.exchange, config.mode)
    now = datetime.now(timezone.utc)
    timeout = config.position_timeout_hours

    if not entries:
        print("No open positions.")
    else:
        print(f"\nOpen Positions ({len(entries)}):")
        print(f"{'Coin':>8} {'Dir':>5} {'Age (h)':>8} {'Remaining':>10}")
        print("-" * 35)
        for e in entries:
            ts = e.get("filled_at") or e.get("placed_at") or e.get("created_at")
            if ts:
                try:
                    t = datetime.fromisoformat(ts)
                    if t.tzinfo is None:
                        t = t.replace(tzinfo=timezone.utc)
                    age = (now - t).total_seconds() / 3600
                    left = max(0, timeout - age)
                    left_str = f"{left:.1f}h" if left > 0 else "EXPIRED"
                except (ValueError, TypeError):
                    age, left_str = 0, "?"
            else:
                age, left_str = 0, "?"
            print(f"{e['coin']:>8} {e['direction']:>5} {age:>8.1f} {left_str:>10}")

    if discord:
        ok = send_open_positions_report(config, conn)
        print("Sent to Discord." if ok else "Discord send failed.")

    conn.close()


def _cmd_smoke(config):
    """Execution-path smoke test."""
    from .executor import preflight_check, validate_execution_params

    errors = []

    # 1. Config loaded (already done)
    print("  Config load: OK")

    # 2. Param validation logic works
    validate_execution_params({
        "direction": "LONG", "entry_low": 100, "entry_high": 110,
        "stop_loss": 90, "leverage": 5,
    })
    print("  Param validation: OK")

    # 3. Monitor imports
    from .monitor import check_positions, sync_order_status  # noqa: F401

    print("  Monitor imports: OK")

    # 4. Preflight (if enabled)
    if config.enabled:
        failures = preflight_check(config)
        if failures:
            errors.extend(failures)
            print(f"  Preflight: FAIL ({len(failures)} issues)")
        else:
            print("  Preflight: OK")
    else:
        print("  Preflight: SKIPPED (execution disabled)")

    if errors:
        print(f"\nSmoke FAILED: {len(errors)} errors")
        raise SystemExit(1)
    print("\nSmoke: ALL PASS")


if __name__ == "__main__":
    main()
