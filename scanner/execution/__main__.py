"""CLI for execution module. Run with: python -m scanner.execution"""

import argparse

from .config import load_execution_config


def main():
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
    parser.add_argument("command", choices=["monitor", "positions", "orders", "kill", "unkill", "report"],
                        help="Subcommand to run")
    parser.add_argument("--days", type=int, default=7, help="Days of history (default: 7)")
    parser.add_argument("--mode", choices=["testnet", "live"], help="Filter by execution mode")
    parser.add_argument("--discord", action="store_true", help="Send report to Discord")

    args = parser.parse_args()
    config = load_execution_config()

    if args.command == "monitor":
        _cmd_monitor(config)
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


def _cmd_monitor(config):
    from scanner.storage import get_connection

    from .monitor import check_positions, sync_order_status

    conn = get_connection()
    updated = sync_order_status(config, conn)
    print(f"Orders updated: {updated}")

    issues = check_positions(config, conn)
    if issues:
        print(f"\nHealth check issues ({len(issues)}):")
        for issue in issues:
            print(f"  [{issue['type']}] {issue['message']}")
    else:
        print("Health check: OK")

    conn.close()


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

    if discord and config.discord_execution_webhook_url:
        embeds = build_execution_embeds(metrics, sim_vs_live, report_mode)
        try:
            req.post(
                config.discord_execution_webhook_url,
                json={"embeds": embeds},
                timeout=10,
            )
            print("\nReport sent to Discord.")
        except Exception as exc:
            print(f"\nDiscord send failed: {exc}")


if __name__ == "__main__":
    main()
