"""Strategy CLI — create, list, activate, disable.

Replay/sweep/calibrate/report/backfill were removed in T-001 (2026-05-12)
when llm-trader stopped persisting candles. Strategy management for the
live scanner remains: a row in `strategies` configures the active screener
parameters that `scanner/pipeline.py` reads via `get_strategy`.
"""

import argparse
import sys
from datetime import datetime, timezone

from ..db import get_connection
from .models import Strategy
from .store import (
    activate_strategy,
    ensure_baseline,
    get_strategy,
    list_strategies,
    save_strategy,
)


def cmd_create(args):
    conn = get_connection()
    sid = args.id or f"v{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"

    strategy = Strategy(
        strategy_id=sid,
        label=args.label,
        swing_lookback=args.swing_lookback,
        sr_cluster_pct=args.sr_cluster_pct,
        proximity_atr_mult=args.proximity_mult,
        max_setups_per_coin=args.max_setups,
        atr_stop_mult=args.atr_stop,
        atr_target_mult=args.atr_target,
        ruleset_version=args.ruleset_version or "",
        notes=args.notes or "",
    )
    save_strategy(strategy, conn)
    conn.close()
    print(f"Created strategy: {sid} ({args.label})")


def cmd_list(args):
    conn = get_connection()
    ensure_baseline(conn)
    strategies = list_strategies(conn)
    conn.close()

    if not strategies:
        print("No strategies found.")
        return

    print(f"{'ID':<20} {'Label':<16} {'Active':>6} {'Stop':>5} {'Target':>6} {'Prox':>5} {'Disabled':<8} {'Created':<20}")
    print("-" * 100)
    for s in strategies:
        active = " *" if s.is_active else ""
        created = s.created_at[:19] if s.created_at else ""
        disabled = s.disabled_types or "-"
        print(
            f"{s.strategy_id:<20} {s.label:<16} {active:>6} "
            f"{s.atr_stop_mult:>5.1f} {s.atr_target_mult:>6.1f} "
            f"{s.proximity_atr_mult:>5.1f} {disabled:<8} {created:<20}"
        )


def cmd_activate(args):
    conn = get_connection()
    if not get_strategy(args.strategy_id, conn):
        print(f"Strategy '{args.strategy_id}' not found.")
        conn.close()
        sys.exit(1)

    activate_strategy(args.strategy_id, conn)
    conn.close()
    print(f"Activated strategy: {args.strategy_id}")


def cmd_disable(args):
    conn = get_connection()
    strategy = get_strategy(args.strategy_id, conn)
    if not strategy:
        print(f"Strategy '{args.strategy_id}' not found.")
        conn.close()
        sys.exit(1)

    if args.types.lower() == "none":
        strategy.disabled_types = ""
        print(f"Cleared disabled types for strategy '{args.strategy_id}'")
    else:
        types = ",".join(t.strip().upper() for t in args.types.split(",") if t.strip())
        strategy.disabled_types = types
        print(f"Disabled types for strategy '{args.strategy_id}': {types}")

    save_strategy(strategy, conn)
    conn.close()


def build_parser() -> argparse.ArgumentParser:
    """Construct the strategy CLI parser. Factored out for smoke testing."""
    parser = argparse.ArgumentParser(
        prog="scanner.strategy",
        description="Strategy versioning + live-scanner parameter management",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # create
    p_create = sub.add_parser("create", help="Create a new strategy")
    p_create.add_argument("--id", type=str, default=None, help="Strategy ID (auto-generated if omitted)")
    p_create.add_argument("--label", type=str, required=True, help="Human-readable label")
    p_create.add_argument("--swing-lookback", type=int, default=5)
    p_create.add_argument("--sr-cluster-pct", type=float, default=0.015)
    p_create.add_argument("--proximity-mult", type=float, default=1.5)
    p_create.add_argument("--max-setups", type=int, default=3)
    p_create.add_argument("--atr-stop", type=float, default=1.5)
    p_create.add_argument("--atr-target", type=float, default=3.0)
    p_create.add_argument("--ruleset-version", type=str, default=None, help="unified-ta-ruleset version (e.g. v1.2.0)")
    p_create.add_argument("--notes", type=str, default=None)
    p_create.set_defaults(func=cmd_create)

    # list
    p_list = sub.add_parser("list", help="List all strategies")
    p_list.set_defaults(func=cmd_list)

    # activate
    p_activate = sub.add_parser("activate", help="Set active strategy")
    p_activate.add_argument("strategy_id", help="Strategy ID to activate")
    p_activate.set_defaults(func=cmd_activate)

    # disable
    p_disable = sub.add_parser("disable", help="Disable setup types for a strategy")
    p_disable.add_argument("strategy_id", help="Strategy ID")
    p_disable.add_argument("types", help="Comma-separated types to disable (e.g. 'A,E'), or 'none' to clear")
    p_disable.set_defaults(func=cmd_disable)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)
