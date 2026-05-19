"""Strategy CLI — list, disable.

A `strategies` row holds the `disabled_types` the live scanner honours
(`scanner/pipeline.py` reads `get_strategy(strategy_id).disabled_types_set`).
"""

import argparse
import sys

from ..db import get_connection
from .store import ensure_baseline, get_strategy, list_strategies, set_disabled_types


def cmd_list(args):
    conn = get_connection()
    ensure_baseline(conn)
    strategies = list_strategies(conn)
    conn.close()

    print(f"{'ID':<24} {'Disabled':<12}")
    print("-" * 38)
    for s in strategies:
        print(f"{s.strategy_id:<24} {s.disabled_types or '-':<12}")


def cmd_disable(args):
    conn = get_connection()
    if not get_strategy(args.strategy_id, conn):
        print(f"Strategy '{args.strategy_id}' not found.")
        conn.close()
        sys.exit(1)

    if args.types.lower() == "none":
        set_disabled_types(args.strategy_id, "", conn)
        print(f"Cleared disabled types for strategy '{args.strategy_id}'")
    else:
        types = ",".join(t.strip().upper() for t in args.types.split(",") if t.strip())
        set_disabled_types(args.strategy_id, types, conn)
        print(f"Disabled types for strategy '{args.strategy_id}': {types}")

    conn.close()


def build_parser() -> argparse.ArgumentParser:
    """Construct the strategy CLI parser. Factored out for smoke testing."""
    parser = argparse.ArgumentParser(
        prog="scanner.strategy",
        description="Live-scanner disabled-setup-type management",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List strategies (id + disabled_types)")
    p_list.set_defaults(func=cmd_list)

    p_disable = sub.add_parser("disable", help="Disable setup types for a strategy")
    p_disable.add_argument("strategy_id", help="Strategy ID")
    p_disable.add_argument("types", help="Comma-separated types to disable (e.g. 'A,E'), or 'none' to clear")
    p_disable.set_defaults(func=cmd_disable)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)
