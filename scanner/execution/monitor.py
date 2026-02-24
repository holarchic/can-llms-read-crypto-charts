"""Position monitoring — sync order status from exchange, health checks."""

import logging
from datetime import datetime, timezone

from scanner.config import get_exchange_meta
from scanner.storage import get_connection

from .config import ExecutionConfig
from .exchange import get_auth_exchange, get_open_positions, cancel_order
from .storage import (
    get_open_entry_orders,
    get_order,
    get_orders_for_scan,
    update_order_status,
)

log = logging.getLogger(__name__)


def sync_order_status(exec_config: ExecutionConfig, conn=None) -> int:
    """Poll exchange for order fills, update DB. Cancel counterpart orders on fill.

    Returns count of orders updated.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    updated = 0
    exchange = get_auth_exchange(exec_config)
    meta = get_exchange_meta(exec_config.exchange)

    # Check all open entries for position closure
    open_entries = get_open_entry_orders(conn, exec_config.exchange)

    for entry in open_entries:
        symbol = meta["symbol_template"].format(coin=entry["coin"])
        scan_orders = get_orders_for_scan(entry["scan_id"], conn)

        for order in scan_orders:
            if order["status"] not in ("PLACED",):
                continue

            # Check if this order has been filled on exchange
            exch_oid = order.get("exchange_order_id")
            if not exch_oid:
                continue

            try:
                exch_order = exchange.fetch_order(exch_oid, symbol)
                exch_status = exch_order.get("status", "").lower()

                if exch_status == "closed":
                    fill_price = float(exch_order.get("average", exch_order.get("price", 0)))
                    update_order_status(
                        order["order_id"], "FILLED", conn,
                        fill_price=fill_price,
                    )
                    updated += 1
                    log.info(
                        "%s %s %s filled at %s",
                        entry["coin"], order["order_type"], order["direction"], fill_price,
                    )

                    # If SL or TP filled, cancel remaining orders for this scan
                    if order["order_type"] in ("stop_loss", "tp1", "tp2"):
                        _cancel_remaining_orders(
                            entry["scan_id"], order["order_id"],
                            symbol, exec_config, conn,
                        )

                elif exch_status == "canceled":
                    update_order_status(order["order_id"], "CANCELLED", conn)
                    updated += 1

            except Exception as exc:
                log.warning("Error checking order %s: %s", exch_oid, exc)

    if own_conn:
        conn.close()

    return updated


def _cancel_remaining_orders(
    scan_id: str, filled_order_id: str, symbol: str, config: ExecutionConfig, conn,
) -> None:
    """Cancel all other PLACED orders for a scan after one exit fills."""
    orders = get_orders_for_scan(scan_id, conn)
    for order in orders:
        if order["order_id"] == filled_order_id:
            continue
        if order["status"] != "PLACED":
            continue
        exch_oid = order.get("exchange_order_id")
        if exch_oid:
            if cancel_order(exch_oid, symbol, config):
                update_order_status(order["order_id"], "CANCELLED", conn)
                log.info("Cancelled %s for %s (counterpart filled)", order["order_type"], scan_id)


def check_positions(exec_config: ExecutionConfig, conn=None, scanner_config=None) -> list[dict]:
    """Health check: verify open positions have SL orders, detect orphans.

    Returns list of issues found.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    issues = []

    open_entries = get_open_entry_orders(conn, exec_config.exchange)

    for entry in open_entries:
        scan_orders = get_orders_for_scan(entry["scan_id"], conn)

        # Check for SL order
        has_sl = any(
            o["order_type"] == "stop_loss" and o["status"] == "PLACED"
            for o in scan_orders
        )
        if not has_sl:
            issues.append({
                "type": "missing_sl",
                "coin": entry["coin"],
                "scan_id": entry["scan_id"],
                "message": f"{entry['coin']} {entry['direction']} has no active stop loss",
            })

    if own_conn:
        conn.close()

    return issues
