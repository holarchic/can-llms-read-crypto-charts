"""Position monitoring — sync order status from exchange, health checks."""

import logging
import time

import requests

from scanner.config import get_exchange_meta
from scanner.storage import get_connection

from .config import ExecutionConfig
from .exchange import cancel_order, get_auth_exchange, get_exchange_open_orders
from .storage import (
    get_open_entry_orders,
    get_orders_for_scan,
    make_cloid,
    update_order_quantity,
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
    open_entries = get_open_entry_orders(conn, exec_config.exchange, exec_config.mode)

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
                    fill_qty = float(exch_order["filled"]) if exch_order.get("filled") else order["quantity"]
                    if fill_qty < order["quantity"]:
                        log.warning(
                            "%s %s partial fill: expected %s, got %s",
                            entry["coin"], order["order_type"], order["quantity"], fill_qty,
                        )
                    update_order_status(
                        order["order_id"], "FILLED", conn,
                        fill_price=fill_price,
                        filled_quantity=fill_qty,
                    )
                    updated += 1
                    log.info(
                        "%s %s %s filled at %s (qty %s)",
                        entry["coin"], order["order_type"], order["direction"], fill_price, fill_qty,
                    )

                    # Handle exit fill: route based on partial vs full close
                    if order["order_type"] in ("stop_loss", "tp1", "tp2"):
                        _handle_exit_fill(
                            order, entry, scan_orders,
                            symbol, exchange, exec_config, conn,
                            fill_qty=fill_qty,
                        )
                        break  # stale list — re-check next cycle

                elif exch_status == "canceled":
                    partial_fill = float(exch_order.get("filled", 0))
                    if partial_fill > 0:
                        # Partial fill before cancel — treat as fill for the filled amount
                        fill_price = float(exch_order.get("average", exch_order.get("price", 0)))
                        log.warning(
                            "%s %s canceled with partial fill: %s of %s at %s",
                            entry["coin"], order["order_type"], partial_fill, order["quantity"], fill_price,
                        )
                        update_order_status(
                            order["order_id"], "FILLED", conn,
                            fill_price=fill_price,
                            filled_quantity=partial_fill,
                        )
                        updated += 1

                        if order["order_type"] in ("stop_loss", "tp1", "tp2"):
                            _handle_exit_fill(
                                order, entry, scan_orders,
                                symbol, exchange, exec_config, conn,
                                fill_qty=partial_fill,
                            )
                            break
                    else:
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


def _handle_exit_fill(
    filled_order: dict, entry: dict, scan_orders: list[dict],
    symbol: str, exchange, config: ExecutionConfig, conn,
    fill_qty: float | None = None,
) -> None:
    """Route exit fill: TP1 → partial close (replace SL), SL/TP2 → full close."""
    if filled_order["order_type"] == "tp1":
        _handle_tp1_fill(filled_order, entry, scan_orders, symbol, exchange, config, conn,
                         fill_qty=fill_qty)
    else:
        # SL or TP2 filled — position fully closed, cancel remaining
        _cancel_remaining_orders(
            entry["scan_id"], filled_order["order_id"],
            symbol, config, conn,
        )


def _handle_tp1_fill(
    tp1_order: dict, entry: dict, scan_orders: list[dict],
    symbol: str, exchange, config: ExecutionConfig, conn,
    fill_qty: float | None = None,
) -> None:
    """TP1 filled: cancel old SL, place replacement SL at remaining quantity, keep TP2."""
    tp1_qty = fill_qty if fill_qty is not None else tp1_order.get("quantity", 0)
    remaining_qty = entry.get("quantity", 0) - tp1_qty

    if remaining_qty <= 0:
        # TP1 was full size — treat as full close
        _cancel_remaining_orders(
            entry["scan_id"], tp1_order["order_id"],
            symbol, config, conn,
        )
        return

    # Find the current SL order
    sl_order = next(
        (o for o in scan_orders if o["order_type"] == "stop_loss" and o["status"] == "PLACED"),
        None,
    )
    if not sl_order:
        log.error("No active SL found for %s after TP1 fill — position unprotected", entry["coin"])
        return

    # Cancel old SL on exchange
    old_exch_oid = sl_order.get("exchange_order_id")
    if old_exch_oid and not cancel_order(old_exch_oid, symbol, config):
        log.error("Failed to cancel old SL %s for %s — position may have duplicate SLs",
                  old_exch_oid, entry["coin"])
        return

    # DB reflects reality: old SL is gone from exchange
    update_order_status(sl_order["order_id"], "CANCELLED", conn)

    # Place replacement SL at remaining quantity
    sl_trigger = sl_order.get("trigger_price")
    exit_side = "sell" if entry["direction"] == "LONG" else "buy"

    rounded_qty = float(exchange.amount_to_precision(symbol, remaining_qty))
    cloid = make_cloid(sl_order["order_id"] + ":replaced")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            sl_resp = exchange.create_order(
                symbol, "STOP_MARKET", exit_side, rounded_qty, sl_trigger,
                {"stopLossPrice": sl_trigger, "reduceOnly": True, "clientOrderId": cloid},
            )
            new_exch_oid = sl_resp.get("id", "")
            update_order_quantity(sl_order["order_id"], remaining_qty, new_exch_oid, conn)
            update_order_status(sl_order["order_id"], "PLACED", conn)
            log.info(
                "Replaced SL for %s: qty %s → %s (TP1 filled %s)",
                entry["coin"], entry["quantity"], remaining_qty, tp1_qty,
            )
            break
        except Exception as exc:
            log.error("SL replacement attempt %d/%d for %s failed: %s",
                      attempt, max_retries, entry["coin"], exc)
            if attempt < max_retries:
                time.sleep(attempt)  # 1s, 2s exponential backoff
            else:
                log.error("All SL replacement attempts exhausted for %s — position unprotected",
                          entry["coin"])
                _send_urgent_sl_alert(entry["coin"], entry["direction"], config)


def _send_urgent_sl_alert(coin: str, direction: str, config: ExecutionConfig) -> bool:
    """Send URGENT Discord alert when SL replacement exhausts retries."""
    webhook = config.discord_execution_webhook_url
    if not webhook:
        return False
    message = (f"@everyone **URGENT**: SL replacement FAILED for {coin} {direction} "
               f"after 3 attempts — position running WITHOUT stop loss")
    try:
        resp = requests.post(webhook, json={"content": message}, timeout=10)
        return resp.status_code == 204
    except Exception as exc:
        log.error("URGENT SL alert send failed: %s", exc)
        return False


def check_positions(exec_config: ExecutionConfig, conn=None, scanner_config=None) -> list[dict]:
    """Health check: verify open positions have SL orders, detect orphans.

    Batch-fetches open orders from exchange to verify PLACED SLs actually exist.
    Returns list of issues found.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    issues = []

    # Fetch all open orders from exchange in one call
    try:
        exchange_orders = get_exchange_open_orders(exec_config)
        exchange_oids = {str(o.get("id", "")) for o in exchange_orders}
    except Exception as exc:
        log.warning("Could not fetch exchange orders for health check: %s", exc)
        exchange_oids = None  # Skip verification if exchange unreachable

    open_entries = get_open_entry_orders(conn, exec_config.exchange, exec_config.mode)

    for entry in open_entries:
        scan_orders = get_orders_for_scan(entry["scan_id"], conn)

        has_sl = False
        for o in scan_orders:
            if o["order_type"] == "stop_loss" and o["status"] == "PLACED":
                has_sl = True
                # Verify SL exists on exchange
                if exchange_oids is not None:
                    exch_oid = o.get("exchange_order_id", "")
                    if exch_oid and exch_oid not in exchange_oids:
                        issues.append({
                            "type": "phantom_sl",
                            "coin": entry["coin"],
                            "scan_id": entry["scan_id"],
                            "message": (
                                f"{entry['coin']} {entry['direction']} SL exists "
                                f"in DB but not on exchange"
                            ),
                        })
                break

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
