"""Position monitoring — sync order status from exchange, health checks."""

import logging
import time
from datetime import datetime, timezone

import requests

from scanner.config import get_exchange_meta
from scanner.storage import get_connection

from .config import ExecutionConfig
from .exchange import cancel_order, get_auth_exchange, get_exchange_open_orders, get_open_positions
from .storage import (
    get_open_entry_orders,
    get_orders_for_scan,
    make_cloid,
    new_order_id,
    save_order,
    update_order_quantity,
    update_order_status,
)

log = logging.getLogger(__name__)


def _extract_fill_price(exch_order: dict) -> float | None:
    """Extract fill price from exchange order. Returns None if unavailable."""
    raw = exch_order.get("average") or exch_order.get("price")
    return float(raw) if raw else None


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
                    fill_price = _extract_fill_price(exch_order)
                    fill_qty = float(exch_order.get("filled") or order["quantity"])
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
                        _send_fill_alert(entry, order, fill_price, exec_config)
                        _handle_exit_fill(
                            order, entry, scan_orders,
                            symbol, exchange, exec_config, conn,
                            fill_qty=fill_qty,
                        )
                        break  # stale list — re-check next cycle

                elif exch_status == "canceled":
                    partial_fill = float(exch_order.get("filled") or 0)
                    if partial_fill > 0:
                        # Partial fill before cancel — treat as fill for the filled amount
                        fill_price = _extract_fill_price(exch_order)
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
                            _send_fill_alert(entry, order, fill_price, exec_config)
                            _handle_exit_fill(
                                order, entry, scan_orders,
                                symbol, exchange, exec_config, conn,
                                fill_qty=partial_fill,
                            )
                            break
                    else:
                        update_order_status(order["order_id"], "CANCELLED", conn)
                        updated += 1

                elif exch_status in ("expired", "rejected"):
                    update_order_status(
                        order["order_id"], "CANCELLED", conn,
                        error_message=f"Exchange status: {exch_status}",
                    )
                    updated += 1
                    log.warning(
                        "%s %s %s on exchange (oid=%s)",
                        entry["coin"], order["order_type"], exch_status, exch_oid,
                    )

                elif exch_status not in ("open", "pending", "new"):
                    log.warning(
                        "%s %s unknown exchange status '%s' (oid=%s)",
                        entry["coin"], order["order_type"], exch_status, exch_oid,
                    )

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
    """Route exit fill: TP1 → partial close (replace SL), SL/TP2 → full close or partial."""
    if filled_order["order_type"] == "tp1":
        _handle_tp1_fill(filled_order, entry, scan_orders, symbol, exchange, config, conn,
                         fill_qty=fill_qty)
    else:
        # Check for partial fill — residual position needs protection
        order_qty = filled_order.get("quantity", 0)
        actual_fill = fill_qty if fill_qty is not None else order_qty
        if actual_fill < order_qty and order_qty > 0:
            _handle_partial_exit_fill(
                filled_order, entry, scan_orders,
                symbol, exchange, config, conn,
                fill_qty=actual_fill,
            )
        else:
            # Full close — cancel everything
            _cancel_remaining_orders(
                entry["scan_id"], filled_order["order_id"],
                symbol, config, conn,
            )


def _get_live_position_qty(
    coin: str, direction: str, db_qty: float, config: ExecutionConfig,
) -> float:
    """Cross-check DB qty against exchange. Use exchange if available, log mismatch."""
    try:
        positions = get_open_positions(config)
        for pos in positions:
            if pos["coin"] == coin and pos["direction"] == direction:
                exch_qty = pos["size"]
                if abs(exch_qty - db_qty) / max(db_qty, 1e-10) > 0.01:
                    log.warning(
                        "%s %s qty mismatch: DB=%.6f exchange=%.6f — using exchange",
                        coin, direction, db_qty, exch_qty,
                    )
                return exch_qty
        log.warning("%s %s not found on exchange — using DB qty %.6f", coin, direction, db_qty)
    except Exception as exc:
        log.warning("Position qty check failed for %s: %s — using DB", coin, exc)
    return db_qty


def _handle_partial_exit_fill(
    filled_order: dict, entry: dict, scan_orders: list[dict],
    symbol: str, exchange, config: ExecutionConfig, conn,
    fill_qty: float,
) -> None:
    """SL or TP2 partially filled: cancel remaining orders, re-place at residual qty."""
    residual_qty = filled_order["quantity"] - fill_qty
    residual_qty = _get_live_position_qty(
        entry["coin"], entry["direction"], residual_qty, config,
    )

    # Cancel all remaining PLACED orders on exchange
    _cancel_remaining_orders(
        entry["scan_id"], filled_order["order_id"],
        symbol, config, conn,
    )

    exit_side = "sell" if entry["direction"] == "LONG" else "buy"
    rounded_qty = float(exchange.amount_to_precision(symbol, residual_qty))

    # Determine which orders to re-place
    filled_type = filled_order["order_type"]

    # Always need an SL for the residual
    sl_order = (
        filled_order if filled_type == "stop_loss"
        else next((o for o in scan_orders if o["order_type"] == "stop_loss"), None)
    )
    if sl_order:
        sl_trigger = sl_order.get("trigger_price")
        _place_replacement_order(
            sl_order, "STOP_MARKET", sl_trigger,
            {"stopLossPrice": sl_trigger, "reduceOnly": True},
            rounded_qty, residual_qty, exit_side,
            symbol, exchange, entry, config, conn,
            suffix=":partial",
        )

    # Re-place TP2 at residual qty
    if filled_type == "stop_loss":
        # SL partial fill — re-place the TP2 that was just cancelled
        tp2_order = next(
            (o for o in scan_orders if o["order_type"] == "tp2" and o["status"] in ("PLACED", "CANCELLED")),
            None,
        )
    elif filled_type == "tp2":
        # TP2 partial fill — re-place the TP2 itself
        tp2_order = filled_order
    else:
        tp2_order = None

    if tp2_order:
            tp2_trigger = tp2_order.get("trigger_price")
            _place_replacement_order(
                tp2_order, "TAKE_PROFIT_MARKET", tp2_trigger,
                {"takeProfitPrice": tp2_trigger, "reduceOnly": True},
                rounded_qty, residual_qty, exit_side,
                symbol, exchange, entry, config, conn,
                suffix=":partial",
            )


def _place_replacement_order(
    order: dict, order_type_str: str, trigger_price: float,
    extra_params: dict, rounded_qty: float, residual_qty: float,
    exit_side: str, symbol: str, exchange, entry: dict,
    config: ExecutionConfig, conn, suffix: str = ":replaced",
) -> None:
    """Place a replacement SL or TP order with retry logic."""
    cloid = make_cloid(order["order_id"] + suffix)
    params = {**extra_params, "clientOrderId": cloid}

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            resp = exchange.create_order(
                symbol, order_type_str, exit_side, rounded_qty, trigger_price, params,
            )
            new_exch_oid = resp.get("id", "")
            update_order_quantity(order["order_id"], residual_qty, new_exch_oid, conn)
            update_order_status(order["order_id"], "PLACED", conn)
            log.info(
                "Replaced %s for %s: qty %s → %s (partial fill)",
                order["order_type"], entry["coin"], order["quantity"], residual_qty,
            )
            break
        except Exception as exc:
            log.error("%s replacement attempt %d/%d for %s failed: %s",
                      order["order_type"], attempt, max_retries, entry["coin"], exc)
            if attempt < max_retries:
                time.sleep(attempt)
            else:
                log.error("All %s replacement attempts exhausted for %s — position unprotected",
                          order["order_type"], entry["coin"])
                if order["order_type"] == "stop_loss":
                    _send_urgent_sl_alert(entry["coin"], entry["direction"], config)


def _handle_tp1_fill(
    tp1_order: dict, entry: dict, scan_orders: list[dict],
    symbol: str, exchange, config: ExecutionConfig, conn,
    fill_qty: float | None = None,
) -> None:
    """TP1 filled: cancel old SL, place replacement SL at remaining quantity, keep TP2."""
    tp1_qty = fill_qty if fill_qty is not None else tp1_order.get("quantity", 0)
    remaining_qty = entry.get("quantity", 0) - tp1_qty
    remaining_qty = _get_live_position_qty(
        entry["coin"], entry["direction"], remaining_qty, config,
    )

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
    webhook = config.discord_urgent_webhook_url
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


def enforce_position_timeouts(exec_config: ExecutionConfig, conn=None) -> int:
    """Market-close positions that exceed position_timeout_hours.

    Returns count of positions timed out.
    """
    if exec_config.position_timeout_hours <= 0:
        return 0

    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    timed_out = 0
    exchange = get_auth_exchange(exec_config)
    meta = get_exchange_meta(exec_config.exchange)
    now = datetime.now(timezone.utc)

    open_entries = get_open_entry_orders(conn, exec_config.exchange, exec_config.mode)

    for entry in open_entries:
        # Parse entry fill time — fallback chain: filled_at → placed_at → created_at
        entry_time_str = entry.get("filled_at") or entry.get("placed_at") or entry.get("created_at")
        if not entry_time_str:
            continue

        try:
            entry_time = datetime.fromisoformat(entry_time_str)
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

        age_hours = (now - entry_time).total_seconds() / 3600
        if age_hours <= exec_config.position_timeout_hours:
            continue

        # Calculate remaining qty
        remaining_qty = _get_remaining_qty(entry, conn)
        if remaining_qty <= 0:
            continue

        symbol = meta["symbol_template"].format(coin=entry["coin"])
        exit_side = "sell" if entry["direction"] == "LONG" else "buy"

        success = _market_close_position(
            entry, symbol, exit_side, remaining_qty, exchange, exec_config, conn,
        )

        if success:
            # Cancel remaining SL/TP orders
            _cancel_remaining_orders(entry["scan_id"], "", symbol, exec_config, conn)
            _send_timeout_alert(entry, age_hours, remaining_qty, exec_config)
            timed_out += 1
        else:
            _send_urgent_sl_alert(entry["coin"], entry["direction"], exec_config)

    if own_conn:
        conn.close()

    return timed_out


def _get_remaining_qty(entry: dict, conn) -> float:
    """Calculate remaining open quantity for an entry (entry qty minus filled exits)."""
    row = conn.execute(
        """SELECT COALESCE(SUM(quantity), 0) as filled_qty FROM orders
        WHERE scan_id = ? AND order_type IN ('stop_loss', 'tp1', 'tp2', 'timeout_close')
        AND status = 'FILLED'""",
        (entry["scan_id"],),
    ).fetchone()
    return entry.get("quantity", 0) - float(row["filled_qty"])


def _market_close_position(
    entry: dict, symbol: str, exit_side: str, qty: float,
    exchange, config: ExecutionConfig, conn,
) -> bool:
    """Place a reduceOnly market order to close a timed-out position. 3 retries.

    HyperLiquid has no native market orders — ccxt simulates via limit IOC
    with slippage. Requires current price as the base for slippage calc.
    """
    order_id = new_order_id()
    cloid = make_cloid(order_id)
    rounded_qty = float(exchange.amount_to_precision(symbol, qty))

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            ticker = exchange.fetch_ticker(symbol)
            price = ticker["last"]
            resp = exchange.create_order(
                symbol, "market", exit_side, rounded_qty, price,
                {"reduceOnly": True, "clientOrderId": cloid},
            )
            fill_price = _extract_fill_price(resp)
            save_order({
                "order_id": order_id,
                "scan_id": entry["scan_id"],
                "coin": entry["coin"],
                "direction": entry["direction"],
                "order_type": "timeout_close",
                "exchange_order_id": resp.get("id", ""),
                "status": "FILLED",
                "fill_price": fill_price,
                "quantity": rounded_qty,
                "exchange": config.exchange,
                "execution_mode": config.mode,
                "filled_at": datetime.now(timezone.utc).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }, conn)
            log.info(
                "Timeout close %s %s: qty %s at %s",
                entry["coin"], entry["direction"], rounded_qty, fill_price,
            )
            return True
        except Exception as exc:
            log.error(
                "Timeout close attempt %d/%d for %s failed: %s",
                attempt, max_retries, entry["coin"], exc,
            )
            if attempt < max_retries:
                time.sleep(attempt)

    log.error("All timeout close attempts exhausted for %s — position still open", entry["coin"])
    return False


def _send_fill_alert(entry: dict, order: dict, fill_price: float | None, config: ExecutionConfig) -> bool:
    """Send Discord notification when an exit order (SL/TP) fills."""
    webhook = config.discord_ops_webhook_url
    if not webhook:
        return False

    type_labels = {"stop_loss": "SL", "tp1": "TP1", "tp2": "TP2"}
    label = type_labels.get(order["order_type"], order["order_type"])
    is_win = order["order_type"] in ("tp1", "tp2")
    emoji = "\U0001f7e2" if is_win else "\U0001f534"
    price_str = f" @ ${fill_price:,.2f}" if fill_price else ""
    message = f"{emoji} **{label} FILLED** {entry['coin']} {entry['direction']}{price_str}"

    try:
        resp = requests.post(webhook, json={"content": message}, timeout=10)
        return resp.status_code == 204
    except Exception as exc:
        log.warning("Fill alert failed: %s", exc)
        return False


def _send_timeout_alert(entry: dict, age_hours: float, qty: float, config: ExecutionConfig) -> bool:
    """Send Discord notification for a timed-out position."""
    webhook = config.discord_ops_webhook_url
    if not webhook:
        return False
    message = (
        f"**TIMEOUT**: {entry['coin']} {entry['direction']} closed after "
        f"{age_hours:.1f}h (limit: {config.position_timeout_hours}h) — qty {qty}"
    )
    try:
        resp = requests.post(webhook, json={"content": message}, timeout=10)
        return resp.status_code == 204
    except Exception as exc:
        log.error("Timeout alert send failed: %s", exc)
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
        issues.append({
            "type": "exchange_fetch_failed",
            "coin": "",
            "scan_id": "",
            "message": f"Exchange order fetch failed: {exc} — SL verification skipped",
        })

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
                            "direction": entry["direction"],
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
                "direction": entry["direction"],
                "scan_id": entry["scan_id"],
                "message": f"{entry['coin']} {entry['direction']} has no active stop loss",
            })

    if own_conn:
        conn.close()

    return issues
