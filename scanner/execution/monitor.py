"""Position monitoring — sync order status from exchange, health checks."""

import logging
import time
from datetime import datetime, timezone

import requests

from scanner.config import get_exchange_meta
from scanner.storage import get_connection

from .config import ExecutionConfig
from .exchange import cancel_order, get_auth_exchange, get_exchange_open_orders, get_open_positions, is_rate_limit_error
from .storage import (
    OPEN_QTY_EPSILON,
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
    """Extract fill price from exchange order. Returns None if unavailable.

    Uses 'average' (avg fill price) only. The 'price' field is unreliable for
    trigger orders (TP/SL) on HyperLiquid — it may return the mark price
    instead of the actual fill price.
    """
    raw = exch_order.get("average")
    if raw is not None and float(raw) > 0:
        return float(raw)
    return None


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
                    filled_raw = exch_order.get("filled")
                    if filled_raw is not None and float(filled_raw) == 0:
                        # "closed" with explicit zero fill = exchange auto-cancelled
                        # (e.g. position closed by another order). Not a real fill.
                        log.warning(
                            "%s %s closed with zero fill — treating as cancel (oid=%s)",
                            entry["coin"], order["order_type"], exch_oid,
                        )
                        update_order_status(
                            order["order_id"], "CANCELLED", conn,
                            error_message="Closed with zero fill",
                        )
                        updated += 1
                        if order["order_type"] in ("stop_loss", "tp1", "tp2"):
                            _attempt_exit_order_replacement(
                                order, entry, symbol, exchange, exec_config, conn,
                            )
                        continue

                    fill_qty = float(filled_raw) if filled_raw is not None else float(order["quantity"])
                    fill_price = _extract_fill_price(exch_order)
                    if fill_price is None and order.get("trigger_price"):
                        # Trigger orders (TP/SL) on HyperLiquid may not populate
                        # 'average'. Use trigger_price as reasonable approximation.
                        fill_price = float(order["trigger_price"])
                        log.warning(
                            "%s %s no fill price from exchange — using trigger_price %.5f",
                            entry["coin"], order["order_type"], fill_price,
                        )
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
                        _send_fill_alert(entry, order, fill_price, exec_config, scan_orders=scan_orders)
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
                        if fill_price is None and order.get("trigger_price"):
                            fill_price = float(order["trigger_price"])
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
                            _send_fill_alert(entry, order, fill_price, exec_config, scan_orders=scan_orders)
                            _handle_exit_fill(
                                order, entry, scan_orders,
                                symbol, exchange, exec_config, conn,
                                fill_qty=partial_fill,
                            )
                            break
                    else:
                        update_order_status(order["order_id"], "CANCELLED", conn)
                        updated += 1
                        if order["order_type"] in ("stop_loss", "tp1", "tp2"):
                            _attempt_exit_order_replacement(
                                order, entry, symbol, exchange, exec_config, conn,
                            )

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
                    if order["order_type"] in ("stop_loss", "tp1", "tp2"):
                        _attempt_exit_order_replacement(
                            order, entry, symbol, exchange, exec_config, conn,
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
    """Cancel all other PLACED orders for a scan after one exit fills.

    Always marks DB as CANCELLED — the position is closing, so these orders
    are dead regardless of whether the exchange cancel succeeds (exchange may
    have already auto-cancelled them when the position closed).
    """
    orders = get_orders_for_scan(scan_id, conn)
    for order in orders:
        if order["order_id"] == filled_order_id:
            continue
        if order["status"] != "PLACED":
            continue
        try:
            exch_oid = order.get("exchange_order_id")
            if exch_oid:
                cancel_order(exch_oid, symbol, config)
        except Exception as exc:
            log.error("Failed to cancel %s on exchange for %s: %s",
                      order["order_type"], scan_id, exc)
        # Always mark DB as cancelled — position is closing regardless
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
            sl_order, "market", sl_trigger,
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
                tp2_order, "market", tp2_trigger,
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
            if is_rate_limit_error(exc):
                log.error("Rate limited — aborting %s replacement for %s",
                          order["order_type"], entry["coin"])
                if order["order_type"] == "stop_loss":
                    _send_urgent_sl_alert(entry["coin"], entry["direction"], config)
                elif order["order_type"] in ("tp1", "tp2"):
                    _send_errors_tp_alert(entry["coin"], entry["direction"],
                                          order["order_type"], config)
                break
            if attempt < max_retries:
                time.sleep(attempt)
            else:
                log.error("All %s replacement attempts exhausted for %s — position unprotected",
                          order["order_type"], entry["coin"])
                if order["order_type"] == "stop_loss":
                    _send_urgent_sl_alert(entry["coin"], entry["direction"], config)
                elif order["order_type"] in ("tp1", "tp2"):
                    _send_errors_tp_alert(entry["coin"], entry["direction"],
                                          order["order_type"], config)


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
                symbol, "market", exit_side, rounded_qty, sl_trigger,
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


def _send_errors_tp_alert(
    coin: str, direction: str, order_type: str, config: ExecutionConfig,
) -> bool:
    """Send Discord errors-channel alert when TP replacement exhausts retries."""
    webhook = config.discord_errors_webhook_url
    if not webhook:
        return False
    label = order_type.upper()
    message = (f"**{label} rejected by exchange** for {coin} {direction} "
               f"— profit target not placed")
    try:
        resp = requests.post(webhook, json={"content": message}, timeout=10)
        return resp.status_code == 204
    except Exception as exc:
        log.error("TP errors alert send failed: %s", exc)
        return False


def _attempt_exit_order_replacement(
    order: dict, entry: dict, symbol: str, exchange,
    config: ExecutionConfig, conn,
) -> None:
    """Re-place a rejected/cancelled exit order (SL, TP1, TP2).

    Guards: only exit orders, skip if position closed, skip if no trigger_price.
    """
    if order["order_type"] not in ("stop_loss", "tp1", "tp2"):
        return

    remaining_qty = _get_remaining_qty(entry, conn)
    if remaining_qty <= 0:
        log.info("Skipping %s replacement for %s — position fully closed",
                 order["order_type"], entry["coin"])
        return

    trigger_price = order.get("trigger_price")
    if not trigger_price:
        log.error("Cannot replace %s for %s — no trigger_price on order",
                  order["order_type"], entry["coin"])
        if order["order_type"] == "stop_loss":
            _send_urgent_sl_alert(entry["coin"], entry["direction"], config)
        return

    if order["order_type"] in ("tp1", "tp2"):
        log.warning("Skipping %s replacement for %s — TP rejection not auto-replaced",
                    order["order_type"], entry["coin"])
        _send_errors_tp_alert(entry["coin"], entry["direction"],
                              order["order_type"], config)
        return

    trigger_price = float(trigger_price)
    exit_side = "sell" if entry["direction"] == "LONG" else "buy"
    rounded_qty = float(exchange.amount_to_precision(symbol, remaining_qty))

    if order["order_type"] == "stop_loss":
        extra_params = {"stopLossPrice": trigger_price, "reduceOnly": True}
    else:
        extra_params = {"takeProfitPrice": trigger_price, "reduceOnly": True}

    try:
        _place_replacement_order(
            order, "market", trigger_price,
            extra_params, rounded_qty, remaining_qty, exit_side,
            symbol, exchange, entry, config, conn,
            suffix=":rejected",
        )
    except Exception as exc:
        log.error("Unexpected error replacing %s for %s: %s",
                  order["order_type"], entry["coin"], exc)
        if order["order_type"] == "stop_loss":
            _send_urgent_sl_alert(entry["coin"], entry["direction"], config)
        else:
            _send_errors_tp_alert(entry["coin"], entry["direction"],
                                  order["order_type"], config)


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

        fill_price = _market_close_position(
            entry, symbol, exit_side, remaining_qty, exchange, exec_config, conn,
        )

        if fill_price is not None:
            # Fetch SL trigger for R calculation
            sl_row = conn.execute(
                "SELECT trigger_price FROM orders WHERE scan_id = ? AND order_type = 'stop_loss' LIMIT 1",
                (entry["scan_id"],),
            ).fetchone()
            sl_price = float(sl_row["trigger_price"]) if sl_row and sl_row["trigger_price"] else None

            # Cancel remaining SL/TP orders
            _cancel_remaining_orders(entry["scan_id"], "", symbol, exec_config, conn)
            _send_timeout_alert(entry, age_hours, remaining_qty, exec_config,
                                fill_price=fill_price, sl_price=sl_price)
            timed_out += 1
        else:
            _send_urgent_sl_alert(entry["coin"], entry["direction"], exec_config)

    if own_conn:
        conn.close()

    return timed_out


def _get_remaining_qty(entry: dict, conn) -> float:
    """Calculate remaining open quantity for an entry (entry qty minus filled exits).

    Clamps to 0.0 when the residual is below OPEN_QTY_EPSILON to avoid
    IEEE 754 floating-point artifacts creating ghost positions.
    """
    row = conn.execute(
        """SELECT COALESCE(SUM(quantity), 0) as filled_qty FROM orders
        WHERE scan_id = ? AND order_type IN ('stop_loss', 'tp1', 'tp2', 'timeout_close', 'emergency_close')
        AND status = 'FILLED'""",
        (entry["scan_id"],),
    ).fetchone()
    remaining = entry.get("quantity", 0) - float(row["filled_qty"])
    return remaining if remaining > OPEN_QTY_EPSILON else 0.0


def _market_close_position(
    entry: dict, symbol: str, exit_side: str, qty: float,
    exchange, config: ExecutionConfig, conn,
) -> float | None:
    """Place a reduceOnly market order to close a timed-out position. 3 retries.

    Returns fill price on success, None on failure.

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
            return fill_price if fill_price else 0.0
        except Exception as exc:
            log.error(
                "Timeout close attempt %d/%d for %s failed: %s",
                attempt, max_retries, entry["coin"], exc,
            )
            if attempt < max_retries:
                time.sleep(attempt)

    log.error("All timeout close attempts exhausted for %s — position still open", entry["coin"])
    return None


def _calc_r(entry_price: float, exit_price: float, sl_price: float, direction: str) -> float | None:
    """Calculate R (risk multiple) for a closed position."""
    risk = abs(entry_price - sl_price)
    if risk <= 0:
        return None
    if direction == "LONG":
        return (exit_price - entry_price) / risk
    else:
        return (entry_price - exit_price) / risk


def _send_fill_alert(
    entry: dict, order: dict, fill_price: float | None, config: ExecutionConfig,
    scan_orders: list[dict] | None = None,
) -> bool:
    """Send Discord embed when an exit order (SL/TP) fills."""
    webhook = config.discord_ops_webhook_url
    if not webhook:
        return False

    type_labels = {"stop_loss": "SL", "tp1": "TP1", "tp2": "TP2"}
    label = type_labels.get(order["order_type"], order["order_type"])
    is_win = order["order_type"] in ("tp1", "tp2")
    emoji = "\U0001f7e2" if is_win else "\U0001f534"
    color = 0x26A69A if is_win else 0xEF5350

    # Calculate R value
    r_val = None
    if fill_price and entry.get("entry_price"):
        if order["order_type"] == "stop_loss":
            sl_price = order.get("trigger_price")
        else:
            sl_order = next(
                (o for o in (scan_orders or []) if o["order_type"] == "stop_loss"),
                None,
            )
            sl_price = sl_order.get("trigger_price") if sl_order else None
        if sl_price:
            r_val = _calc_r(float(entry["entry_price"]), fill_price, float(sl_price), entry["direction"])

    fields = []
    if fill_price:
        fields.append({"name": "Fill Price", "value": f"${fill_price:,.5f}", "inline": True})
    if r_val is not None:
        fields.append({"name": "R Value", "value": f"{r_val:+.2f}R", "inline": True})

    embed = {
        "title": f"{emoji} {label} FILLED — {entry['coin']} {entry['direction']}",
        "color": color,
        "fields": fields,
        "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
    }

    try:
        resp = requests.post(webhook, json={"embeds": [embed]}, timeout=10)
        return resp.status_code == 204
    except Exception as exc:
        log.warning("Fill alert failed: %s", exc)
        return False


def _send_timeout_alert(
    entry: dict, age_hours: float, qty: float, config: ExecutionConfig,
    *, fill_price: float | None = None, sl_price: float | None = None,
) -> bool:
    """Send Discord embed for a timed-out position."""
    webhook = config.discord_ops_webhook_url
    if not webhook:
        return False

    fields = [
        {"name": "Duration", "value": f"{age_hours:.1f}h / {config.position_timeout_hours}h limit", "inline": True},
        {"name": "Quantity", "value": str(qty), "inline": True},
    ]

    if fill_price and sl_price and entry.get("entry_price"):
        r_val = _calc_r(float(entry["entry_price"]), fill_price, sl_price, entry["direction"])
        if r_val is not None:
            fields.append({"name": "R Value", "value": f"{r_val:+.2f}R", "inline": True})

    embed = {
        "title": f"\u23f0 TIMEOUT — {entry['coin']} {entry['direction']}",
        "color": 0xFFA726,
        "fields": fields,
        "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
    }

    try:
        resp = requests.post(webhook, json={"embeds": [embed]}, timeout=10)
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


def attempt_sl_recovery(
    issue: dict, config: ExecutionConfig, conn,
) -> bool:
    """Try to place a missing SL for an unprotected position.

    Looks up trigger_price from the FAILED SL order in DB (saved by executor).
    If none found, falls back to the scan's machine_block + reanchor.
    Returns True if SL successfully placed.
    """
    scan_id = issue["scan_id"]
    coin = issue["coin"]
    direction = issue["direction"]

    # Get the entry order for qty / fill_price
    scan_orders = get_orders_for_scan(scan_id, conn)
    entry = next(
        (o for o in scan_orders if o["order_type"] == "entry" and o["status"] == "FILLED"),
        None,
    )
    if not entry:
        log.error("SL recovery: no filled entry for scan %s", scan_id)
        return False

    remaining_qty = _get_remaining_qty(entry, conn)
    if remaining_qty <= 0:
        log.info("SL recovery: position already closed for %s", coin)
        return True

    # Find trigger_price from FAILED SL record
    sl_order = next(
        (o for o in scan_orders if o["order_type"] == "stop_loss"),
        None,
    )
    trigger_price = float(sl_order["trigger_price"]) if sl_order and sl_order.get("trigger_price") else None

    if trigger_price is None:
        # Fallback: recompute from scan machine_block
        try:
            import json

            from scanner.storage import get_connection as get_scanner_conn
            scanner_conn = get_scanner_conn()
            row = scanner_conn.execute(
                "SELECT machine_block, stop_loss FROM scans WHERE scan_id = ?", (scan_id,),
            ).fetchone()
            scanner_conn.close()
            if row and row["machine_block"]:
                mb = json.loads(row["machine_block"])
                fill_price = float(entry.get("fill_price") or entry.get("entry_price") or 0)
                if fill_price and mb.get("stop_loss"):
                    from .sizing import reanchor_exit_prices
                    reanchored = reanchor_exit_prices(mb, fill_price)
                    trigger_price = reanchored["stop_loss"]
            elif row and row["stop_loss"]:
                # Parse "$37.10" format
                trigger_price = float(str(row["stop_loss"]).replace("$", "").replace(",", ""))
        except Exception as exc:
            log.error("SL recovery: failed to get trigger_price from scan: %s", exc)

    if trigger_price is None:
        log.error("SL recovery: no trigger_price available for %s", coin)
        return False

    # Place SL
    meta = get_exchange_meta(config.exchange)
    symbol = meta["symbol_template"].format(coin=coin)
    exit_side = "sell" if direction == "LONG" else "buy"
    exchange = get_auth_exchange(config)
    rounded_qty = float(exchange.amount_to_precision(symbol, remaining_qty))

    order_id = new_order_id()
    cloid = make_cloid(order_id)
    params = {"stopLossPrice": trigger_price, "reduceOnly": True, "clientOrderId": cloid}

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            resp = exchange.create_order(
                symbol, "market", exit_side, rounded_qty, trigger_price, params,
            )
            # Mark old FAILED SL as superseded
            if sl_order and sl_order["status"] == "FAILED":
                update_order_status(sl_order["order_id"], "CANCELLED", conn)
            save_order({
                "order_id": order_id,
                "scan_id": scan_id,
                "coin": coin,
                "direction": direction,
                "order_type": "stop_loss",
                "exchange_order_id": resp.get("id", ""),
                "status": "PLACED",
                "trigger_price": trigger_price,
                "quantity": rounded_qty,
                "exchange": config.exchange,
                "execution_mode": config.mode,
                "placed_at": datetime.now(timezone.utc).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }, conn)
            log.info("SL recovery SUCCESS for %s %s: SL at %s", coin, direction, trigger_price)
            return True
        except Exception as exc:
            log.error("SL recovery attempt %d/%d for %s failed: %s", attempt, max_retries, coin, exc)
            if is_rate_limit_error(exc):
                log.error("Rate limited — aborting SL recovery for %s", coin)
                break
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    log.error("SL recovery exhausted for %s %s — position still unprotected", coin, direction)
    return False
