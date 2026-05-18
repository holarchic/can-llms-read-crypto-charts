"""Exit-order lifecycle — fill routing, terminal handlers, replacement, counterpart cancels.

Extracted from `monitor.py` (T-016 Phase 4a, SPEC #27): the cohesive
exit-order-lifecycle cluster. `_handle_exit_fill` routes an exchange-side
exit fill to its terminal (`_handle_tp1_fill` / `_handle_partial_exit_fill` /
direct full-close); the cluster owns the atomic full-close boundary's DB-mark
half (`_cancel_counterparts_on_exchange` exchange pass + `_mark_counterparts_cancelled`
in-txn DB-mark) and SL/TP replacement. These process orders ALREADY filled on
the exchange — they never call `market_close_position`, so there is no
`exit_fills -> position_close` edge.

Logger provenance is byte-identical: logs through the
`"scanner.execution.monitor"` logger (the literal name, NOT `__name__`).
One-directional: monitor -> exit_fills; exit_fills -> monitor_alerts (urgent/
errors alert composition) + .storage/.exchange/.config. No back-edge.
"""

import logging
import time
from datetime import datetime, timezone

from .config import ExecutionConfig
from .exchange import cancel_order, get_open_positions, is_rate_limit_error
from .monitor_alerts import _send_errors_tp_alert, _send_urgent_sl_alert
from .storage import (
    EXIT_ORDER_TYPES_SQL,
    OPEN_QTY_EPSILON,
    get_orders_for_scan,
    make_cloid,
    new_order_id,
    save_order,
    transaction,
    update_order_status,
)

log = logging.getLogger("scanner.execution.monitor")


def _cancel_counterparts_on_exchange(
    scan_id: str, filled_order_id: str, symbol: str, config: ExecutionConfig, conn,
) -> list[dict]:
    """Exchange-cancel pass for the closing position's PLACED counterparts.

    The exchange half of the atomic full-close boundary (C4#6). Issues every
    `cancel_order()` here — BEFORE the caller opens its close `transaction()`
    — because the invariant forbids an exchange REST call inside a DB
    transaction. Per-order try/except preserved: the exchange may have
    auto-cancelled a counterpart when the position closed. Returns the orders
    to mark CANCELLED; the caller passes them to `_mark_counterparts_cancelled`
    INSIDE the same transaction as the close-state write, so a rollback leaves
    the close unrecorded → the position stays in `get_open_entry_orders()` →
    next monitor cycle retries (self-healing, DB-failure scoped).
    """
    targets = []
    for order in get_orders_for_scan(scan_id, conn):
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
        targets.append(order)
    return targets


def _mark_counterparts_cancelled(targets: list[dict], conn) -> None:
    """DB-mark half: mark every counterpart CANCELLED regardless of the
    exchange-cancel outcome (the position is closing, so they are dead).

    MUST run inside the caller's close `transaction()` so it commits as one
    unit with the close-state write — the always-mark-CANCELLED invariant of
    the former `cancel_remaining_orders`, now atomic with the close.
    """
    for order in targets:
        update_order_status(order["order_id"], "CANCELLED", conn)
        log.info("Cancelled %s for %s (counterpart of close)",
                 order["order_type"], order["scan_id"])


def _handle_exit_fill(
    filled_order: dict, entry: dict, scan_orders: list[dict],
    symbol: str, exchange, config: ExecutionConfig, conn,
    fill_price: float | None = None,
    fill_qty: float | None = None,
) -> None:
    """Route an exit fill and OWN its FILLED DB write.

    The filled-exit FILLED mark is OWNED by the route terminal, never written
    here, so EVERY full-close outcome — SL/tp2 direct, full-size tp1, and a
    partial that the live-qty check resolves to zero residual — commits the
    FILLED row atomically with the counterpart-CANCELLED marks (the atomic
    full-close boundary, C4#6). A close-state write committed before its
    counterpart cancels would strand PLACED counterparts on a position
    `get_open_entry_orders()` no longer returns — the phantom_sl family this
    exists to kill. The full-vs-partial decision for tp1/partial needs a live
    `_get_live_position_qty` exchange call, so it is made INSIDE those
    handlers; they own the FILLED write per terminal (atomic when it resolves
    to a full close, standalone when the position stays open with a
    replacement order — the exchange-interleaved replacement saga is out of
    C4#6 scope). The exchange-cancel pass always runs OUTSIDE the transaction
    (invariant: no exchange REST call inside a DB txn).
    """
    oid = filled_order["order_id"]
    if filled_order["order_type"] == "tp1":
        _handle_tp1_fill(filled_order, entry, scan_orders, symbol, exchange, config, conn,
                         fill_price=fill_price, fill_qty=fill_qty)
        return
    # Check for partial fill — residual position needs protection
    order_qty = filled_order.get("quantity", 0)
    actual_fill = fill_qty if fill_qty is not None else order_qty
    if actual_fill < order_qty and order_qty > 0:
        _handle_partial_exit_fill(
            filled_order, entry, scan_orders,
            symbol, exchange, config, conn,
            fill_qty=actual_fill, fill_price=fill_price,
        )
        return
    # SL/tp2 direct full close — atomic boundary: exchange-cancel pass first
    # (outside the txn), then the exit FILLED mark + counterpart CANCELLED
    # marks commit as ONE unit. A rollback leaves the close unrecorded so the
    # position stays in the open set and the next cycle retries.
    targets = _cancel_counterparts_on_exchange(
        entry["scan_id"], oid, symbol, config, conn,
    )
    with transaction(conn):
        update_order_status(oid, "FILLED", conn,
                            fill_price=fill_price, filled_quantity=fill_qty)
        _mark_counterparts_cancelled(targets, conn)


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


_SIDE_TO_DIRECTION = {
    "long": "LONG", "short": "SHORT", "buy": "LONG", "sell": "SHORT",
}


def _position_closed_on_exchange(coin: str, direction: str, exchange) -> bool:
    """Return True iff the exchange confirms the position is closed.

    On exchange error, returns False so callers fall through to existing
    logic — an erroneous alert is safer than silently skipping a real failure.
    Uses the passed-in exchange object directly so callers that already hold
    an authenticated session don't pay a reconnect.
    """
    try:
        positions = exchange.fetch_positions()
    except Exception as exc:
        log.warning("Position check failed for %s %s: %s — assuming open",
                    coin, direction, exc)
        return False
    for pos in positions:
        try:
            size = float(pos.get("contracts", 0) or 0)
        except (TypeError, ValueError):
            continue
        if size == 0:
            continue
        symbol = pos.get("symbol", "")
        pos_coin = symbol.split("/")[0] if symbol else ""
        side = (pos.get("side") or "")
        side_key = side.lower() if isinstance(side, str) else ""
        pos_dir = _SIDE_TO_DIRECTION.get(side_key)
        if pos_coin == coin and pos_dir == direction:
            return False
    return True


def _handle_partial_exit_fill(
    filled_order: dict, entry: dict, scan_orders: list[dict],
    symbol: str, exchange, config: ExecutionConfig, conn,
    fill_qty: float, fill_price: float | None = None,
) -> None:
    """SL or TP2 partially filled: protect residual with new SL, re-place TP2.

    Owns the exit FILLED DB write (C4#6): if the live-qty check resolves to
    zero residual this is actually a FULL close, so the FILLED mark + the
    counterpart CANCELLED marks commit in ONE transaction after the exchange
    cancel pass (no stranded counterpart on a closed position). Otherwise the
    position stays open with a replacement SL — FILLED commits standalone,
    then the existing exchange-interleaved saga (out of C4#6 scope).

    Placement order (residual path): new SL first → cancel counterpart TPs →
    new TP2. This keeps an SL active on the exchange through the transition.
    """
    oid = filled_order["order_id"]
    residual_qty = filled_order["quantity"] - fill_qty
    residual_qty = _get_live_position_qty(
        entry["coin"], entry["direction"], residual_qty, config,
    )
    if residual_qty <= 0:
        # Zero residual → actually a FULL close. Exchange-cancel pass OUTSIDE
        # the txn, then exit FILLED + counterpart CANCELLED as ONE unit — a
        # rollback leaves the close unrecorded so the position stays open and
        # the next cycle retries (no PLACED counterpart stranded).
        targets = _cancel_counterparts_on_exchange(
            entry["scan_id"], oid, symbol, config, conn,
        )
        with transaction(conn):
            update_order_status(oid, "FILLED", conn,
                                fill_price=fill_price, filled_quantity=fill_qty)
            _mark_counterparts_cancelled(targets, conn)
        return

    # Residual remains → position stays open. Commit the exit FILLED
    # standalone (a partial fill never strands: SUM(filled exit qty) < entry
    # qty so the position correctly stays in get_open_entry_orders), then the
    # replacement saga.
    update_order_status(oid, "FILLED", conn,
                        fill_price=fill_price, filled_quantity=fill_qty)
    filled_type = filled_order["order_type"]

    # Step 1: Protect the residual with a fresh SL BEFORE cancelling counterparts.
    sl_source = (
        filled_order if filled_type == "stop_loss"
        else next(
            (o for o in scan_orders if o["order_type"] == "stop_loss"
             and o["status"] in ("PLACED", "FILLED")),
            None,
        )
    )
    if sl_source:
        _place_replacement_exit_order(
            sl_source, float(sl_source["trigger_price"]), residual_qty,
            entry, symbol, exchange, config, conn,
            suffix=":partial",
        )

    # Step 2: Cancel any remaining PLACED counterpart orders (TP1, TP2, old SL).
    current_orders = get_orders_for_scan(entry["scan_id"], conn)
    for o in current_orders:
        if o["order_id"] == filled_order["order_id"]:
            continue
        if o["order_type"] not in ("tp1", "tp2"):
            continue
        if o["status"] != "PLACED":
            continue
        exch_oid = o.get("exchange_order_id")
        if exch_oid:
            cancel_order(exch_oid, symbol, config)
        update_order_status(o["order_id"], "CANCELLED", conn)
        log.info("Cancelled %s for %s (partial fill cleanup)",
                 o["order_type"], entry["scan_id"])

    # Step 3: Re-place TP2 at residual qty.
    tp2_source = (
        filled_order if filled_type == "tp2"
        else next(
            (o for o in scan_orders if o["order_type"] == "tp2"),
            None,
        )
    )
    if tp2_source and tp2_source.get("trigger_price"):
        _place_replacement_exit_order(
            tp2_source, float(tp2_source["trigger_price"]), residual_qty,
            entry, symbol, exchange, config, conn,
            suffix=":partial",
        )


def _place_replacement_exit_order(
    old_order: dict,
    new_trigger: float,
    new_qty: float,
    entry: dict,
    symbol: str,
    exchange,
    config: ExecutionConfig,
    conn,
    suffix: str = ":replaced",
) -> bool:
    """Place a new SL/TP order and supersede an old one.

    Insert-new-then-cancel-old flow. The old row is kept as a historical record
    (FILLED rows stay FILLED — partial-fill audit trail preserved). If the old
    row is PLACED we cancel it on the exchange AFTER the new order is live so
    the position is never naked. If exchange cancel fails, old row stays PLACED
    and the duplicate on exchange is reduceOnly-safe (monitor will retry on the
    next cycle).

    Returns True if the replacement order was placed on the exchange.
    """
    order_type = old_order["order_type"]
    if order_type not in ("stop_loss", "tp1", "tp2"):
        log.error("Cannot replace unsupported order_type %s for %s",
                  order_type, entry["coin"])
        return False
    if new_qty <= 0:
        log.warning("Skipping %s replacement for %s — non-positive qty %s",
                    order_type, entry["coin"], new_qty)
        return False

    direction = entry["direction"]
    exit_side = "sell" if direction == "LONG" else "buy"
    rounded_qty = float(exchange.amount_to_precision(symbol, new_qty))
    # Deterministic cloid from the old order + suffix → idempotent retries.
    cloid = make_cloid(old_order["order_id"] + suffix)

    if order_type == "stop_loss":
        params = {"stopLossPrice": new_trigger, "reduceOnly": True, "clientOrderId": cloid}
    else:
        params = {"takeProfitPrice": new_trigger, "reduceOnly": True, "clientOrderId": cloid}

    resp = None
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            resp = exchange.create_order(
                symbol, "market", exit_side, rounded_qty, new_trigger, params,
            )
            break
        except Exception as exc:
            log.error("%s replacement attempt %d/%d for %s failed: %s",
                      order_type, attempt, max_retries, entry["coin"], exc)
            if is_rate_limit_error(exc):
                log.error("Rate limited — aborting %s replacement for %s",
                          order_type, entry["coin"])
                break
            if attempt < max_retries:
                time.sleep(attempt)

    if resp is None:
        # Placement failed. Leave old row untouched — if it was PLACED the
        # position is still protected at the prior trigger/qty.
        log.error("%s replacement exhausted for %s — old order kept as-is",
                  order_type, entry["coin"])
        if order_type == "stop_loss":
            _send_urgent_sl_alert(entry["coin"], direction, config)
        else:
            _send_errors_tp_alert(entry["coin"], direction, order_type, config)
        return False

    new_exch_oid = resp.get("id", "")
    new_oid = new_order_id()
    now = datetime.now(timezone.utc).isoformat()
    save_order({
        "order_id": new_oid,
        "scan_id": old_order["scan_id"],
        "coin": old_order["coin"],
        "direction": direction,
        "order_type": order_type,
        "exchange_order_id": new_exch_oid,
        "status": "PLACED",
        "quantity": rounded_qty,
        "trigger_price": new_trigger,
        "leverage": old_order.get("leverage"),
        "exchange": old_order.get("exchange", config.exchange),
        "execution_mode": old_order.get("execution_mode", config.mode),
        "risk_mult": old_order.get("risk_mult", 1.0),
        "placed_at": now,
        "created_at": now,
    }, conn)
    log.info(
        "Replaced %s for %s: qty=%s trigger=%s (new oid=%s, supersedes %s)",
        order_type, entry["coin"], rounded_qty, new_trigger, new_oid,
        old_order["order_id"],
    )

    # Supersede the old order: only PLACED rows need exchange cancellation.
    # FILLED rows stay FILLED (historical partial-fill record). CANCELLED /
    # FAILED rows are already inactive.
    if old_order.get("status") == "PLACED":
        old_exch_oid = old_order.get("exchange_order_id")
        cancel_ok = True
        if old_exch_oid:
            cancel_ok = cancel_order(old_exch_oid, symbol, config)
        if cancel_ok:
            update_order_status(old_order["order_id"], "CANCELLED", conn)
            old_order["status"] = "CANCELLED"
        else:
            # New order is live and the exchange already cleared the old one
            # (typically "Order was never placed, already canceled, or filled").
            # reduceOnly guarantees at most one fills. No paging — the prior
            # URGENT here false-positived on every successful drift sl_move.
            log.warning(
                "Could not cancel old %s %s for %s — duplicate on exchange (reduceOnly safe)",
                order_type, old_exch_oid, entry["coin"],
            )

    return True


def _handle_tp1_fill(
    tp1_order: dict, entry: dict, scan_orders: list[dict],
    symbol: str, exchange, config: ExecutionConfig, conn,
    fill_price: float | None = None,
    fill_qty: float | None = None,
) -> None:
    """TP1 filled: place replacement SL at remaining quantity, then cancel old SL.

    Owns the TP1 FILLED DB write (C4#6): a full-size TP1 (zero remaining after
    the live-qty check — live on counter-trend single-TP trades) is a FULL
    close, so the FILLED mark + counterpart CANCELLED marks commit in ONE
    transaction after the exchange-cancel pass — never FILLED-then-strand.
    Otherwise the position stays open with a replacement SL: FILLED commits
    standalone, then the saga.

    Placement order is critical: new SL goes on the exchange FIRST so the
    position is never naked across the swap. reduceOnly guarantees at most one
    fills when the trigger hits.
    """
    oid = tp1_order["order_id"]
    tp1_qty = fill_qty if fill_qty is not None else tp1_order.get("quantity", 0)
    remaining_qty = entry.get("quantity", 0) - tp1_qty
    remaining_qty = _get_live_position_qty(
        entry["coin"], entry["direction"], remaining_qty, config,
    )

    if remaining_qty <= 0:
        # Full-size TP1 → FULL close. Exchange-cancel pass OUTSIDE the txn,
        # then TP1 FILLED + counterpart CANCELLED as ONE unit — a rollback
        # leaves the close unrecorded so the position stays open and the next
        # cycle retries (no PLACED counterpart stranded on a closed position).
        targets = _cancel_counterparts_on_exchange(
            entry["scan_id"], oid, symbol, config, conn,
        )
        with transaction(conn):
            update_order_status(oid, "FILLED", conn,
                                fill_price=fill_price, filled_quantity=fill_qty)
            _mark_counterparts_cancelled(targets, conn)
        return

    # Partial TP1 → position stays open. Commit TP1 FILLED standalone (a
    # partial never strands: SUM(filled exit qty) < entry qty), then the
    # replacement-SL saga (exchange-interleaved, out of C4#6 scope).
    update_order_status(oid, "FILLED", conn,
                        fill_price=fill_price, filled_quantity=fill_qty)

    # Find the current SL order
    sl_order = next(
        (o for o in scan_orders if o["order_type"] == "stop_loss" and o["status"] == "PLACED"),
        None,
    )
    if not sl_order:
        log.error("No active SL found for %s after TP1 fill — position unprotected", entry["coin"])
        _send_urgent_sl_alert(entry["coin"], entry["direction"], config)
        return

    _place_replacement_exit_order(
        sl_order, float(sl_order["trigger_price"]), remaining_qty,
        entry, symbol, exchange, config, conn,
        suffix=":replaced",
    )


def _attempt_exit_order_replacement(
    order: dict, entry: dict, symbol: str, exchange,
    config: ExecutionConfig, conn,
) -> None:
    """Re-place a rejected/cancelled SL order.

    TP rejections are alert-only (not auto-replaced). Guards: skip if position
    fully closed or trigger_price missing.
    """
    if order["order_type"] not in ("stop_loss", "tp1", "tp2"):
        return

    remaining_qty = _get_remaining_qty(entry, conn)
    if remaining_qty <= 0:
        log.info("Skipping %s replacement for %s — position fully closed",
                 order["order_type"], entry["coin"])
        return

    # Exchange is source of truth: DB remaining_qty can read open while the
    # position is already flat (SL fill this cycle triggers HL auto-cancel of
    # reduceOnly TPs; the TPs reach this path before the SL fill is recorded).
    # Without this check, collateral cancels surface as false TP rejection
    # alerts and wasted SL re-placements on a closed position.
    if _position_closed_on_exchange(entry["coin"], entry["direction"], exchange):
        log.info("Skipping %s replacement for %s — position closed on exchange",
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

    _place_replacement_exit_order(
        order, float(trigger_price), remaining_qty,
        entry, symbol, exchange, config, conn,
        suffix=":rejected",
    )


def _get_remaining_qty(entry: dict, conn) -> float:
    """Calculate remaining open quantity for an entry (entry qty minus filled exits).

    Clamps to 0.0 when the residual is below OPEN_QTY_EPSILON to avoid
    IEEE 754 floating-point artifacts creating ghost positions.
    """
    row = conn.execute(
        f"""SELECT COALESCE(SUM(quantity), 0) as filled_qty FROM orders
        WHERE scan_id = ? AND order_type IN {EXIT_ORDER_TYPES_SQL}
        AND status = 'FILLED'""",
        (entry["scan_id"],),
    ).fetchone()
    remaining = entry.get("quantity", 0) - float(row["filled_qty"])
    return remaining if remaining > OPEN_QTY_EPSILON else 0.0
