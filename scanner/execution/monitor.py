"""Position monitoring — sync order status from exchange, health checks."""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from scanner.config import get_exchange_meta
from scanner.db import get_connection

from .config import ExecutionConfig
from .exchange import cancel_order, get_auth_exchange, get_exchange_open_orders, get_open_positions, is_rate_limit_error
from .storage import (
    EXIT_ORDER_TYPES_SQL,
    OPEN_QTY_EPSILON,
    get_open_entry_orders,
    get_orders_for_scan,
    make_cloid,
    new_order_id,
    save_order,
    transaction,
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


def _resolve_close_fill(src: dict) -> tuple[float | None, float | None, str]:
    """Resolve (fill_price, filled_qty, status) from a ccxt order / fetch_order dict.

    fill_price: `average` first, then `price` when > 0 — a market close is NOT a
    trigger order, so `price` is a usable fill reference here (unlike
    `_extract_fill_price`'s average-only rule, which exists because TP/SL trigger
    orders report mark price in `price`). filled_qty: `None` when the field is
    absent — distinct from an explicit `0`: `status=="closed"` with filled
    absent is a full fill, with explicit `0` an exchange auto-cancel (the
    existing monitor convention at the `sync_order_status` "closed"/"canceled"
    branches). status: lowercased ("" when absent).
    """
    fill_price = _extract_fill_price(src)
    if fill_price is None:
        raw_p = src.get("price")
        if raw_p is not None and float(raw_p) > 0:
            fill_price = float(raw_p)
    filled_raw = src.get("filled")
    filled = float(filled_raw) if filled_raw is not None else None
    status = (src.get("status") or "").lower()
    return fill_price, filled, status


def sync_order_status(exec_config: ExecutionConfig, conn=None,
                      exchange_open_orders: list[dict] | None = None) -> int:
    """Poll exchange for order fills, update DB. Cancel counterpart orders on fill.

    When exchange_open_orders is provided, uses it to skip orders still open
    on exchange (avoiding per-order fetch_order calls). Only orders that
    disappeared from the open set get individually fetched for fill details.

    Returns count of orders updated.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    updated = 0
    exchange = get_auth_exchange(exec_config)
    meta = get_exchange_meta(exec_config.exchange)

    # Batch fetch: build set of exchange order IDs still open
    if exchange_open_orders is not None:
        _open_orders = exchange_open_orders
    else:
        _open_orders = exchange.fetch_open_orders()
    open_oid_set = {str(o.get("id", "")) for o in _open_orders}

    # Check all open entries for position closure
    open_entries = get_open_entry_orders(conn, exec_config.exchange, exec_config.mode)

    for entry in open_entries:
        symbol = meta["symbol_template"].format(coin=entry["coin"])
        scan_orders = get_orders_for_scan(entry["scan_id"], conn)

        for order in scan_orders:
            if order["status"] not in ("PLACED",):
                continue

            exch_oid = order.get("exchange_order_id")
            if not exch_oid:
                continue

            # Still open on exchange — skip (no API call needed)
            if exch_oid in open_oid_set:
                continue

            # Order disappeared from exchange — fetch details for fill/cancel
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
                    original_qty = float(order["quantity"])
                    is_exit = order["order_type"] in ("stop_loss", "tp1", "tp2")
                    if not is_exit:
                        # Entry (or other non-routed) fill — standalone FILLED
                        # commit, exactly as before.
                        update_order_status(
                            order["order_id"], "FILLED", conn,
                            fill_price=fill_price,
                            filled_quantity=fill_qty,
                        )
                    # Keep in-memory consistent with DB so replacement helpers
                    # see the fill state without re-querying.
                    order["status"] = "FILLED"
                    order["quantity"] = original_qty  # keep for residual math
                    updated += 1
                    log.info(
                        "%s %s %s filled at %s (qty %s)",
                        entry["coin"], order["order_type"], order["direction"], fill_price, fill_qty,
                    )

                    # Exit fill: _handle_exit_fill OWNS the FILLED DB write so
                    # the full-close route commits it atomically with the
                    # counterpart CANCELLED marks (atomic full-close boundary).
                    if is_exit:
                        _send_fill_alert(entry, order, fill_price, exec_config, scan_orders=scan_orders)
                        try:
                            _handle_exit_fill(
                                order, entry, scan_orders,
                                symbol, exchange, exec_config, conn,
                                fill_price=fill_price,
                                fill_qty=fill_qty,
                            )
                        except Exception as hexc:
                            # Close cluster rolled back. Do NOT fall to the
                            # generic per-order handler and keep iterating the
                            # now-stale scan_orders — that would commit later
                            # counterpart marks OUTSIDE the rolled-back txn
                            # (partial-write leak). Abort this scan; the close
                            # is unrecorded so the position stays open and the
                            # next cycle retries from fresh DB state.
                            log.error(
                                "Exit-fill cluster failed for %s %s (scan %s) — "
                                "rolled back; position retained, retried next "
                                "cycle: %s",
                                entry["coin"], order["order_type"],
                                entry["scan_id"], hexc,
                            )
                        break  # stale list (success OR rolled-back) — re-check next cycle

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
                        original_qty = float(order["quantity"])
                        is_exit = order["order_type"] in ("stop_loss", "tp1", "tp2")
                        if not is_exit:
                            update_order_status(
                                order["order_id"], "FILLED", conn,
                                fill_price=fill_price,
                                filled_quantity=partial_fill,
                            )
                        # Keep in-memory consistent with DB so replacement helpers
                        # see the fill state without re-querying.
                        order["status"] = "FILLED"
                        order["quantity"] = original_qty  # keep for residual math
                        updated += 1

                        if is_exit:
                            _send_fill_alert(entry, order, fill_price, exec_config, scan_orders=scan_orders)
                            try:
                                _handle_exit_fill(
                                    order, entry, scan_orders,
                                    symbol, exchange, exec_config, conn,
                                    fill_price=fill_price,
                                    fill_qty=partial_fill,
                                )
                            except Exception as hexc:
                                # See the closed-branch rationale: a rolled-back
                                # close must abort this scan, not continue
                                # iterating stale orders and leak counterpart
                                # writes outside the txn.
                                log.error(
                                    "Exit-fill cluster failed for %s %s (scan "
                                    "%s) — rolled back; position retained, "
                                    "retried next cycle: %s",
                                    entry["coin"], order["order_type"],
                                    entry["scan_id"], hexc,
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


def _send_urgent_sl_alert(coin: str, direction: str, config: ExecutionConfig) -> bool:
    """Send URGENT Discord alert when SL replacement exhausts retries.

    Retry-with-durable-log path so a Discord blip never silences the page.
    """
    from .alerts import send_urgent
    message = (f"@everyone **URGENT**: SL replacement FAILED for {coin} {direction} "
               f"after 3 attempts — position running WITHOUT stop loss")
    return send_urgent(config.discord_urgent_webhook_url, message)


def _send_urgent_close_alert(coin: str, direction: str, config: ExecutionConfig) -> bool:
    """URGENT when a timeout market-close did NOT confirm (zero/partial/
    unpriceable/retry-exhausted → `MarketCloseResult.kind` in {partial,none}).

    The protective SL is RETAINED — counterpart cancels run only on a
    confirmed full close (`kind=="full"`), so the position is NOT naked
    (distinct from `_send_urgent_sl_alert`'s "running WITHOUT stop loss"
    condition). The position/remainder is re-attempted next monitor cycle;
    reconcile from exchange truth.
    """
    from .alerts import send_urgent
    message = (f"@everyone **URGENT**: TIMEOUT close did NOT confirm for {coin} "
               f"{direction} — position retained WITH its stop loss; remainder "
               f"re-attempted next monitor cycle. Reconcile from exchange.")
    return send_urgent(config.discord_urgent_webhook_url, message)


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
    from .alerts import post_content
    return post_content(webhook, message, "TP errors alert send",
                        failure_logger=log, failure_level=logging.ERROR)


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
        # Counter-trend positions use shorter timeout
        r_mult = float(entry.get("risk_mult") or 1.0)
        timeout = (exec_config.counter_trend_timeout_hours
                   if r_mult < 1.0 - 1e-9
                   else exec_config.position_timeout_hours)
        if age_hours <= timeout:
            continue

        # Calculate remaining qty
        remaining_qty = _get_remaining_qty(entry, conn)
        if remaining_qty <= 0:
            continue

        symbol = meta["symbol_template"].format(coin=entry["coin"])
        exit_side = "sell" if entry["direction"] == "LONG" else "buy"

        result = market_close_position(
            entry, symbol, exit_side, remaining_qty, exchange, exec_config,
        )

        if result.kind == "full":
            # Fetch ORIGINAL SL trigger for R calculation. Ordering matters —
            # without it the engine returns whichever row it picks first, which
            # for drift/trail-affected scans is the moved SL and collapses risk.
            sl_row = conn.execute(
                "SELECT trigger_price FROM orders WHERE scan_id = ? AND order_type = 'stop_loss' "
                "ORDER BY created_at LIMIT 1",
                (entry["scan_id"],),
            ).fetchone()
            sl_price = float(sl_row["trigger_price"]) if sl_row and sl_row["trigger_price"] else None

            # Atomic full-close boundary: exchange-cancel pass first (outside
            # the txn), then the close row + counterpart CANCELLED marks commit
            # as ONE unit. NOTE: the market close already executed on the
            # exchange (irreversible). A DB rollback here does NOT cleanly
            # "retry" — the exchange is flat but DB shows the position open,
            # the confirmed-close payload is lost, and next-cycle recovery can
            # degrade to a synthetic manual_close at entry price. This
            # exchange-closed/DB-open window is the EXPLICITLY NAMED Phase-3b
            # reconcile residual (SPEC #25 self-healing scope: DB-cluster
            # failure with no exchange call interleaved is retryable; a
            # confirmed-exchange-close-then-DB-rollback is NOT, by design — a
            # saga/outbox journal was explicitly rejected). DB-cluster failure
            # on local WAL SQLite is rare; the reconcile pass owns truth-vs-DB.
            targets = _cancel_counterparts_on_exchange(
                entry["scan_id"], "", symbol, exec_config, conn,
            )
            with transaction(conn):
                save_order(result.payload, conn)
                _mark_counterparts_cancelled(targets, conn)
            _send_timeout_alert(entry, age_hours, remaining_qty, exec_config,
                                fill_price=result.fill_price, sl_price=sl_price)
            timed_out += 1
        else:
            # kind in {"partial","none"}: zero/partial/unpriceable/exhausted.
            # No counterpart cancel → position retained WITH its SL, not naked
            # (Fix-3 safety preserved). A partial leg is persisted truthfully
            # (the row market_close_position used to write itself); remainder
            # retried next cycle.
            if result.kind == "partial":
                save_order(result.payload, conn)
            _send_urgent_close_alert(entry["coin"], entry["direction"], exec_config)

    if own_conn:
        conn.close()

    return timed_out


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


@dataclass(frozen=True)
class MarketCloseResult:
    """Typed outcome of `market_close_position` (C4#6 — KL#9(a) market-close slice).

    `market_close_position` re-derives fill state from the exchange but no
    longer self-persists; the caller owns persistence so the close-state write
    can commit atomically with its counterpart-CANCELLED marks (the atomic
    full-close boundary — load-bearing invariant: no DB txn spans an exchange
    REST call). The three-way `kind` preserves Fix-3's safety exactly — a flat
    `payload | None` could not (a truthy partial would make callers cancel the
    SL + fire a false full-close alert; a None partial would lose the truthful
    partial row):

      - "full"    → `payload` is the close `save_order` dict; caller composes
                    the atomic close+cancel transaction. `fill_price` set.
      - "partial" → `payload` is the truthful partial leg (actual filled qty);
                    caller persists ONLY that row and RETAINS the SL/counterparts
                    (remainder retried next cycle). `fill_price` set.
      - "none"    → no row, no cancel (zero/unconfirmed/unpriceable/exhausted);
                    `payload`/`fill_price` are None.
    """

    kind: str
    payload: dict | None
    fill_price: float | None


# Shared sentinel — every non-confirmed/exception/exhausted path returns this.
_CLOSE_NONE = MarketCloseResult("none", None, None)


def market_close_position(
    entry: dict, symbol: str, exit_side: str, qty: float,
    exchange, config: ExecutionConfig,
    order_type: str = "timeout_close",
) -> MarketCloseResult:
    """Place a reduceOnly market order to close a position. 3 retries.

    Returns a `MarketCloseResult` — persistence is the CALLER's job (so the
    close row commits atomically with counterpart cancels; see the type's
    docstring). `kind="full"` only on a confirmed FULL priced close;
    `kind="partial"` carries the truthful partial leg (caller retains the SL,
    remainder retried next cycle); `kind="none"` on a zero/unconfirmed/
    unpriceable result (NO row — a `FILLED`/NULL-price row would fabricate a
    close, the 2026-04-17 phantom_sl class) and on retry exhaustion.
    `create_order` succeeding is not proof of a fill — HL has no native market
    order; ccxt simulates via limit-IOC which can under/not-fill, so fill state
    is re-derived from `status`/`filled` (T-014(a), SPEC #24). The confirmed-
    fill GATING is preserved verbatim from Fix-3 — only persistence moved out.

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
            exch_id = resp.get("id", "")
            fill_price, filled, status = _resolve_close_fill(resp)
            # `price` may merely echo the submitted IOC limit, so it is NOT
            # fill evidence unless the order is `closed`. Re-derive from the
            # exchange when the create resp is not a confirmed priced close.
            if (status != "closed" or fill_price is None) and exch_id:
                try:
                    fill_price, filled, status = _resolve_close_fill(
                        exchange.fetch_order(exch_id, symbol)
                    )
                except Exception as fexc:
                    log.warning(
                        "%s fetch_order(%s) for %s failed: %s",
                        order_type, exch_id, entry["coin"], fexc,
                    )

            # Confirmed priced close requires status closed AND a price AND a
            # non-zero fill (filled absent ≠ explicit 0 — see _resolve_close_fill).
            confirmed = status == "closed" and fill_price is not None and filled != 0
            if confirmed:
                full = filled is None or filled + OPEN_QTY_EPSILON >= rounded_qty
                recorded_qty = rounded_qty if full else filled
                # Two separate now() calls — byte-identical to the pre-recontract
                # save_order payload (only the persistence boundary moved).
                payload = {
                    "order_id": order_id,
                    "scan_id": entry["scan_id"],
                    "coin": entry["coin"],
                    "direction": entry["direction"],
                    "order_type": order_type,
                    "exchange_order_id": exch_id,
                    "status": "FILLED",
                    "fill_price": fill_price,
                    "quantity": recorded_qty,
                    "exchange": config.exchange,
                    "execution_mode": config.mode,
                    "filled_at": datetime.now(timezone.utc).isoformat(),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                if full:
                    log.info(
                        "%s %s %s: qty %s at %s",
                        order_type, entry["coin"], entry["direction"], recorded_qty, fill_price,
                    )
                    return MarketCloseResult("full", payload, fill_price)
                log.warning(
                    "%s %s %s: PARTIAL close %s/%s at %s — remainder open, "
                    "retried next monitor cycle",
                    order_type, entry["coin"], entry["direction"],
                    recorded_qty, rounded_qty, fill_price,
                )
                return MarketCloseResult("partial", payload, fill_price)

            # Not a confirmed priced close (status not closed, zero fill, or
            # unpriceable). Write NO row — a FILLED/NULL-price row fabricates a
            # close (phantom_sl class). No row to reconcile against, so the
            # durable exchange id is preserved in the log.
            log.error(
                "%s for %s NOT confirmed (status=%s filled=%s price=%s) — no row "
                "written; order_id=%s cloid=%s exchange_id=%s; reconcile from exchange",
                order_type, entry["coin"], status, filled, fill_price,
                order_id, cloid, exch_id,
            )
            return _CLOSE_NONE
        except Exception as exc:
            log.error(
                "%s attempt %d/%d for %s failed: %s",
                order_type, attempt, max_retries, entry["coin"], exc,
            )
            if attempt < max_retries:
                time.sleep(attempt)

    log.error("All %s attempts exhausted for %s — position still open", order_type, entry["coin"])
    return _CLOSE_NONE


def _calc_r(entry_price: float, exit_price: float, sl_price: float, direction: str) -> float | None:
    """Calculate R (risk multiple) for a closed position."""
    risk = abs(entry_price - sl_price)
    if risk <= 0:
        return None
    if direction == "LONG":
        return (exit_price - entry_price) / risk
    else:
        return (entry_price - exit_price) / risk


def _original_sl_price(scan_orders: list[dict] | None) -> float | None:
    """First-placed stop_loss trigger price for this scan.

    Risk anchor for R reporting. Drift `sl_move` and post-TP1 trail mutate
    the current SL; using the original keeps the R denominator stable across
    the position lifecycle: drift-flatten reports +0.00R (not None) and
    trailed runners report true R vs. initial risk.

    `get_orders_for_scan` already orders by created_at then type — the first
    stop_loss row is the original.
    """
    if not scan_orders:
        return None
    for o in scan_orders:
        if o["order_type"] == "stop_loss" and o.get("trigger_price"):
            return float(o["trigger_price"])
    return None


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

    # Calculate R value — scaled by fraction of original position this fill covers
    r_val = None
    if fill_price and entry.get("entry_price"):
        # Original (first-placed) SL is the risk anchor — drift sl_move and
        # post-TP1 trail mutate the current SL, so using the order that just
        # filled collapses risk to 0 on drift-flatten.
        sl_price = _original_sl_price(scan_orders)
        # Fallback for SL fills when scan_orders not passed (legacy callers
        # / tests). For TP fills there's no fallback — without scan_orders we
        # have no SL reference, so R stays None (existing behavior).
        if sl_price is None and order["order_type"] == "stop_loss":
            sl_price = order.get("trigger_price")
        if sl_price:
            r_val = _calc_r(float(entry["entry_price"]), fill_price, float(sl_price), entry["direction"])
            # Scale by position fraction (e.g. SL on half position after TP1 = -0.5R not -1R)
            if r_val is not None and order.get("quantity") and entry.get("quantity"):
                frac = float(order["quantity"]) / float(entry["quantity"])
                if frac < 1.0 - 1e-9:
                    r_val *= frac
            # Scale by risk_mult (counter-trend 0.5x = -0.5R not -1R)
            if r_val is not None:
                r_mult = float(entry.get("risk_mult") or 1.0)
                if r_mult < 1.0 - 1e-9:
                    r_val *= r_mult

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

    from .alerts import post_embed
    return post_embed(webhook, embed, "Fill alert", failure_logger=log)


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
            r_mult = float(entry.get("risk_mult") or 1.0)
            if r_mult < 1.0 - 1e-9:
                r_val *= r_mult
            fields.append({"name": "R Value", "value": f"{r_val:+.2f}R", "inline": True})

    embed = {
        "title": f"\u23f0 TIMEOUT — {entry['coin']} {entry['direction']}",
        "color": 0xFFA726,
        "fields": fields,
        "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
    }

    from .alerts import post_embed
    return post_embed(webhook, embed, "Timeout alert send",
                      failure_logger=log, failure_level=logging.ERROR)


def _reclassify_missing_sl(
    sl_order: dict,
    entry: dict,
    scan_orders: list[dict],
    config: ExecutionConfig,
    conn,
) -> str:
    """Re-check a PLACED SL that isn't in exchange open_orders.

    Distinguishes "phantom" (exchange cancelled the order — position still
    exists and needs a new SL) from "filled" (SL triggered and closed the
    position between the last open-orders fetch and now). Without this
    distinction, an auto-recovery placed a replacement SL that the exchange
    rejected for having no position to reduce.

    Returns 'filled', 'phantom', or 'unknown' (fetch failed → caller treats
    conservatively as phantom).
    """
    exch_oid = sl_order.get("exchange_order_id")
    if not exch_oid:
        return "phantom"
    try:
        meta = get_exchange_meta(config.exchange)
        symbol = meta["symbol_template"].format(coin=entry["coin"])
        exchange = get_auth_exchange(config)
        exch_order = exchange.fetch_order(exch_oid, symbol)
    except Exception as exc:
        log.warning(
            "reclassify missing SL: fetch_order(%s) failed — treating as phantom: %s",
            exch_oid, exc,
        )
        return "phantom"

    status = (exch_order.get("status") or "").lower()
    filled_raw = exch_order.get("filled")
    filled_qty = float(filled_raw) if filled_raw is not None else 0.0

    if status == "closed" and filled_qty > 0:
        fill_price = _extract_fill_price(exch_order)
        if fill_price is None and sl_order.get("trigger_price"):
            fill_price = float(sl_order["trigger_price"])
        log.info(
            "Detected undetected SL fill during health check: %s qty=%s at %s",
            entry["coin"], filled_qty, fill_price,
        )
        original_qty = float(sl_order.get("quantity") or 0)
        # SL (exit) fill — _handle_exit_fill OWNS the FILLED DB write so the
        # full-close route commits it atomically with the counterpart cancels.
        sl_order["status"] = "FILLED"
        sl_order["quantity"] = original_qty  # preserve for residual math
        _send_fill_alert(entry, sl_order, fill_price, config, scan_orders=scan_orders)
        _handle_exit_fill(
            sl_order, entry, scan_orders,
            symbol, exchange, config, conn,
            fill_price=fill_price,
            fill_qty=filled_qty,
        )
        return "filled"

    return "phantom"


def check_positions(exec_config: ExecutionConfig, conn=None, scanner_config=None,
                    exchange_open_orders: list[dict] | None = None) -> list[dict]:
    """Health check: verify open positions have SL orders, detect orphans.

    When exchange_open_orders is provided, uses it instead of fetching.
    Returns list of issues found.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    issues = []

    # Build exchange order ID set — use shared data or fetch our own
    if exchange_open_orders is not None:
        exchange_oids = {str(o.get("id", "")) for o in exchange_open_orders}
    else:
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
                        # The SL isn't in open_orders — could be a real phantom
                        # (exchange cancelled it) OR a fill we haven't processed
                        # yet (SL triggered between placement and this check, e.g.
                        # drift moves SL to breakeven and price immediately
                        # reverses through it). fetch_order disambiguates.
                        reclassified = _reclassify_missing_sl(
                            o, entry, scan_orders, exec_config, conn,
                        )
                        if reclassified == "filled":
                            has_sl = False  # position closed, not a phantom
                        else:  # "phantom" or "unknown" → conservative
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
            # Re-check: the reclassify step may have marked the exit FILLED +
            # cancelled counterparts, which closes the position. If so, skip.
            current = get_orders_for_scan(entry["scan_id"], conn)
            filled_exits = [
                x for x in current
                if x["order_type"] in ("stop_loss", "tp1", "tp2",
                                        "timeout_close", "emergency_close",
                                        "drift_close", "manual_close")
                and x["status"] == "FILLED"
            ]
            total_filled = sum(float(x.get("quantity") or 0) for x in filled_exits)
            if total_filled + OPEN_QTY_EPSILON >= float(entry.get("quantity") or 0):
                continue  # position closed, no SL needed

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
    """Owned recovery contract for the entry-`FILLED`→SL orphan window
    (C4#6 Chunk 3 named + injected-failure-tested invariant; SPEC #25).

    A `FILLED` entry whose SL write/placement did not durably land converges
    to **≥1 on-exchange SL** here: the monitor health check re-detects the
    unprotected position every cycle and re-drives this until an SL is live
    (cloid idempotency + reclassify-before-recovery keep the retries safe).
    No mechanism change — this docstring NAMES the already-present saga as the
    owned contract rather than an emergent one.

    For missing_sl: no PLACED SL exists; reads trigger_price from the last
    SL row (FAILED/CANCELLED) or falls back to scan machine_block.
    For phantom_sl: a PLACED SL row exists but the exchange has no matching
    open order; marks that row CANCELLED and places a fresh one.
    Returns True if SL successfully placed.

    NAMED RESIDUAL (NOT closed here): if a *random* `new_order_id` is lost on
    a crash AFTER exchange placement but BEFORE the SL row is saved, the next
    cycle mints a FRESH cloid and HL does not dedupe → a transient duplicate
    reduceOnly SL (at most one fills). Converges to ≥1 (never zero — no naked
    position); the duplicate is the KL#0/Phase-3b `reconcile_positions`
    boundary. Stable-keyed cloids (scan_id+role) would make this fully
    idempotent but are a mechanism redesign out of C4#6 scope.
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

    # Guard against the liquidation / external-close loop: if the exchange
    # has no matching position, placing a reduceOnly SL is guaranteed to
    # reject ("no position to reduce"). Rather than trigger 3-retry + urgent
    # alert + next-cycle repeat, record the closure as manual_close and bail.
    #
    # NAMED Phase-3b reconcile residual (SPEC #25): this synthetic
    # manual_close is booked at the ENTRY price. It is the correct fabrication
    # for a genuinely external close (liquidation/manual), but it is ALSO the
    # degraded outcome when a confirmed timeout_close/drift_close DB cluster
    # rolled back (exchange flat, DB still open) — the real fill price/PnL is
    # then mis-recorded as a synthetic breakeven. C4#6 owns DB-atomicity, NOT
    # exchange-truth-vs-DB reconstruction post-rollback; the reconcile pass
    # (Phase-3b) owns replaying the true close. A saga/outbox journal to make
    # this lossless was explicitly rejected by the approved design.
    try:
        from .exchange import get_open_positions
        positions = get_open_positions(config)
        exchange_has_position = any(
            p.get("coin") == coin and p.get("direction") == direction
            and float(p.get("size") or 0) > OPEN_QTY_EPSILON
            for p in positions
        )
        if not exchange_has_position:
            log.warning(
                "SL recovery: %s %s not on exchange — recording manual_close to stop recovery loop",
                coin, direction,
            )
            from .storage import new_order_id as _new_oid
            now = datetime.now(timezone.utc).isoformat()
            save_order({
                "order_id": _new_oid(),
                "scan_id": scan_id,
                "coin": coin,
                "direction": direction,
                "order_type": "manual_close",
                "status": "FILLED",
                "quantity": remaining_qty,
                "fill_price": float(entry.get("fill_price") or entry.get("entry_price") or 0),
                "exchange": config.exchange,
                "execution_mode": config.mode,
                "filled_at": now,
                "created_at": now,
            }, conn)
            return True
    except Exception as exc:
        # Don't abort recovery on a reconciliation failure — fall through to
        # the usual placement flow. Better to try placing (and potentially
        # fail) than skip when the position may actually exist.
        log.warning("SL recovery position reconciliation failed: %s", exc)

    # Prefer the most recent PLACED SL (phantom scenario) — its exch_oid is
    # stale and its trigger_price is the one we want to re-place. Fall back
    # to any SL row for trigger_price lookup.
    sl_orders = [o for o in scan_orders if o["order_type"] == "stop_loss"]
    sl_order = next(
        (o for o in reversed(sl_orders) if o["status"] == "PLACED"),
        None,
    )
    if sl_order is None:
        sl_order = next(iter(sl_orders), None)
    trigger_price = float(sl_order["trigger_price"]) if sl_order and sl_order.get("trigger_price") else None

    if trigger_price is None:
        # Fallback: recompute from scan machine_block
        try:
            import json

            from scanner.db import get_connection as get_scanner_conn
            scanner_conn = get_scanner_conn()
            row = scanner_conn.execute(
                "SELECT machine_block, stop_loss FROM scans WHERE scan_id = ?", (scan_id,),
            ).fetchone()
            scanner_conn.close()
            if row and row["machine_block"]:
                # T-016 C3: normalize the stored block through the single
                # canonical boundary before the now-typed reanchor helper
                # (preserves the single-boundary invariant; from_raw -> None on
                # a degraded stored block falls through exactly as the prior
                # missing-stop_loss guard did).
                from scanner.models import MachineBlock
                mb = MachineBlock.from_raw(json.loads(row["machine_block"]))
                fill_price = float(entry.get("fill_price") or entry.get("entry_price") or 0)
                if fill_price and mb is not None and mb.stop_loss:
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
            # Supersede the old SL row (PLACED phantom or FAILED).
            if sl_order and sl_order["status"] in ("FAILED", "PLACED"):
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


def send_open_positions_report(exec_config: ExecutionConfig, conn=None) -> bool:
    """Send daily Discord embed showing open positions and time remaining until timeout."""
    from scanner.config import ScannerConfig
    cfg = ScannerConfig()
    webhook = cfg.discord_reports_webhook_url or exec_config.discord_ops_webhook_url
    if not webhook:
        return False

    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    open_entries = get_open_entry_orders(conn, exec_config.exchange, exec_config.mode)
    now = datetime.now(timezone.utc)
    timeout_hours = exec_config.position_timeout_hours

    if not open_entries:
        embed = {
            "title": "\U0001f4cb Daily Positions — No Open Positions",
            "color": 0x999999,
            "footer": {"text": now.strftime("%Y-%m-%d %H:%M UTC")},
        }
    else:
        lines = []
        for entry in open_entries:
            entry_time_str = entry.get("filled_at") or entry.get("placed_at") or entry.get("created_at")
            if entry_time_str:
                try:
                    entry_time = datetime.fromisoformat(entry_time_str)
                    if entry_time.tzinfo is None:
                        entry_time = entry_time.replace(tzinfo=timezone.utc)
                    age_hours = (now - entry_time).total_seconds() / 3600
                    remaining = max(0, timeout_hours - age_hours)
                    remaining_str = f"{remaining:.1f}h left" if remaining > 0 else "EXPIRED"
                except (ValueError, TypeError):
                    age_hours = 0
                    remaining_str = "unknown"
            else:
                age_hours = 0
                remaining_str = "unknown"

            lines.append(
                f"**{entry['coin']}** {entry['direction']} — "
                f"{age_hours:.1f}h open, {remaining_str}"
            )

        embed = {
            "title": f"\U0001f4cb Daily Positions — {len(open_entries)} Open",
            "description": "\n".join(lines),
            "color": 0x42A5F5,
            "footer": {"text": f"Timeout: {timeout_hours}h | {now.strftime('%Y-%m-%d %H:%M UTC')}"},
        }

    if own_conn:
        conn.close()

    from .alerts import post_embed
    return post_embed(webhook, embed, "Open positions report",
                      failure_logger=log, failure_level=logging.ERROR)
