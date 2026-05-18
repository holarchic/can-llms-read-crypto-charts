"""Position monitoring — sync order status from exchange, health checks."""

import logging
import time
from datetime import datetime, timezone

from scanner.config import get_exchange_meta
from scanner.db import get_connection

from .config import ExecutionConfig
from .exchange import get_auth_exchange, get_exchange_open_orders, is_rate_limit_error

# T-016 Phase 4a (SPEC #27): exit-order lifecycle cluster extracted to
# exit_fills.py. Re-exported so coordinators (sync_order_status calls
# _handle_exit_fill/_attempt_exit_order_replacement; enforce_position_timeouts
# calls _get_remaining_qty/_cancel_counterparts_on_exchange/
# _mark_counterparts_cancelled), drift.py, and tests resolve through the
# monitor namespace. One-directional: monitor -> exit_fills -> monitor_alerts.
from .exit_fills import (  # noqa: F401  (re-export surface)
    _attempt_exit_order_replacement,
    _cancel_counterparts_on_exchange,
    _get_live_position_qty,
    _get_remaining_qty,
    _handle_exit_fill,
    _handle_partial_exit_fill,
    _handle_tp1_fill,
    _mark_counterparts_cancelled,
    _place_replacement_exit_order,
    _position_closed_on_exchange,
)

# T-016 Phase 4a (SPEC #27): alert-composition + R math extracted to
# monitor_alerts.py. Re-exported here so coordinators (this module), drift.py,
# __main__.py, and tests resolve these symbols through the monitor namespace
# byte-stably (single-owner lives in monitor_alerts; this is a pure re-export,
# not a second definition). One-directional: monitor -> monitor_alerts.
from .monitor_alerts import (  # noqa: F401  (re-export surface)
    _calc_r,
    _send_errors_tp_alert,
    _send_fill_alert,
    _send_timeout_alert,
    _send_urgent_close_alert,
    _send_urgent_sl_alert,
)

# T-016 Phase 4a (SPEC #27): market-close + the typed close-result single
# owner (CloseResult, renamed from MarketCloseResult — KL#9(a)) extracted to
# position_close.py. Re-exported so coordinators (enforce_position_timeouts,
# sync_order_status, _reclassify_missing_sl), drift.py, and tests resolve
# through the monitor namespace. One-directional: monitor -> position_close.
from .position_close import (  # noqa: F401  (re-export surface)
    CloseResult,
    _extract_fill_price,
    _resolve_close_fill,
    market_close_position,
)
from .storage import (
    OPEN_QTY_EPSILON,
    get_open_entry_orders,
    get_orders_for_scan,
    make_cloid,
    new_order_id,
    original_stop_price,
    save_order,
    transaction,
    update_order_status,
)

log = logging.getLogger(__name__)


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
                    log.info(
                        "%s %s %s filled at %s (qty %s)",
                        entry["coin"], order["order_type"], order["direction"], fill_price, fill_qty,
                    )

                    # Exit fill: _handle_exit_fill OWNS the FILLED DB write so
                    # the full-close route commits it atomically with the
                    # counterpart CANCELLED marks (atomic full-close boundary).
                    # The fill-alert + `updated` count fire only AFTER the
                    # cluster commits: on a rollback the close is unrecorded and
                    # the next cycle re-detects the fill, so emitting them first
                    # would double-alert + over-count (the C4#6-deferred Phase-4
                    # LOWs). Non-exit fills persisted unconditionally above, so
                    # they count immediately.
                    if not is_exit:
                        updated += 1
                    else:
                        try:
                            _handle_exit_fill(
                                order, entry, scan_orders,
                                symbol, exchange, exec_config, conn,
                                fill_price=fill_price,
                                fill_qty=fill_qty,
                            )
                            _send_fill_alert(entry, order, fill_price, exec_config, conn)
                            updated += 1
                        except Exception as hexc:
                            # Close cluster rolled back. Do NOT fall to the
                            # generic per-order handler and keep iterating the
                            # now-stale scan_orders — that would commit later
                            # counterpart marks OUTSIDE the rolled-back txn
                            # (partial-write leak). Abort this scan; the close
                            # is unrecorded so the position stays open and the
                            # next cycle retries from fresh DB state. No
                            # fill-alert, no `updated` (the next cycle emits
                            # exactly one of each on the successful close).
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

                        # Same fill-alert/`updated`-after-commit ordering as
                        # the closed branch (C4#6-deferred Phase-4 LOWs):
                        # non-exit persisted above → counts now; exit counts +
                        # alerts only after the cluster commits.
                        if not is_exit:
                            updated += 1
                        else:
                            try:
                                _handle_exit_fill(
                                    order, entry, scan_orders,
                                    symbol, exchange, exec_config, conn,
                                    fill_price=fill_price,
                                    fill_qty=partial_fill,
                                )
                                _send_fill_alert(entry, order, fill_price, exec_config, conn)
                                updated += 1
                            except Exception as hexc:
                                # See the closed-branch rationale: a rolled-back
                                # close must abort this scan, not continue
                                # iterating stale orders and leak counterpart
                                # writes outside the txn. No alert, no count —
                                # the next cycle emits exactly one of each.
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
            # Original-SL risk anchor — single owner (KL#2(iv), SPEC #27): the
            # Phase-2a storage.original_stop_price (ORDER BY created_at ASC
            # LIMIT 1, mode+exchange-scoped). `or None` preserves the prior
            # `... if sl_row and sl_row["trigger_price"] else None` falsy→None
            # coercion byte-identically (storage returns 0.0 for a trigger==0
            # row; `0.0 or None` → None).
            sl_price = original_stop_price(
                entry["scan_id"], conn, exec_config.mode, exec_config.exchange,
            ) or None

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
        # Fill-alert only AFTER the close cluster commits (same invariant as
        # the C-LOW sync_order_status fix, SPEC #27): there is no rollback
        # catch here — if _handle_exit_fill raises, the close is unrecorded
        # and the next health-check cycle re-detects the SL fill, so alerting
        # first would double-alert on the rare rollback.
        _handle_exit_fill(
            sl_order, entry, scan_orders,
            symbol, exchange, config, conn,
            fill_price=fill_price,
            fill_qty=filled_qty,
        )
        _send_fill_alert(entry, sl_order, fill_price, config, conn)
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
        # strict=True: a lossy snapshot (unparseable live position) must NOT be
        # read as "flat" and book a false synthetic manual_close. PositionParseError
        # is caught below → fall through to the usual SL-placement flow (no false
        # close, SL still placed → never naked). Phase-3b / SPEC #29 end-to-end
        # strictness: every close-on-position-absent path is fail-closed.
        positions = get_open_positions(config, strict=True)
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
