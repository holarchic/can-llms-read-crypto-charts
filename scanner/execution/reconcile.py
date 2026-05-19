"""Proactive exchange-truth-vs-DB reconciliation — Phase-3b (KL#0, SPEC #29).

A dedicated `_cmd_monitor` step, run before the SL-path health check and
independent of it, that closes two C4#6 crash-residual windows:

  (a) exchange-closed / DB-open  → book a best-effort synthetic manual_close
  (c) stranded TRIGGERED pending → resolve TRIGGER_FAILED

THE load-bearing safety invariant: the position snapshot is read ONCE, FIRST,
via `get_open_positions(strict=True)`; on ANY exception the whole pass returns
0 and mutates nothing. Booking a close / resolving a pending from an absent OR
lossy snapshot would act on a position that may actually be open — the single
catastrophic failure mode. Idempotent: a booked manual_close drops the entry
out of `get_open_entry_orders` (remaining-qty math) and a resolved pending
leaves the TRIGGERED set, so a second run is a no-op. No DB transaction spans
an exchange REST call (#25): one up-front read, then independent
`_maybe_commit` writes only.

Window (a) is fail-closed-safe but not corrective for ≥2 DB-open rows sharing
(coin,direction) against one HL net position (named accepted residual — per-
order disambiguation is window (b), deferred). Window (c) is capital-safe but
not provenance-complete (a market entry that filled but whose entry-row save
was lost, then closed, is mislabeled TRIGGER_FAILED — no live position, so
nothing at risk; true provenance needs the rejected saga).
"""

import logging
from datetime import datetime, timezone

from .exit_fills import (
    _get_remaining_qty,
    book_external_close,
    has_open_exchange_position,
)
from .storage import (
    get_open_entry_orders,
    get_orders_for_scan,
    get_triggered_pending_entries,
    update_pending_status,
)

log = logging.getLogger(__name__)

# A TRIGGERED pending with no resulting trade (no FILLED entry, no exchange
# position) older than this is unambiguously stranded: a triggered entry is a
# market order and execute_trade completes in seconds, so a same-cycle
# just-claimed pending is gated out and never false-flipped.
RECONCILE_STRANDED_MIN_AGE_S = 3600


def _book_missed_closes(positions: list[dict], conn, exchange: str, mode: str) -> int:
    """Window (a): a DB-open entry with no matching exchange position is
    closed-on-exchange / DB-open → book a best-effort synthetic manual_close
    via the single-owner `book_external_close` (same decision the reactive
    SL-recovery path delegates to; KL#0 "surface sooner", #25 rejected the
    lossless true-PnL saga). Quantity is the single-owner remaining qty (not
    the raw entry qty) so a partially-exited position books the truthful
    residual and stays idempotent."""
    booked = 0
    for e in get_open_entry_orders(conn, exchange, mode):
        if has_open_exchange_position(positions, e["coin"], e["direction"]):
            continue
        remaining = _get_remaining_qty(e, conn)
        if remaining <= 0:
            continue  # already flat in DB (defensive; get_open_entry_orders filters)
        book_external_close(e, remaining, conn, exchange=exchange, mode=mode)
        log.warning(
            "reconcile: %s %s DB-open but flat on exchange — booked manual_close "
            "qty=%s (best-effort entry price; SPEC #29 window a)",
            e["coin"], e["direction"], remaining,
        )
        booked += 1
    return booked


def _resolve_stranded_pendings(positions: list[dict], conn, exchange: str, mode: str) -> int:
    """Window (c): a TRIGGERED pending with no resulting trade is stranded
    (cross-process crash after claim_pending ACTIVE→TRIGGERED, before the
    in-process resolution). Resolve it TRIGGER_FAILED. Three conjunctive
    conditions — a false-flip of a legitimately-triggered/in-flight pending
    is the danger, so every condition must hold."""
    now = datetime.now(timezone.utc)
    resolved = 0
    for p in get_triggered_pending_entries(conn, exchange, mode):
        coin, direction = p["coin"], p["direction"]

        # 1. no FILLED entry order for this scan (a placed trade ⇒ legit).
        scan_orders = get_orders_for_scan(p["scan_id"], conn)
        if any(o["order_type"] == "entry" and o["status"] == "FILLED" for o in scan_orders):
            continue
        # 2. no live exchange position (an untracked position is window (a)/
        #    check_positions' concern, NOT a stranded pending).
        if has_open_exchange_position(positions, coin, direction):
            continue
        # 3. age-gated: only flip a TRIGGERED pending whose claim is old enough
        #    that an in-flight execute_trade cannot still be running. A missing/
        #    unparseable resolved_at is treated as NOT stranded (conservative —
        #    never false-flip on bad data).
        resolved_at = p.get("resolved_at")
        if not resolved_at:
            continue
        try:
            claimed = datetime.fromisoformat(resolved_at)
        except (TypeError, ValueError):
            log.warning(
                "reconcile: pending %s has unparseable resolved_at=%r — "
                "skipping (conservative)", p["pending_id"], resolved_at,
            )
            continue
        # claim_pending writes a tz-aware UTC isoformat; a tz-naive value is
        # anomalous (manual edit / legacy / foreign writer). Treat it as bad
        # data and skip — `aware - naive` would raise TypeError and a parse
        # quirk must NEVER abort the monitor cycle or false-flip a pending.
        if claimed.tzinfo is None:
            log.warning(
                "reconcile: pending %s resolved_at=%r is tz-naive — "
                "skipping (conservative)", p["pending_id"], resolved_at,
            )
            continue
        if (now - claimed).total_seconds() < RECONCILE_STRANDED_MIN_AGE_S:
            continue

        update_pending_status(p["pending_id"], "TRIGGER_FAILED", conn)
        log.warning(
            "reconcile: pending %s (%s %s) stranded TRIGGERED with no trade — "
            "resolved TRIGGER_FAILED (SPEC #29 window c)",
            p["pending_id"], coin, direction,
        )
        resolved += 1
    return resolved


def reconcile_positions(config, conn) -> int:
    """Proactive exchange-truth-vs-DB reconcile. Returns the number of
    reconciled items (missed closes booked + stranded pendings resolved).

    Fail-closed: the strict position snapshot is taken first; on any failure
    (raised read OR a successful-but-lossy snapshot, which the strict snapshot
    turns into a raise) the pass logs and returns 0 having mutated nothing.
    `Exception` only — `KeyboardInterrupt`/`SystemExit` must propagate (a
    shutdown signal is not a "skip this cycle").
    """
    from .exchange import get_open_positions_strict

    try:
        positions = get_open_positions_strict(config)
    except Exception as exc:
        log.warning(
            "reconcile: strict position snapshot unavailable (%s: %s) — "
            "skipping this cycle, no DB mutation", type(exc).__name__, exc,
        )
        return 0

    exchange, mode = config.exchange, config.mode
    n = _book_missed_closes(positions, conn, exchange, mode)
    n += _resolve_stranded_pendings(positions, conn, exchange, mode)
    return n
