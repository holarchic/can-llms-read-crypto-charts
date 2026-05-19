"""Proactive exchange-truth-vs-DB reconciliation — Phase-3b (KL#0, SPEC #29 + #30).

A dedicated `_cmd_monitor` step, run before the SL-path health check and
independent of it, that closes all three C4#6 crash-residual windows:

  (a) exchange-closed / DB-open  → book a best-effort synthetic manual_close
  (c) stranded TRIGGERED pending → resolve TRIGGER_FAILED
  (b) duplicate reduceOnly SL    → cancel untracked extras, converge to ≥1
                                    tracked keeper (SPEC #30; the only window
                                    that cancels live exchange orders)

THE load-bearing safety invariant: the strict position snapshot AND the
open-orders snapshot are read ONCE, FIRST, together; on ANY exception the whole
pass returns 0 and mutates nothing. Booking a close / resolving a pending /
cancelling an SL from an absent OR lossy snapshot would act on a position that
may actually be open — the single catastrophic failure mode. Window (b)
additionally never cancels unless ≥1 tracked keeper SL is confirmed live in
the snapshot (keeper-absent ⇒ zero cancels; never-naked by construction).
Idempotent: a booked manual_close drops the entry out of
`get_open_entry_orders`, a resolved pending leaves the TRIGGERED set, a
cancelled duplicate leaves the next snapshot — so a second run is a no-op. No
DB transaction spans an exchange REST call (#25): one up-front read, then
independent `_maybe_commit` writes (a/c) or exchange-only cancels (b, DB-free).

Window (a) is fail-closed-safe but not corrective for ≥2 DB-open rows sharing
(coin,direction) against one HL net position (named accepted residual — per-
*scan* disambiguation is out of reach). Window (c) is capital-safe but not
provenance-complete (a market entry that filled but whose entry-row save was
lost, then closed, is mislabeled TRIGGER_FAILED — no live position, so nothing
at risk; true provenance needs the rejected saga). Window (b) is net-grouped
and identity-keyed (OID ∪ cloid-over-the-whole-chain); an operator-hand-placed
SL on a bot-tracked position would be cancelled as an untracked extra (named
accepted residual — the runbook does not hand-place SLs on tracked positions).
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
    make_cloid,
    update_pending_status,
)

log = logging.getLogger(__name__)

# Every cloid suffix the codebase can attach to a *resting* reduceOnly
# stop-loss order. A replacement SL's live clientOrderId is
# `make_cloid(PREDECESSOR_row.order_id + suffix)` while its saved row carries a
# fresh random order_id — so keeper identity must apply every suffix to every
# stop_loss row of the scan chain, never the current row alone (SPEC #30).
# Frozen, grep-pinned to each emit site:
#   ""         base / recovery placement  (executor.py:488, monitor.py:710)
#   ":replaced" _place_replacement_exit_order default (exit_fills.py:294,466)
#   ":partial"  partial-exit replacement   (exit_fills.py:250,281)
#   ":rejected" rejected-replacement retry (exit_fills.py:516)
#   ":drift"    _move_sl_to_price default  (drift.py:235)
#   ":trail"    post-TP1 trailing SL move  (drift.py:421)
# emergency_close / market_close make_cloid sites are FILLED market closes,
# NOT resting SL keepers — deliberately excluded. A new SL-replacement suffix
# is a one-line, reviewed addition here.
_SL_CLOID_SUFFIXES = ("", ":replaced", ":partial", ":rejected", ":drift", ":trail")

# A TRIGGERED pending with no resulting trade (no FILLED entry, no exchange
# position) older than this is unambiguously stranded: a triggered entry is a
# market order and execute_trade completes in seconds, so a same-cycle
# just-claimed pending is gated out and never false-flipped.
RECONCILE_STRANDED_MIN_AGE_S = 3600


def _keeper_ids(stop_loss_rows: list[dict]) -> set[str]:
    """Every exchange identifier (oid OR clientOrderId) that a tracked SL of
    this scan chain could be live under. Window (b) cancels only SLs matching
    NONE of these. Union over EVERY stop_loss row (any status — decision #13
    preserves the whole chain) of: the row's `exchange_order_id` when present,
    plus `make_cloid(order_id + s)` for every `s` in `_SL_CLOID_SUFFIXES` — a
    replacement's live cloid is predecessor-derived, so the suffixes must be
    applied to every chain row, not the current row alone (SPEC #30)."""
    ids: set[str] = set()
    for row in stop_loss_rows:
        oid = row.get("exchange_order_id")
        if oid:
            ids.add(str(oid))
        order_id = row.get("order_id")
        if order_id:
            for suffix in _SL_CLOID_SUFFIXES:
                ids.add(make_cloid(order_id + suffix))
    return ids


def _is_reduce_only_sl(open_order: dict, coin: str, direction: str) -> bool:
    """Classify a fetched ccxt-HL open order as a reduceOnly STOP-LOSS for the
    given position, grounded in the real `frontendOpenOrders` parse shape
    (ccxt 4.5.50): unified `reduceOnly` + raw `info.{isTrigger,
    triggerCondition}` — fetched orders carry NO `stopLossPrice`/`isStopLoss`
    (those are creation params; SPEC #30). SL-vs-TP is
    the trigger-direction vs position-direction relationship: a reduceOnly
    exit-side trigger is a stop-loss iff LONG→price-below or SHORT→price-above
    (the opposite is a take-profit). `"N/A"`/unparseable triggerCondition ⇒
    NOT classified as SL ⇒ never cancelled (conservative)."""
    if not open_order.get("reduceOnly"):
        return False
    info = open_order.get("info") or {}
    if not info.get("isTrigger"):
        return False
    symbol = open_order.get("symbol") or ""
    if symbol.split("/")[0] != coin:
        return False
    exit_side = "sell" if direction == "LONG" else "buy"
    if (open_order.get("side") or "").lower() != exit_side:
        return False
    condition = str(info.get("triggerCondition") or "").lower()
    if "below" in condition:
        return direction == "LONG"
    if "above" in condition:
        return direction == "SHORT"
    return False  # "N/A" / unparseable → conservative: not an SL


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


def _order_ids(oo: dict) -> set[str]:
    """The non-empty exchange identifiers a fetched order can be matched on
    (oid + clientOrderId). HL trigger orders may carry either or both."""
    return {str(oo.get("id") or ""), str(oo.get("clientOrderId") or "")} - {""}


def _converge_duplicate_stop_losses(
    open_orders: list[dict], conn, exchange: str, mode: str, config,
) -> int:
    """Window (b): cancel UNTRACKED live reduceOnly stop-loss orders per net
    `(coin,direction)` position down to the tracked keeper(s) (SPEC #30,
    C4#6 residual b). The lost-random-`order_id` crash mints a fresh cloid on
    recovery → HL does not dedupe → ≥2 live reduceOnly SLs; this converges to
    ≥1, never 0.

    Net-grouped (HL is net-position) so scaled-in scans don't drive a double
    cancel. **Never-naked by construction:** a net group is acted on ONLY iff
    **EVERY** DB-open scan in the group has ≥1 of its OWN keeper ids live in
    the snapshot. A group-wide "any keeper live" gate is unsafe: with two
    DB-open scans on one net position, scan A's live keeper would satisfy a
    group gate while scan B's only live SL
    (no DB chain — the exact lost-`order_id` crash this window exists for) is
    cloid-unmatched and gets cancelled → B's share transiently naked. The
    per-scan-complete gate means an extra is cancelled only once every scan's
    own protection is confirmed live, so the cancelled order can never be some
    scan's sole SL. A mixed group (some scan has no live keeper) is **skipped
    whole** — fail-closed-safe, not corrective: per-*scan* SL attribution
    against one HL net position is the named accepted residual, mirroring
    window (a)'s multi-row disposition; `attempt_sl_recovery`/`check_positions`
    owns re-establishing the missing scan's SL. Keeper-absent ⇒ zero cancels.
    DB-mutation-free: the orphan has no DB row, the keeper row is untouched
    (#25 holds trivially)."""
    from .exchange import cancel_order

    groups: dict[tuple[str, str], list[str]] = {}
    for e in get_open_entry_orders(conn, exchange, mode):
        groups.setdefault((e["coin"], e["direction"]), []).append(e["scan_id"])

    cancelled = 0
    for (coin, direction), scan_ids in groups.items():
        scan_keepers = {
            scan_id: _keeper_ids([
                o for o in get_orders_for_scan(scan_id, conn)
                if o["order_type"] == "stop_loss"
            ])
            for scan_id in scan_ids
        }
        keeper_ids: set[str] = set().union(*scan_keepers.values())

        # Every scan's OWN keeper must be live before any cancel — else a
        # scan whose SL has no DB chain would lose its only live SL to
        # another scan's keeper proving the group.
        if not all(
            any(_order_ids(oo) & ks for oo in open_orders)
            for ks in scan_keepers.values()
        ):
            continue  # mixed/absent keeper → skip whole group (fail-closed)

        for oo in open_orders:
            if not _is_reduce_only_sl(oo, coin, direction):
                continue
            if _order_ids(oo) & keeper_ids:
                continue  # this IS a tracked keeper
            if cancel_order(str(oo.get("id") or ""), oo.get("symbol"), config):
                cancelled += 1
                log.warning(
                    "reconcile: cancelled untracked duplicate SL for %s %s "
                    "(id=%s cloid=%s) — converged to tracked keeper "
                    "(SPEC #30 window b)",
                    coin, direction, oo.get("id"), oo.get("clientOrderId"),
                )
    return cancelled


def reconcile_positions(config, conn) -> int:
    """Proactive exchange-truth-vs-DB reconcile. Returns the number of
    reconciled items (missed closes booked + stranded pendings resolved +
    duplicate SLs cancelled).

    Fail-closed: the strict position snapshot AND the open-orders snapshot are
    taken first, together; on any failure (raised read OR a successful-but-
    lossy position snapshot, which the strict snapshot turns into a raise) the
    pass logs and returns 0 having mutated nothing. `Exception` only —
    `KeyboardInterrupt`/`SystemExit` must propagate (a shutdown signal is not a
    "skip this cycle").
    """
    from .exchange import get_exchange_open_orders, get_open_positions_strict

    try:
        positions = get_open_positions_strict(config)
        open_orders = get_exchange_open_orders(config)
    except Exception as exc:
        log.warning(
            "reconcile: exchange snapshot unavailable (%s: %s) — "
            "skipping this cycle, no DB mutation", type(exc).__name__, exc,
        )
        return 0

    exchange, mode = config.exchange, config.mode
    n = _book_missed_closes(positions, conn, exchange, mode)
    n += _resolve_stranded_pendings(positions, conn, exchange, mode)
    n += _converge_duplicate_stop_losses(open_orders, conn, exchange, mode, config)
    return n
