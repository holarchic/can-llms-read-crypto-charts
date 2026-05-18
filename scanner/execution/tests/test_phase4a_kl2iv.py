"""T-016 Phase 4a — KL#2(iv) single-owner re-point gate (SPEC #27).

`monitor._original_sl_price` is DELETED; `_send_fill_alert` and the
`enforce_position_timeouts` inline-SQL original-SL lookup are re-pointed at
the Phase-2a single owner `storage.original_stop_price`
(`ORDER BY created_at ASC LIMIT 1`, mode+exchange-scoped).

By-construction (deterministic, in-memory) — the LESSONS "Green ≠ verified"
anti-vacuity gate: the new typed lookup must *drive* the original-SL anchor,
not be vacuously present. `_frozen_old_original_sl` is the verbatim deleted
algorithm as a frozen oracle. The earliest-`NULL`/`0`-trigger-with-later-
usable fixtures encode the SPEC #27 named **off-corpus latent-correctness
divergence** (prod-vacuous per SPEC #22/KL#2(v): zero NULL-or-zero-trigger
SL rows in the prod corpus; the single-owner semantic is the deliberate
canonical anchor — a degenerate earliest SL ≡ no usable anchor, SPEC #24).
"""

import sqlite3
from unittest.mock import MagicMock, patch

from scanner.execution.config import ExecutionConfig
from scanner.execution.monitor_alerts import _send_fill_alert
from scanner.execution.storage import (
    ensure_orders_table,
    get_orders_for_scan,
    original_stop_price,
)

_CFG = ExecutionConfig(discord_ops_webhook_url="https://discord.com/ops")  # testnet/hyperliquid defaults


def _conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_orders_table(c)
    return c


def _ins(conn, scan_id, otype, *, trig=None, created, status="PLACED", qty=1.0):
    """Raw insert — `created` controls created_at independently of rowid so
    the adversarial insertion-order≠created_at case is by construction."""
    conn.execute(
        """INSERT INTO orders (order_id, scan_id, coin, direction, order_type,
        status, trigger_price, entry_price, quantity, exchange,
        execution_mode, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (f"{scan_id}_{otype}_{created}_{trig}", scan_id, "ABC", "LONG", otype,
         status, trig, 100.0, qty, "hyperliquid", "testnet", created),
    )
    conn.commit()


def _frozen_old_original_sl(scan_orders):
    """Verbatim of the DELETED monitor._original_sl_price (frozen oracle)."""
    if not scan_orders:
        return None
    for o in scan_orders:
        if o["order_type"] == "stop_loss" and o.get("trigger_price"):
            return float(o["trigger_price"])
    return None


def _old(conn, scan_id):
    return _frozen_old_original_sl(get_orders_for_scan(scan_id, conn))


def _new(conn, scan_id):
    return original_stop_price(scan_id, conn, "testnet", "hyperliquid") or None


# --- prod-shaped equivalence (old frozen == new), non-vacuous ---------------

def test_single_sl_equivalent_and_nonvacuous():
    conn = _conn()
    _ins(conn, "s1", "entry", created="2026-01-01T00:00:00+00:00")
    _ins(conn, "s1", "stop_loss", trig=95.0, created="2026-01-01T00:00:01+00:00")
    assert _old(conn, "s1") == _new(conn, "s1") == 95.0  # anti-vacuity: a real value


def test_multi_sl_created_at_order_both_pick_original():
    """Original SL (earlier created_at) then a drift-moved SL. Both old and
    new pick the ORIGINAL — the stable risk anchor across the lifecycle."""
    conn = _conn()
    _ins(conn, "s1", "entry", created="2026-01-01T00:00:00+00:00")
    _ins(conn, "s1", "stop_loss", trig=100.0, status="CANCELLED",
         created="2026-01-01T00:00:01+00:00")          # ORIGINAL
    _ins(conn, "s1", "stop_loss", trig=95.0,
         created="2026-01-02T00:00:00+00:00")           # drift-moved
    assert _old(conn, "s1") == _new(conn, "s1") == 100.0
    assert _new(conn, "s1") != 95.0  # NOT the drifted SL


def test_adversarial_insertion_order_ne_created_at():
    """By construction: the drift SL is inserted FIRST (lower rowid) but has a
    LATER created_at; the original is inserted second (earlier created_at). A
    bare `LIMIT 1` (undefined/rowid order) would pick the drift 95.0; both the
    frozen-old (get_orders_for_scan ORDER BY created_at) and the new
    (original_stop_price ORDER BY created_at ASC) pick the ORIGINAL 100.0."""
    conn = _conn()
    _ins(conn, "s1", "entry", created="2026-01-01T00:00:00+00:00")
    _ins(conn, "s1", "stop_loss", trig=95.0,
         created="2026-01-02T00:00:00+00:00")           # drift — inserted first
    _ins(conn, "s1", "stop_loss", trig=100.0, status="CANCELLED",
         created="2026-01-01T00:00:01+00:00")           # ORIGINAL — inserted 2nd
    # A bare `LIMIT 1` with no ORDER BY is SQLite-undefined order (the exact
    # KL#2 hazard) — its value is nondeterministic, so we do NOT assert it.
    # The invariant: both the frozen-old (get_orders_for_scan ORDER BY
    # created_at) and the new (original_stop_price ORDER BY created_at ASC)
    # are created_at-driven and pick the ORIGINAL despite insertion order.
    assert _old(conn, "s1") == _new(conn, "s1") == 100.0
    assert _new(conn, "s1") != 95.0           # never the drift-moved SL


# --- SPEC #27 named off-corpus latent-correctness divergence ---------------
# (prod-vacuous: SPEC #22/KL#2(v) — zero NULL-or-zero-trigger SL rows in prod)

def test_earliest_NULL_trigger_with_later_usable_named_divergence():
    """Earliest-by-created_at SL has a NULL trigger; a later SL is usable.
    frozen-old skips the NULL and scans on to the later real SL; the single
    owner returns the earliest (NULL → None). Documented latent-correctness:
    a degenerate earliest SL ≡ no usable anchor (SPEC #24/KL#2(v))."""
    conn = _conn()
    _ins(conn, "s1", "entry", created="2026-01-01T00:00:00+00:00")
    _ins(conn, "s1", "stop_loss", trig=None,
         created="2026-01-01T00:00:01+00:00")           # earliest, NULL
    _ins(conn, "s1", "stop_loss", trig=88.0,
         created="2026-01-02T00:00:00+00:00")           # later, usable
    assert _old(conn, "s1") == 88.0           # old: skips NULL → later real
    assert _new(conn, "s1") is None           # new: earliest (NULL) → no anchor
    assert _old(conn, "s1") != _new(conn, "s1")  # the named divergence


def test_earliest_ZERO_trigger_with_later_usable_named_divergence():
    """Same divergence class with trigger==0 (storage returns 0.0; the
    call-site `or None` coercion → None — byte-identical to the old
    `... else None` for the *returned* value)."""
    conn = _conn()
    _ins(conn, "s1", "entry", created="2026-01-01T00:00:00+00:00")
    _ins(conn, "s1", "stop_loss", trig=0.0,
         created="2026-01-01T00:00:01+00:00")           # earliest, 0
    _ins(conn, "s1", "stop_loss", trig=88.0,
         created="2026-01-02T00:00:00+00:00")           # later, usable
    assert _old(conn, "s1") == 88.0           # old: skips 0 (falsy) → later
    assert original_stop_price("s1", conn, "testnet", "hyperliquid") == 0.0
    assert _new(conn, "s1") is None           # `0.0 or None` → None


# --- the SL-fill fallback is preserved verbatim ----------------------------

@patch("scanner.execution.alerts.requests.post")
def test_sl_fill_fallback_preserved_when_no_original_row(mock_post):
    """SL fill, NO original-SL row in the DB → new anchor None → the L78-79
    SL-fill fallback (use the just-filled SL's own trigger) still fires."""
    mock_post.return_value = MagicMock(status_code=204)
    conn = _conn()  # empty — no original SL row for this scan
    entry = {"scan_id": "s_none", "coin": "BTC", "direction": "LONG", "entry_price": 62000}
    order = {"order_type": "stop_loss", "trigger_price": 59800}
    assert _send_fill_alert(entry, order, 59800.0, _CFG, conn) is True
    fields = {f["name"]: f["value"]
              for f in mock_post.call_args[1]["json"]["embeds"][0]["fields"]}
    # Fallback → risk |62000-59800|=2200, R=(59800-62000)/2200 = -1.00R
    assert fields["R Value"] == "-1.00R"


@patch("scanner.execution.alerts.requests.post")
def test_send_fill_alert_anchors_on_original_not_drift(mock_post):
    """End-to-end anti-vacuity (Green ≠ verified): the re-pointed lookup must
    DRIVE the anchor. Original SL 100 (earlier) + drift SL 95 (later); entry
    105 LONG, fill 102. Original anchor → R=(102-105)/|105-100|=-0.60R; the
    drift SL would give -0.30R. The alert must report -0.60R."""
    mock_post.return_value = MagicMock(status_code=204)
    conn = _conn()
    _ins(conn, "sX", "entry", created="2026-01-01T00:00:00+00:00")
    _ins(conn, "sX", "stop_loss", trig=100.0, status="CANCELLED",
         created="2026-01-01T00:00:01+00:00")           # ORIGINAL
    _ins(conn, "sX", "stop_loss", trig=95.0,
         created="2026-01-02T00:00:00+00:00")           # drift-moved
    entry = {"scan_id": "sX", "coin": "BTC", "direction": "LONG", "entry_price": 105.0}
    order = {"order_type": "tp1"}  # TP → no fallback; R only via the DB anchor
    assert _send_fill_alert(entry, order, 102.0, _CFG, conn) is True
    fields = {f["name"]: f["value"]
              for f in mock_post.call_args[1]["json"]["embeds"][0]["fields"]}
    assert fields["R Value"] == "-0.60R"      # original anchor, NOT -0.30R drift
