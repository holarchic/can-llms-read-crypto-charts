"""T-016 Phase-3b — reconcile_positions (KL#0 / SPEC #29), windows (a)+(c).

Drives the REAL `reconcile_positions` over a real in-memory DB with mocked
exchange truth. Pins: window (a) book/skip, the multi-row accepted residual,
window (c) flip + every untouched discriminator, the fail-closed paths
(absent AND strict-lossy → zero mutation), idempotency, double-book
prevention, and the return count. Green ≠ verified — every assertion drives
a real DB write/non-write through the production function.
"""

import sqlite3
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from scanner.execution import monitor as M
from scanner.execution.config import ExecutionConfig
from scanner.execution.exchange import PositionParseError
from scanner.execution.reconcile import (
    RECONCILE_STRANDED_MIN_AGE_S,
    reconcile_positions,
)
from scanner.execution.storage import (
    claim_pending,
    ensure_orders_table,
    save_order,
    save_pending_entry,
)


def _cfg():
    return ExecutionConfig(exchange="hyperliquid", mode="testnet")


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_orders_table(c)
    return c


def _seed_entry(conn, scan_id="s1", coin="ABC", direction="LONG", qty=1.0):
    save_order({
        "order_id": f"e_{scan_id}", "scan_id": scan_id, "coin": coin,
        "direction": direction, "order_type": "entry", "status": "FILLED",
        "fill_price": 100.0, "entry_price": 100.0, "quantity": qty,
        "exchange": "hyperliquid", "execution_mode": "testnet",
        "created_at": "2026-01-01T00:00:00+00:00",
        "filled_at": "2026-01-01T00:00:00+00:00",
    }, conn)


def _seed_triggered_pending(conn, scan_id="p1", coin="ABC", age_s=99999):
    save_pending_entry({
        "pending_id": scan_id, "scan_id": scan_id, "coin": coin,
        "direction": "LONG", "machine_block": "{}", "entry_trigger": "limit_at",
        "entry_trigger_price": 100.0, "invalidation_price": 90.0,
        "expiry_candles": 12, "exchange": "hyperliquid",
        "execution_mode": "testnet", "created_at": "2026-01-01T00:00:00+00:00",
    }, conn)
    claim_pending(scan_id, "TRIGGERED", conn)  # sets resolved_at = now
    old = (datetime.now(timezone.utc) - timedelta(seconds=age_s)).isoformat()
    conn.execute("UPDATE pending_entries SET resolved_at=? WHERE pending_id=?",
                 (old, scan_id))
    conn.commit()


def _manual_closes(conn, scan_id=None):
    q = "SELECT * FROM orders WHERE order_type='manual_close'"
    args: tuple = ()
    if scan_id:
        q += " AND scan_id=?"
        args = (scan_id,)
    return conn.execute(q, args).fetchall()


def _pending_status(conn, pid):
    return conn.execute(
        "SELECT status FROM pending_entries WHERE pending_id=?", (pid,),
    ).fetchone()["status"]


@contextmanager
def _patch_positions(value=None, side_effect=None):
    # Phase-3b-(b) added a `get_exchange_open_orders` read into the SAME
    # up-front fail-closed try. The a+c tests don't exercise window (b), so
    # default open-orders to [] (window b → no-op); when get_open_positions
    # raises, the try aborts before open-orders matters (fail-closed intact).
    with ExitStack() as es:
        es.enter_context(patch(
            "scanner.execution.exchange.get_open_positions",
            return_value=value if value is not None else [],
            side_effect=side_effect))
        es.enter_context(patch(
            "scanner.execution.exchange.get_exchange_open_orders",
            return_value=[]))
        yield


# --- Window (a): exchange-closed / DB-open ---

class TestWindowABookMissedClose:
    def test_db_open_exchange_flat_books_manual_close(self, conn):
        _seed_entry(conn, scan_id="s1", coin="ABC", qty=2.0)
        with _patch_positions([]):  # exchange flat
            n = reconcile_positions(_cfg(), conn)
        rows = _manual_closes(conn, "s1")
        assert n == 1
        assert len(rows) == 1
        r = rows[0]
        assert r["coin"] == "ABC" and r["direction"] == "LONG"
        assert r["status"] == "FILLED" and r["quantity"] == 2.0
        assert r["fill_price"] == 100.0

    def test_idempotent_second_run_noop(self, conn):
        _seed_entry(conn, scan_id="s1")
        with _patch_positions([]):
            assert reconcile_positions(_cfg(), conn) == 1
            assert reconcile_positions(_cfg(), conn) == 0  # entry now closed
        assert len(_manual_closes(conn, "s1")) == 1

    def test_live_position_not_closed(self, conn):
        _seed_entry(conn, scan_id="s1", coin="ABC", direction="LONG")
        with _patch_positions([{"coin": "ABC", "direction": "LONG", "size": 1.0}]):
            n = reconcile_positions(_cfg(), conn)
        assert n == 0
        assert _manual_closes(conn) == []

    def test_multi_row_same_coin_dir_with_position_zero_closes(self, conn):
        # codex plan-review R1 MEDIUM accepted residual: fail-closed-safe,
        # not corrective — one net position ⇒ none of the N rows closed.
        _seed_entry(conn, scan_id="s1", coin="ABC", direction="LONG")
        _seed_entry(conn, scan_id="s2", coin="ABC", direction="LONG")
        with _patch_positions([{"coin": "ABC", "direction": "LONG", "size": 1.0}]):
            n = reconcile_positions(_cfg(), conn)
        assert n == 0
        assert _manual_closes(conn) == []


# --- Window (c): stranded TRIGGERED pending ---

class TestWindowCStrandedPending:
    def test_stranded_triggered_resolved(self, conn):
        _seed_triggered_pending(conn, scan_id="p1", coin="ZZZ", age_s=99999)
        with _patch_positions([]):
            n = reconcile_positions(_cfg(), conn)
        assert n == 1
        assert _pending_status(conn, "p1") == "TRIGGER_FAILED"

    def test_filled_entry_means_legit_not_flipped(self, conn):
        # Realistic legit case: triggered → entry filled → position live. The
        # FILLED entry blocks window (c); the live position blocks window (a).
        _seed_triggered_pending(conn, scan_id="p1", coin="ZZZ", age_s=99999)
        _seed_entry(conn, scan_id="p1", coin="ZZZ")  # a real trade placed
        with _patch_positions([{"coin": "ZZZ", "direction": "LONG", "size": 1.0}]):
            n = reconcile_positions(_cfg(), conn)
        assert n == 0
        assert _pending_status(conn, "p1") == "TRIGGERED"
        assert _manual_closes(conn) == []

    def test_live_position_means_not_flipped(self, conn):
        _seed_triggered_pending(conn, scan_id="p1", coin="ZZZ", age_s=99999)
        with _patch_positions([{"coin": "ZZZ", "direction": "LONG", "size": 1.0}]):
            n = reconcile_positions(_cfg(), conn)
        assert n == 0
        assert _pending_status(conn, "p1") == "TRIGGERED"

    def test_tz_naive_resolved_at_skipped_not_crash(self, conn):
        # codex phase-review (correctness MEDIUM): a parseable-but-tz-naive
        # resolved_at must be a conservative SKIP, not an uncaught TypeError
        # on aware-naive subtraction that aborts the whole monitor cycle.
        _seed_triggered_pending(conn, scan_id="p1", coin="ZZZ", age_s=99999)
        conn.execute(
            "UPDATE pending_entries SET resolved_at=? WHERE pending_id='p1'",
            ("2026-01-01T00:00:00",),  # parseable, NO tz offset
        )
        conn.commit()
        with _patch_positions([]):
            n = reconcile_positions(_cfg(), conn)  # must not raise
        assert n == 0
        assert _pending_status(conn, "p1") == "TRIGGERED"

    def test_recent_claim_not_flipped(self, conn):
        # Younger than the age gate → an in-flight execute_trade may be running.
        _seed_triggered_pending(conn, scan_id="p1", coin="ZZZ",
                                age_s=RECONCILE_STRANDED_MIN_AGE_S - 60)
        with _patch_positions([]):
            n = reconcile_positions(_cfg(), conn)
        assert n == 0
        assert _pending_status(conn, "p1") == "TRIGGERED"


# --- THE fail-closed invariant (the single most important tests) ---

class TestFailClosed:
    def test_exchange_read_raises_zero_mutation(self, conn):
        _seed_entry(conn, scan_id="s1")
        _seed_triggered_pending(conn, scan_id="p1", coin="ZZZ", age_s=99999)
        with _patch_positions(side_effect=RuntimeError("network down")):
            n = reconcile_positions(_cfg(), conn)
        assert n == 0
        assert _manual_closes(conn) == []
        assert _pending_status(conn, "p1") == "TRIGGERED"

    def test_strict_lossy_snapshot_zero_mutation(self, conn):
        # strict=True turns a successful-but-lossy snapshot into a raise →
        # the SAME fail-closed guard. A seeded DB-open row is NOT closed.
        _seed_entry(conn, scan_id="s1")
        with _patch_positions(side_effect=PositionParseError("unparseable side")):
            n = reconcile_positions(_cfg(), conn)
        assert n == 0
        assert _manual_closes(conn) == []


# --- Double-book prevention (codex plan-review R1 LOW) ---

class TestDoubleBookPrevention:
    def test_reconcile_close_blocks_reactive_synthetic_same_cycle(self, conn):
        """After reconcile books the manual_close, attempt_sl_recovery for the
        same scan must NOT book a second close (the entry has dropped out of
        the open set → recovery sees 'already closed')."""
        _seed_entry(conn, scan_id="s1", coin="ABC", direction="LONG")
        with _patch_positions([]):
            assert reconcile_positions(_cfg(), conn) == 1
        assert len(_manual_closes(conn, "s1")) == 1

        issue = {"scan_id": "s1", "coin": "ABC", "direction": "LONG",
                 "type": "missing_sl"}
        with patch.object(M, "get_auth_exchange", return_value=object()), \
             patch.object(M, "get_exchange_meta",
                          return_value={"symbol_template": "{coin}/USDC:USDC"}):
            recovered = M.attempt_sl_recovery(issue, _cfg(), conn)

        assert recovered is True  # remaining qty 0 → "already closed", no-op
        assert len(_manual_closes(conn, "s1")) == 1  # still exactly one


# --- Return count ---

def test_count_is_closes_plus_resolutions(conn):
    _seed_entry(conn, scan_id="s1", coin="ABC")          # window (a)
    _seed_triggered_pending(conn, scan_id="p1", coin="ZZZ", age_s=99999)  # (c)
    with _patch_positions([]):
        n = reconcile_positions(_cfg(), conn)
    assert n == 2
