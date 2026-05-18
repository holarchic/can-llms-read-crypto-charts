"""Tests for orders table CRUD operations."""

import sqlite3

import pytest

from scanner.execution.storage import (
    claim_pending,
    count_open_longs,
    count_open_positions,
    ensure_orders_table,
    get_open_entry_orders,
    get_open_position_coins,
    get_order,
    get_orders_for_scan,
    get_triggered_pending_entries,
    has_open_position,
    has_unprotected_positions,
    make_cloid,
    new_order_id,
    save_order,
    save_pending_entry,
    sum_open_notional,
    update_order_status,
)


@pytest.fixture
def conn():
    """In-memory DB with orders table."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_orders_table(c)
    return c


def _make_order(scan_id="scan1", coin="BTC", direction="LONG", order_type="entry", **kwargs):
    """Helper to build an order dict with defaults."""
    base = {
        "order_id": new_order_id(),
        "scan_id": scan_id,
        "coin": coin,
        "direction": direction,
        "order_type": order_type,
        "status": "PENDING",
        "exchange": "hyperliquid",
        "execution_mode": "testnet",
        "created_at": "2026-02-24T00:00:00+00:00",
    }
    base.update(kwargs)
    return base


class TestEnsureOrdersTable:
    def test_table_created(self, conn):
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='orders'"
        ).fetchone()
        assert tables is not None

    def test_idempotent(self, conn):
        ensure_orders_table(conn)  # second call doesn't crash
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='orders'"
        ).fetchone()
        assert tables is not None


class TestSaveAndGet:
    def test_round_trip(self, conn):
        order = _make_order(entry_price=62000, quantity=0.1, leverage=5)
        save_order(order, conn)
        result = get_order(order["order_id"], conn)
        assert result is not None
        assert result["coin"] == "BTC"
        assert result["entry_price"] == 62000
        assert result["quantity"] == 0.1

    def test_get_missing_returns_none(self, conn):
        assert get_order("nonexistent", conn) is None

    def test_orders_for_scan(self, conn):
        o1 = _make_order(scan_id="scanA", order_type="entry")
        o2 = _make_order(scan_id="scanA", order_type="stop_loss")
        o3 = _make_order(scan_id="scanB", order_type="entry")
        for o in [o1, o2, o3]:
            save_order(o, conn)

        orders = get_orders_for_scan("scanA", conn)
        assert len(orders) == 2
        assert all(o["scan_id"] == "scanA" for o in orders)


class TestUpdateOrderStatus:
    def test_status_update(self, conn):
        order = _make_order()
        save_order(order, conn)
        update_order_status(order["order_id"], "PLACED", conn)
        result = get_order(order["order_id"], conn)
        assert result["status"] == "PLACED"

    def test_update_with_fill(self, conn):
        order = _make_order()
        save_order(order, conn)
        update_order_status(order["order_id"], "PLACED", conn)
        update_order_status(
            order["order_id"], "FILLED", conn,
            fill_price=62100, exchange_order_id="exch123",
        )
        result = get_order(order["order_id"], conn)
        assert result["status"] == "FILLED"
        assert result["fill_price"] == 62100
        assert result["exchange_order_id"] == "exch123"
        assert result["filled_at"] is not None

    def test_cancelled_sets_timestamp(self, conn):
        order = _make_order()
        save_order(order, conn)
        update_order_status(order["order_id"], "CANCELLED", conn)
        result = get_order(order["order_id"], conn)
        assert result["cancelled_at"] is not None

    def test_update_with_pnl(self, conn):
        order = _make_order()
        save_order(order, conn)
        update_order_status(order["order_id"], "PLACED", conn)
        update_order_status(
            order["order_id"], "FILLED", conn,
            realized_pnl=150.50, funding_paid=-2.30,
        )
        result = get_order(order["order_id"], conn)
        assert result["realized_pnl"] == 150.50
        assert result["funding_paid"] == -2.30

    def test_filled_to_cancelled_blocked(self, conn):
        """FILLED is terminal — cannot regress to CANCELLED."""
        order = _make_order()
        save_order(order, conn)
        update_order_status(order["order_id"], "PLACED", conn)
        update_order_status(order["order_id"], "FILLED", conn, fill_price=62100)
        update_order_status(order["order_id"], "CANCELLED", conn)
        result = get_order(order["order_id"], conn)
        assert result["status"] == "FILLED"

    def test_placed_to_filled_allowed(self, conn):
        """Normal fill transition: PLACED → FILLED."""
        order = _make_order()
        save_order(order, conn)
        update_order_status(order["order_id"], "PLACED", conn)
        update_order_status(order["order_id"], "FILLED", conn, fill_price=62100)
        result = get_order(order["order_id"], conn)
        assert result["status"] == "FILLED"

    def test_cancelled_is_terminal(self, conn):
        """CANCELLED is terminal — resurrecting to PLACED is rejected.

        Replacements now insert new rows instead of mutating the old row, so
        the old CANCELLED row should stay CANCELLED.
        """
        order = _make_order()
        save_order(order, conn)
        update_order_status(order["order_id"], "PLACED", conn)
        update_order_status(order["order_id"], "CANCELLED", conn)
        update_order_status(order["order_id"], "PLACED", conn)  # no-op, logs warning
        result = get_order(order["order_id"], conn)
        assert result["status"] == "CANCELLED"

    def test_filled_is_terminal(self, conn):
        """FILLED is terminal — partial-fill residual handled via new rows."""
        order = _make_order()
        save_order(order, conn)
        update_order_status(order["order_id"], "PLACED", conn)
        update_order_status(order["order_id"], "FILLED", conn, fill_price=62100)
        update_order_status(order["order_id"], "PLACED", conn)  # no-op, logs warning
        result = get_order(order["order_id"], conn)
        assert result["status"] == "FILLED"


class TestPositionQueries:
    def _setup_open_position(self, conn, coin="BTC", direction="LONG", scan_id="scan1",
                             entry_price=62000, quantity=0.1):
        """Create a filled entry order (open position)."""
        entry = _make_order(
            scan_id=scan_id, coin=coin, direction=direction,
            order_type="entry", status="FILLED",
            entry_price=entry_price, quantity=quantity,
        )
        save_order(entry, conn)
        return entry

    def _close_position(self, conn, scan_id="scan1", order_type="stop_loss", quantity=None):
        """Create a filled exit order to close a position.

        If quantity is None, looks up entry quantity for this scan (full close).
        """
        if quantity is None:
            row = conn.execute(
                "SELECT quantity FROM orders WHERE scan_id = ? AND order_type = 'entry'",
                (scan_id,),
            ).fetchone()
            quantity = row["quantity"] if row else 0.1
        exit_order = _make_order(
            scan_id=scan_id, order_type=order_type, status="FILLED", quantity=quantity,
        )
        save_order(exit_order, conn)

    def test_count_open_positions(self, conn):
        self._setup_open_position(conn, scan_id="s1")
        self._setup_open_position(conn, coin="ETH", scan_id="s2")
        assert count_open_positions(conn) == 2

    def test_closed_position_not_counted(self, conn):
        self._setup_open_position(conn, scan_id="s1")
        self._close_position(conn, scan_id="s1")
        assert count_open_positions(conn) == 0

    def test_cancelled_exit_still_open(self, conn):
        self._setup_open_position(conn, scan_id="s1")
        exit_order = _make_order(scan_id="s1", order_type="stop_loss", status="CANCELLED")
        save_order(exit_order, conn)
        assert count_open_positions(conn) == 1

    def test_has_open_position(self, conn):
        self._setup_open_position(conn, coin="BTC", direction="LONG")
        assert has_open_position("BTC", "LONG", conn) is True
        assert has_open_position("BTC", "SHORT", conn) is False
        assert has_open_position("BTC", None, conn) is True  # coin-only check
        assert has_open_position("ETH", "LONG", conn) is False
        assert has_open_position("ETH", None, conn) is False

    def test_count_open_longs(self, conn):
        self._setup_open_position(conn, direction="LONG", scan_id="s1")
        self._setup_open_position(conn, direction="SHORT", scan_id="s2")
        assert count_open_longs(conn) == 1

    def test_sum_open_notional(self, conn):
        self._setup_open_position(conn, scan_id="s1", entry_price=60000, quantity=0.1)
        self._setup_open_position(conn, scan_id="s2", coin="ETH", entry_price=3000, quantity=1.0)
        # 60000 * 0.1 + 3000 * 1.0 = 6000 + 3000 = 9000
        assert sum_open_notional(conn) == 9000.0

    def test_get_open_entry_orders(self, conn):
        self._setup_open_position(conn, scan_id="s1")
        self._setup_open_position(conn, scan_id="s2", coin="ETH")
        self._close_position(conn, scan_id="s2")

        open_entries = get_open_entry_orders(conn)
        assert len(open_entries) == 1
        assert open_entries[0]["scan_id"] == "s1"

    def test_partial_exit_still_open(self, conn):
        """TP1 fills ~50% → position still counted as open."""
        self._setup_open_position(conn, scan_id="s1", quantity=0.1)
        # TP1 fills for half the quantity
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="FILLED", quantity=0.05)
        save_order(tp1, conn)
        assert count_open_positions(conn) == 1

    def test_full_exit_tp1_tp2_closed(self, conn):
        """TP1 + TP2 fill full quantity → position closed."""
        self._setup_open_position(conn, scan_id="s1", quantity=0.1)
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="FILLED", quantity=0.05)
        tp2 = _make_order(scan_id="s1", order_type="tp2", status="FILLED", quantity=0.05)
        save_order(tp1, conn)
        save_order(tp2, conn)
        assert count_open_positions(conn) == 0

    def test_tp1_then_sl_closed(self, conn):
        """TP1 fills half, SL fills remainder → position closed."""
        self._setup_open_position(conn, scan_id="s1", quantity=0.1)
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="FILLED", quantity=0.05)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="FILLED", quantity=0.05)
        save_order(tp1, conn)
        save_order(sl, conn)
        assert count_open_positions(conn) == 0

    def test_has_open_position_partial(self, conn):
        """Partial exit still blocks duplicate entry for same coin+direction."""
        self._setup_open_position(conn, scan_id="s1", coin="BTC", direction="LONG", quantity=0.1)
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="FILLED", quantity=0.05)
        save_order(tp1, conn)
        assert has_open_position("BTC", "LONG", conn) is True

    def test_sum_open_notional_partial(self, conn):
        """Notional reflects remaining quantity after partial exit."""
        self._setup_open_position(conn, scan_id="s1", entry_price=60000, quantity=0.1)
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="FILLED", quantity=0.05)
        save_order(tp1, conn)
        # Remaining: 60000 * 0.05 = 3000
        assert sum_open_notional(conn) == 3000.0

    def test_count_open_longs_partial(self, conn):
        """Partial LONG exit → still counted as open long."""
        self._setup_open_position(conn, scan_id="s1", direction="LONG", quantity=0.1)
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="FILLED", quantity=0.05)
        save_order(tp1, conn)
        assert count_open_longs(conn) == 1


class TestTimeoutCloseExclusion:
    """FILLED timeout_close orders close positions just like SL/TP."""

    def test_timeout_close_excludes_from_open(self, conn):
        """Entry with FILLED timeout_close → not returned by get_open_entry_orders."""
        entry = _make_order(scan_id="s1", status="FILLED", entry_price=62000, quantity=0.1)
        save_order(entry, conn)
        assert len(get_open_entry_orders(conn)) == 1

        timeout = _make_order(
            scan_id="s1", order_type="timeout_close", status="FILLED", quantity=0.1,
        )
        save_order(timeout, conn)
        assert len(get_open_entry_orders(conn)) == 0


class TestExecutionModeIsolation:
    """Testnet and live orders must not cross-contaminate risk checks."""

    def _setup_testnet_position(self, conn):
        entry = _make_order(
            scan_id="t1", coin="BTC", direction="LONG", status="FILLED",
            entry_price=60000, quantity=0.1, execution_mode="testnet",
        )
        save_order(entry, conn)

    def test_live_queries_ignore_testnet_orders(self, conn):
        self._setup_testnet_position(conn)
        assert count_open_positions(conn, execution_mode="live") == 0
        assert has_open_position("BTC", "LONG", conn, execution_mode="live") is False
        assert count_open_longs(conn, execution_mode="live") == 0
        assert sum_open_notional(conn, execution_mode="live") == 0.0
        assert get_open_entry_orders(conn, execution_mode="live") == []

    def test_testnet_queries_see_testnet_orders(self, conn):
        self._setup_testnet_position(conn)
        assert count_open_positions(conn, execution_mode="testnet") == 1
        assert has_open_position("BTC", "LONG", conn, execution_mode="testnet") is True
        assert count_open_longs(conn, execution_mode="testnet") == 1
        assert sum_open_notional(conn, execution_mode="testnet") == 6000.0
        assert len(get_open_entry_orders(conn, execution_mode="testnet")) == 1


class TestMakeCloid:
    def test_deterministic(self):
        assert make_cloid("abc123") == make_cloid("abc123")

    def test_format(self):
        cloid = make_cloid("test-order")
        assert cloid.startswith("0x")
        assert len(cloid) == 34  # 0x + 32 hex chars

    def test_unique_inputs(self):
        assert make_cloid("order-1") != make_cloid("order-2")


class TestOrderScanOrdering:
    """get_orders_for_scan returns deterministic entry→SL→TP1→TP2 order."""

    def test_same_timestamp_returns_entry_sl_tp1_tp2(self, conn):
        """Orders with identical created_at sort by type: entry→stop_loss→tp1→tp2."""
        ts = "2026-02-24T00:00:00+00:00"
        # Insert in reverse order to prove it's not insertion-order dependent
        tp2 = _make_order(scan_id="s1", order_type="tp2", created_at=ts)
        tp1 = _make_order(scan_id="s1", order_type="tp1", created_at=ts)
        sl = _make_order(scan_id="s1", order_type="stop_loss", created_at=ts)
        entry = _make_order(scan_id="s1", order_type="entry", created_at=ts)
        for o in [tp2, tp1, sl, entry]:
            save_order(o, conn)

        orders = get_orders_for_scan("s1", conn)
        types = [o["order_type"] for o in orders]
        assert types == ["entry", "stop_loss", "tp1", "tp2"]


class TestFloatingPointPrecision:
    """IEEE 754 floating-point residuals must not create ghost positions."""

    def _setup_open_position(self, conn, scan_id="scan1", coin="BTC", direction="SHORT",
                             quantity=0.085):
        entry = _make_order(
            scan_id=scan_id, coin=coin, direction=direction,
            order_type="entry", status="FILLED",
            entry_price=476.89, quantity=quantity,
        )
        save_order(entry, conn)
        return entry

    def test_tp1_plus_timeout_closes_position(self, conn):
        """BCH scenario: 0.043 + 0.042 = 0.085 must close 0.085 entry."""
        self._setup_open_position(conn, scan_id="bch1", quantity=0.085)
        tp1 = _make_order(scan_id="bch1", order_type="tp1", status="FILLED", quantity=0.043)
        timeout = _make_order(scan_id="bch1", order_type="timeout_close", status="FILLED", quantity=0.042)
        save_order(tp1, conn)
        save_order(timeout, conn)

        assert count_open_positions(conn) == 0
        assert get_open_entry_orders(conn) == []
        assert has_open_position("BTC", "SHORT", conn) is False
        assert sum_open_notional(conn) == 0.0
        assert get_open_position_coins(conn) == []
        assert has_unprotected_positions(conn) is False

    def test_three_exit_fills_close_position(self, conn):
        """Multiple small exits that sum to entry qty must close position."""
        self._setup_open_position(conn, scan_id="s1", quantity=0.1)
        for qty in [0.033, 0.033, 0.034]:
            exit_order = _make_order(scan_id="s1", order_type="tp1", status="FILLED", quantity=qty)
            save_order(exit_order, conn)

        assert count_open_positions(conn) == 0

    def test_genuine_partial_still_open(self, conn):
        """A real partial exit (not just FP residual) keeps position open."""
        self._setup_open_position(conn, scan_id="s1", quantity=0.1)
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="FILLED", quantity=0.05)
        save_order(tp1, conn)

        assert count_open_positions(conn) == 1
        assert len(get_open_entry_orders(conn)) == 1
        assert len(get_open_position_coins(conn)) == 1


def _make_pending(scan_id, coin="BTC", exchange="hyperliquid", mode="testnet"):
    return {
        "pending_id": new_order_id(),
        "scan_id": scan_id,
        "coin": coin,
        "direction": "LONG",
        "machine_block": "{}",
        "entry_trigger": "limit_at",
        "entry_trigger_price": 100.0,
        "invalidation_price": 90.0,
        "expiry_candles": 4,
        "exchange": exchange,
        "execution_mode": mode,
        "created_at": "2026-02-24T00:00:00+00:00",
    }


class TestGetTriggeredPendingEntries:
    """Phase-3b C1 (SPEC #29): the storage-owned read for the stranded-pending
    reconcile window — TRIGGERED only, scoped by exchange/mode."""

    def test_returns_only_triggered_for_scope(self, conn):
        # ACTIVE (untouched), TRIGGERED in-scope, TRIGGERED out-of-scope.
        active = _make_pending("s_active")
        in_scope = _make_pending("s_trig")
        other_mode = _make_pending("s_trig_live", mode="live")
        for p in (active, in_scope, other_mode):
            save_pending_entry(p, conn)
        assert claim_pending(in_scope["pending_id"], "TRIGGERED", conn) is True
        assert claim_pending(other_mode["pending_id"], "TRIGGERED", conn) is True

        rows = get_triggered_pending_entries(conn, "hyperliquid", "testnet")

        assert [r["scan_id"] for r in rows] == ["s_trig"]
        assert rows[0]["status"] == "TRIGGERED"

    def test_empty_when_none_triggered(self, conn):
        save_pending_entry(_make_pending("s1"), conn)  # stays ACTIVE
        assert get_triggered_pending_entries(conn, "hyperliquid", "testnet") == []
