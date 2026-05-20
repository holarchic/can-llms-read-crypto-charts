"""Tests for position monitoring + order status sync."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from scanner.execution.config import ExecutionConfig
from scanner.execution.monitor import (
    _calc_r,
    _send_fill_alert,
    check_positions,
    enforce_position_timeouts,
    sync_order_status,
)
from scanner.execution.storage import (
    ensure_orders_table,
    get_order,
    get_orders_for_scan,
    get_remaining_qty,
    new_order_id,
    save_order,
)


@pytest.fixture
def config():
    return ExecutionConfig(
        enabled=True,
        mode="testnet",
        private_key="0xtest",
        wallet_address="0xwallet",
    )


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_orders_table(c)
    return c


def _make_order(scan_id="scan1", coin="BTC", direction="LONG", order_type="entry",
                status="FILLED", exchange_order_id="exch1", **kwargs):
    base = {
        "order_id": new_order_id(),
        "scan_id": scan_id,
        "coin": coin,
        "direction": direction,
        "order_type": order_type,
        "status": status,
        "exchange_order_id": exchange_order_id,
        "exchange": "hyperliquid",
        "execution_mode": "testnet",
        "entry_price": 62000,
        "quantity": 0.1,
        "created_at": "2026-02-24T00:00:00+00:00",
    }
    base.update(kwargs)
    return base


def _open_position(coin="BTC", direction="LONG", size=0.1) -> list[dict]:
    """Mock fetch_positions return shape for an open position."""
    return [{
        "symbol": f"{coin}/USDC:USDC",
        "contracts": size,
        "side": direction,
        "entryPrice": 62000,
        "unrealizedPnl": 0,
        "leverage": 5,
    }]


def _latest_placed(conn, scan_id: str, order_type: str) -> dict | None:
    """Find the most recent PLACED row for a scan + type.

    Replacements insert NEW rows instead of mutating old ones, so tests
    should look up the "current" SL/TP by this query rather than by the
    original order_id.
    """
    row = conn.execute(
        "SELECT * FROM orders WHERE scan_id = ? AND order_type = ? "
        "AND status = 'PLACED' ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (scan_id, order_type),
    ).fetchone()
    return dict(row) if row else None


class TestSyncOrderStatus:
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_detects_sl_fill(self, mock_auth, config, conn):
        """When exchange reports SL as closed, update DB status."""
        # Create open position with SL
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED")
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800)
        save_order(entry, conn)
        save_order(sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {
            "status": "closed",
            "average": 59750,
        }
        mock_auth.return_value = mock_exchange

        updated = sync_order_status(config, conn)
        assert updated >= 1

        sl_order = get_order(sl["order_id"], conn)
        assert sl_order["status"] == "FILLED"
        assert sl_order["fill_price"] == 59750

    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_detects_fill_with_none_price_fields(self, mock_auth, config, conn):
        """HL trigger orders return None for average/price — falls back to trigger_price."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED")
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800)
        save_order(entry, conn)
        save_order(sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {
            "status": "closed",
            "average": None,
            "price": None,
            "filled": 0.1,
        }
        mock_auth.return_value = mock_exchange

        updated = sync_order_status(config, conn)
        assert updated >= 1

        sl_order = get_order(sl["order_id"], conn)
        assert sl_order["status"] == "FILLED"
        assert sl_order["fill_price"] == 59800.0  # falls back to trigger_price

    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_closed_with_zero_fill_treated_as_cancel(self, mock_auth, config, conn):
        """Exchange returns 'closed' with filled=0 (auto-cancelled trigger order).

        This happens on HyperLiquid when a TP/SL trigger order is cancelled
        because the position was closed by another order. The exchange marks
        the lifecycle as 'closed' but nothing actually filled.
        """
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED")
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="PLACED",
                          exchange_order_id="tp1_exch1", trigger_price=65000,
                          quantity=0.05)
        save_order(entry, conn)
        save_order(tp1, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {
            "status": "closed",
            "average": 0,
            "price": 70000,  # junk market price — must not be used
            "filled": 0,
        }
        mock_auth.return_value = mock_exchange

        updated = sync_order_status(config, conn)
        assert updated >= 1

        order = get_order(tp1["order_id"], conn)
        assert order["status"] == "CANCELLED"
        assert order["error_message"] == "Closed with zero fill"
        assert order["fill_price"] is None  # no phantom fill recorded

    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_cancels_remaining_on_fill(self, mock_auth, mock_cancel, config, conn):
        """When SL fills, cancel TP orders."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED")
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1")
        tp = _make_order(scan_id="s1", order_type="tp1", status="PLACED",
                         exchange_order_id="tp_exch1")
        save_order(entry, conn)
        save_order(sl, conn)
        save_order(tp, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.side_effect = lambda oid, sym: (
            {"status": "closed", "average": 59750} if oid == "sl_exch1"
            else {"status": "open"}
        )
        mock_auth.return_value = mock_exchange

        sync_order_status(config, conn)

        tp_order = get_order(tp["order_id"], conn)
        assert tp_order["status"] == "CANCELLED"

    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_no_positions_no_updates(self, mock_auth, config, conn):
        """No open entries -> 0 updates."""
        mock_exchange = MagicMock()
        mock_auth.return_value = mock_exchange

        updated = sync_order_status(config, conn)
        assert updated == 0


class TestCheckPositions:
    @patch("scanner.execution.monitor.get_exchange_open_orders", return_value=[])
    def test_detects_missing_sl(self, mock_orders, config, conn):
        """Open position without active SL -> issue reported."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED")
        # SL was cancelled — no active SL
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="CANCELLED")
        save_order(entry, conn)
        save_order(sl, conn)

        issues = check_positions(config, conn)
        assert len(issues) == 1
        assert issues[0]["type"] == "missing_sl"

    @patch("scanner.execution.monitor.get_exchange_open_orders", return_value=[])
    def test_healthy_position(self, mock_orders, config, conn):
        """Position with active SL -> no issues."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED")
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1")
        save_order(entry, conn)
        save_order(sl, conn)

        # SL is in DB but not on exchange → phantom_sl, not "no issues"
        # For a true healthy test, exchange must return the SL order
        mock_orders.return_value = [{"id": "sl_exch1", "status": "open"}]
        issues = check_positions(config, conn)
        assert len(issues) == 0

    @patch("scanner.execution.monitor.get_exchange_open_orders", return_value=[])
    def test_no_positions(self, mock_orders, config, conn):
        """No open positions -> no issues."""
        issues = check_positions(config, conn)
        assert len(issues) == 0


class TestPhantomSLDetection:
    """check_positions verifies PLACED SL orders exist on exchange."""

    @patch("scanner.execution.monitor.get_exchange_open_orders", return_value=[])
    def test_check_positions_detects_phantom_sl(self, mock_orders, config, conn):
        """DB has PLACED SL but exchange returns empty → phantom_sl issue."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED")
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1")
        save_order(entry, conn)
        save_order(sl, conn)

        issues = check_positions(config, conn)
        assert len(issues) == 1
        assert issues[0]["type"] == "phantom_sl"
        assert "DB but not on exchange" in issues[0]["message"]

    @patch("scanner.execution.monitor.get_exchange_open_orders",
           side_effect=Exception("API timeout"))
    def test_check_positions_exchange_error_reports_failure(self, mock_orders, config, conn):
        """Exchange raises → exchange_fetch_failed issue reported."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED")
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1")
        save_order(entry, conn)
        save_order(sl, conn)

        issues = check_positions(config, conn)
        assert any(i["type"] == "exchange_fetch_failed" for i in issues)
        assert not any(i["type"] == "phantom_sl" for i in issues)  # SL check skipped

    @patch("scanner.execution.monitor.get_exchange_open_orders")
    def test_check_positions_sl_exists_on_exchange(self, mock_orders, config, conn):
        """DB and exchange agree SL exists → no issues."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED")
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1")
        save_order(entry, conn)
        save_order(sl, conn)

        mock_orders.return_value = [{"id": "sl_exch1", "status": "open"}]

        issues = check_positions(config, conn)
        assert len(issues) == 0

    @patch("scanner.execution.monitor.get_auth_exchange")
    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_exchange_open_orders", return_value=[])
    def test_missing_from_open_orders_reclassifies_fill_not_phantom(
        self, mock_orders, mock_cancel, mock_auth, config, conn,
    ):
        """SL gone from open_orders but fetch_order says status=closed,
        filled>0 → NOT a phantom. Mark FILLED, no phantom_sl issue.

        Regression for the 2026-04-17 DOT loop: drift placed a new SL that
        immediately triggered; check_positions previously treated it as
        phantom and recovery placed replacement SLs the exchange rejected.
        """
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED",
                            quantity=171.4)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="385089532315",
                         trigger_price=1.3273, quantity=171.4)
        save_order(entry, conn)
        save_order(sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {
            "status": "closed", "filled": 171.4, "average": 1.3293,
        }
        mock_auth.return_value = mock_exchange

        issues = check_positions(config, conn)

        assert not any(i["type"] == "phantom_sl" for i in issues)
        assert not any(i["type"] == "missing_sl" for i in issues)
        sl_after = get_order(sl["order_id"], conn)
        assert sl_after["status"] == "FILLED"
        assert sl_after["fill_price"] == 1.3293


class TestBatchFetchOptimization:
    """Batch fetch_open_orders reduces per-cycle API calls."""

    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_sync_skips_open_orders(self, mock_auth, config, conn):
        """Orders still in exchange open set are not individually fetched."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED")
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800)
        save_order(entry, conn)
        save_order(sl, conn)

        mock_exchange = MagicMock()
        mock_auth.return_value = mock_exchange

        exchange_open_orders = [{"id": "sl_exch1", "status": "open"}]
        updated = sync_order_status(config, conn,
                                    exchange_open_orders=exchange_open_orders)

        assert updated == 0
        mock_exchange.fetch_order.assert_not_called()

    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_sync_fetches_disappeared_orders(self, mock_auth, config, conn):
        """Orders not in exchange open set get individual fetch_order calls."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED")
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800)
        save_order(entry, conn)
        save_order(sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {
            "status": "closed", "average": 59750, "filled": 0.1,
        }
        mock_auth.return_value = mock_exchange

        exchange_open_orders = []  # SL disappeared
        updated = sync_order_status(config, conn,
                                    exchange_open_orders=exchange_open_orders)

        assert updated >= 1
        mock_exchange.fetch_order.assert_called_once_with("sl_exch1", "BTC/USDC:USDC")

        sl_order = get_order(sl["order_id"], conn)
        assert sl_order["status"] == "FILLED"

    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_sync_fallback_when_none(self, mock_auth, config, conn):
        """When exchange_open_orders=None, sync fetches its own open orders."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED")
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800)
        save_order(entry, conn)
        save_order(sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_open_orders.return_value = [{"id": "sl_exch1"}]
        mock_auth.return_value = mock_exchange

        updated = sync_order_status(config, conn)  # no exchange_open_orders

        assert updated == 0
        mock_exchange.fetch_open_orders.assert_called_once()
        mock_exchange.fetch_order.assert_not_called()

    @patch("scanner.execution.monitor.get_exchange_open_orders")
    def test_check_positions_uses_passed_orders(self, mock_get_orders, config, conn):
        """check_positions skips fetch when exchange_open_orders is passed."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED")
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1")
        save_order(entry, conn)
        save_order(sl, conn)

        issues = check_positions(config, conn,
                                 exchange_open_orders=[{"id": "sl_exch1"}])

        assert len(issues) == 0
        mock_get_orders.assert_not_called()


class TestStaleListSafety:
    """Exit fill breaks the inner loop to avoid stale-list corruption."""

    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_tp1_fill_does_not_reprocess_stale_sl(self, mock_auth, mock_cancel, config, conn):
        """After TP1 fills and replaces SL, the old SL must not be checked in same cycle."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="PLACED",
                          exchange_order_id="tp1_exch1", quantity=0.05)
        save_order(entry, conn)
        save_order(sl, conn)
        save_order(tp1, conn)

        mock_exchange = MagicMock()
        # SL still open, TP1 filled
        mock_exchange.fetch_order.side_effect = lambda oid, sym: (
            {"status": "closed", "average": 65000} if oid == "tp1_exch1"
            else {"status": "open"}
        )
        mock_exchange.create_order.return_value = {"id": "sl_exch2"}
        mock_exchange.amount_to_precision.return_value = "0.05"
        mock_auth.return_value = mock_exchange

        sync_order_status(config, conn)

        # fetch_order should be called for SL (open) then TP1 (filled)
        # After TP1 handling, the loop breaks — no further fetch_order calls
        fetched_oids = [call.args[0] for call in mock_exchange.fetch_order.call_args_list]
        assert "sl_exch1" in fetched_oids
        assert "tp1_exch1" in fetched_oids
        # Key: SL should NOT be fetched again after TP1 fill
        assert fetched_oids.count("sl_exch1") == 1

        # Old SL marked CANCELLED (superseded); new SL row has the replacement id
        assert get_order(sl["order_id"], conn)["status"] == "CANCELLED"
        new_sl = _latest_placed(conn, "s1", "stop_loss")
        assert new_sl is not None
        assert new_sl["exchange_order_id"] == "sl_exch2"


class TestPartialExitHandling:
    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_tp1_fill_adjusts_sl_keeps_tp2(self, mock_auth, mock_cancel, config, conn):
        """TP1 fills → SL cancelled + replaced at reduced qty, TP2 stays PLACED."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="PLACED",
                          exchange_order_id="tp1_exch1", quantity=0.05)
        tp2 = _make_order(scan_id="s1", order_type="tp2", status="PLACED",
                          exchange_order_id="tp2_exch1", quantity=0.05)
        save_order(entry, conn)
        save_order(sl, conn)
        save_order(tp1, conn)
        save_order(tp2, conn)

        mock_exchange = MagicMock()
        # TP1 reports as closed; others still open
        mock_exchange.fetch_order.side_effect = lambda oid, sym: (
            {"status": "closed", "average": 65000} if oid == "tp1_exch1"
            else {"status": "open"}
        )
        # SL replacement succeeds
        mock_exchange.create_order.return_value = {"id": "sl_exch2"}
        mock_exchange.amount_to_precision.return_value = "0.05"
        mock_auth.return_value = mock_exchange

        sync_order_status(config, conn)

        # TP1 should be FILLED
        tp1_result = get_order(tp1["order_id"], conn)
        assert tp1_result["status"] == "FILLED"

        # Old SL row kept as history (CANCELLED); new SL row inserted
        old_sl = get_order(sl["order_id"], conn)
        assert old_sl["status"] == "CANCELLED"
        assert old_sl["quantity"] == 0.1  # historical qty preserved
        new_sl = _latest_placed(conn, "s1", "stop_loss")
        assert new_sl is not None
        assert new_sl["quantity"] == 0.05
        assert new_sl["exchange_order_id"] == "sl_exch2"
        assert new_sl["order_id"] != sl["order_id"]

        # TP2 should still be PLACED (not cancelled)
        tp2_result = get_order(tp2["order_id"], conn)
        assert tp2_result["status"] == "PLACED"

        # Old SL should have been cancelled on exchange
        mock_cancel.assert_called_once()

    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_sl_replacement_has_cloid(self, mock_auth, mock_cancel, config, conn):
        """SL replacement create_order includes clientOrderId in params."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="PLACED",
                          exchange_order_id="tp1_exch1", quantity=0.05)
        save_order(entry, conn)
        save_order(sl, conn)
        save_order(tp1, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.side_effect = lambda oid, sym: (
            {"status": "closed", "average": 65000} if oid == "tp1_exch1"
            else {"status": "open"}
        )
        mock_exchange.create_order.return_value = {"id": "sl_exch2"}
        mock_exchange.amount_to_precision.return_value = "0.05"
        mock_auth.return_value = mock_exchange

        sync_order_status(config, conn)

        # Verify the SL replacement create_order got clientOrderId
        assert mock_exchange.create_order.called
        call_args = mock_exchange.create_order.call_args
        params = call_args[0][-1]  # last positional arg is params dict
        assert "clientOrderId" in params
        assert params["clientOrderId"].startswith("0x")

    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_tp2_fill_cancels_sl(self, mock_auth, mock_cancel, config, conn):
        """TP2 fills → SL cancelled (fully closed)."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        # TP1 already filled
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="FILLED",
                          exchange_order_id="tp1_exch1", quantity=0.05)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.05)
        tp2 = _make_order(scan_id="s1", order_type="tp2", status="PLACED",
                          exchange_order_id="tp2_exch1", quantity=0.05)
        save_order(entry, conn)
        save_order(tp1, conn)
        save_order(sl, conn)
        save_order(tp2, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.side_effect = lambda oid, sym: (
            {"status": "closed", "average": 68000} if oid == "tp2_exch1"
            else {"status": "open"}
        )
        mock_auth.return_value = mock_exchange

        sync_order_status(config, conn)

        # TP2 filled
        tp2_result = get_order(tp2["order_id"], conn)
        assert tp2_result["status"] == "FILLED"

        # SL cancelled (position fully closed)
        sl_result = get_order(sl["order_id"], conn)
        assert sl_result["status"] == "CANCELLED"

    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_closed_exit_uses_actual_fill_qty(self, mock_auth, mock_cancel, config, conn):
        """TP1 quantity=0.05 but exchange filled=0.03 → SL replacement uses entry - 0.03."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="PLACED",
                          exchange_order_id="tp1_exch1", quantity=0.05)
        save_order(entry, conn)
        save_order(sl, conn)
        save_order(tp1, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.side_effect = lambda oid, sym: (
            {"status": "closed", "average": 65000, "filled": 0.03} if oid == "tp1_exch1"
            else {"status": "open"}
        )
        mock_exchange.create_order.return_value = {"id": "sl_exch2"}
        mock_exchange.amount_to_precision.return_value = "0.07"
        mock_auth.return_value = mock_exchange

        sync_order_status(config, conn)

        # TP1 DB quantity updated to actual fill
        tp1_result = get_order(tp1["order_id"], conn)
        assert tp1_result["status"] == "FILLED"
        assert tp1_result["quantity"] == 0.03

        # Old SL cancelled; new SL inserted for remaining = 0.1 - 0.03 = 0.07
        assert get_order(sl["order_id"], conn)["status"] == "CANCELLED"
        new_sl = _latest_placed(conn, "s1", "stop_loss")
        assert new_sl is not None
        assert new_sl["quantity"] == 0.07
        assert new_sl["exchange_order_id"] == "sl_exch2"

    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_canceled_with_partial_fill_treated_as_fill(self, mock_auth, mock_cancel, config, conn):
        """SL canceled with filled=0.08/0.10 → DB shows FILLED qty=0.08, TPs cancelled."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="PLACED",
                          exchange_order_id="tp1_exch1", quantity=0.05)
        save_order(entry, conn)
        save_order(sl, conn)
        save_order(tp1, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.side_effect = lambda oid, sym: (
            {"status": "canceled", "filled": 0.08, "average": 59750} if oid == "sl_exch1"
            else {"status": "open"}
        )
        mock_auth.return_value = mock_exchange

        # create_order needed because residual placement kicks in
        mock_exchange.create_order.return_value = {"id": "sl_exch2"}
        mock_exchange.amount_to_precision.return_value = "0.02"

        sync_order_status(config, conn)

        # SL row is FILLED with actual partial fill quantity (history)
        sl_result = get_order(sl["order_id"], conn)
        assert sl_result["status"] == "FILLED"
        assert sl_result["quantity"] == 0.08
        assert sl_result["fill_price"] == 59750

        # TP1 should be cancelled (partial fill cleanup)
        tp1_result = get_order(tp1["order_id"], conn)
        assert tp1_result["status"] == "CANCELLED"

    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_canceled_with_zero_fill_gets_replaced(self, mock_auth, mock_sleep, config, conn):
        """SL canceled with filled=0 → new replacement row inserted."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        save_order(entry, conn)
        save_order(sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {"status": "canceled", "filled": 0}
        mock_exchange.fetch_positions.return_value = _open_position()
        mock_exchange.amount_to_precision.return_value = "0.1"
        mock_exchange.create_order.return_value = {"id": "sl_exch2"}
        mock_auth.return_value = mock_exchange

        updated = sync_order_status(config, conn)
        assert updated == 1

        # Old SL row stays CANCELLED; replacement inserted as PLACED row
        assert get_order(sl["order_id"], conn)["status"] == "CANCELLED"
        new_sl = _latest_placed(conn, "s1", "stop_loss")
        assert new_sl is not None
        assert new_sl["exchange_order_id"] == "sl_exch2"

    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_tp1_sl_replacement_retries_on_failure(self, mock_auth, mock_cancel, mock_sleep,
                                                    config, conn):
        """SL replacement fails twice then succeeds on 3rd attempt → SL placed."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="PLACED",
                          exchange_order_id="tp1_exch1", quantity=0.05)
        save_order(entry, conn)
        save_order(sl, conn)
        save_order(tp1, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.side_effect = lambda oid, sym: (
            {"status": "closed", "average": 65000} if oid == "tp1_exch1"
            else {"status": "open"}
        )
        mock_exchange.amount_to_precision.return_value = "0.05"
        # Fail twice, succeed on third
        mock_exchange.create_order.side_effect = [
            Exception("timeout"), Exception("timeout"), {"id": "sl_exch2"},
        ]
        mock_auth.return_value = mock_exchange

        sync_order_status(config, conn)

        # New SL row inserted as PLACED; old one marked CANCELLED
        assert get_order(sl["order_id"], conn)["status"] == "CANCELLED"
        new_sl = _latest_placed(conn, "s1", "stop_loss")
        assert new_sl is not None
        assert new_sl["exchange_order_id"] == "sl_exch2"
        assert mock_exchange.create_order.call_count == 3
        assert mock_sleep.call_count == 2  # backoff between retries

    @patch("scanner.execution.alerts.requests.post")
    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_tp1_sl_replacement_exhausted_sends_urgent(self, mock_auth, mock_cancel,
                                                        mock_sleep, mock_post, config, conn):
        """All 3 SL replacement attempts fail → URGENT Discord alert sent."""
        config.discord_urgent_webhook_url = "https://discord.com/urgent"
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="PLACED",
                          exchange_order_id="tp1_exch1", quantity=0.05)
        save_order(entry, conn)
        save_order(sl, conn)
        save_order(tp1, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.side_effect = lambda oid, sym: (
            {"status": "closed", "average": 65000} if oid == "tp1_exch1"
            else {"status": "open"}
        )
        mock_exchange.amount_to_precision.return_value = "0.05"
        mock_exchange.create_order.side_effect = Exception("exchange timeout")
        mock_auth.return_value = mock_exchange
        mock_post.return_value = MagicMock(status_code=204)

        sync_order_status(config, conn)

        # Old SL row stays PLACED (never superseded — replacement never succeeded)
        # Position is still protected at the original trigger/qty on exchange.
        sl_result = get_order(sl["order_id"], conn)
        assert sl_result["status"] == "PLACED"
        # No new PLACED SL row was inserted
        new_placed = conn.execute(
            "SELECT COUNT(*) as cnt FROM orders WHERE scan_id = ? "
            "AND order_type = 'stop_loss' AND status = 'PLACED'", ("s1",),
        ).fetchone()
        assert new_placed["cnt"] == 1  # only the original

        # URGENT alert should have been sent
        mock_post.assert_called_once()
        call_body = mock_post.call_args[1]["json"]["content"]
        assert "URGENT" in call_body
        assert "BTC" in call_body

    @patch("scanner.execution.alerts.requests.post")
    @patch("scanner.execution.exit_fills.cancel_order", return_value=False)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_tp1_cancel_failure_replaces_silently(
        self, mock_auth, mock_cancel, mock_post, config, conn,
    ):
        """SL cleanup-cancel after TP1 returns False (typically because the
        exchange already cleared the old SL via reduceOnly auto-cancel). New
        SL is placed first, so the position is fully protected — this branch
        is reduceOnly-safe and must NOT page. URGENT only fires on placement
        failure (separate test above)."""
        config.discord_urgent_webhook_url = "https://discord.com/urgent"
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="PLACED",
                          exchange_order_id="tp1_exch1", quantity=0.05)
        save_order(entry, conn)
        save_order(sl, conn)
        save_order(tp1, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.side_effect = lambda oid, sym: (
            {"status": "closed", "average": 65000} if oid == "tp1_exch1"
            else {"status": "open"}
        )
        mock_exchange.create_order.return_value = {"id": "sl_exch2"}
        mock_exchange.amount_to_precision.return_value = "0.05"
        mock_auth.return_value = mock_exchange
        mock_post.return_value = MagicMock(status_code=204)

        sync_order_status(config, conn)

        # No URGENT — cleanup-cancel-failure is reduceOnly-safe
        urgent_calls = [c for c in mock_post.call_args_list
                        if "URGENT" in str(c)]
        assert urgent_calls == []

        # New SL row placed at residual; old SL row stays PLACED (cancel failed)
        # Duplicate on exchange is reduceOnly-safe.
        new_sl = _latest_placed(conn, "s1", "stop_loss")
        assert new_sl is not None
        assert new_sl["exchange_order_id"] == "sl_exch2"
        assert new_sl["quantity"] == 0.05
        old_sl = get_order(sl["order_id"], conn)
        assert old_sl["status"] == "PLACED"  # cancel failed, not flipped


class TestUnknownOrderStatuses:
    """expired/rejected → CANCELLED; unknown → left PLACED with warning."""

    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_expired_order_gets_replaced(self, mock_auth, mock_sleep, config, conn):
        """Exchange reports status='expired' → new replacement row inserted."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        save_order(entry, conn)
        save_order(sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {"status": "expired"}
        mock_exchange.fetch_positions.return_value = _open_position()
        mock_exchange.amount_to_precision.return_value = "0.1"
        mock_exchange.create_order.return_value = {"id": "sl_exch2"}
        mock_auth.return_value = mock_exchange

        updated = sync_order_status(config, conn)
        assert updated == 1

        # Original SL row stays CANCELLED; new row inserted for residual
        assert get_order(sl["order_id"], conn)["status"] == "CANCELLED"
        new_sl = _latest_placed(conn, "s1", "stop_loss")
        assert new_sl is not None
        assert new_sl["exchange_order_id"] == "sl_exch2"

    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_rejected_order_gets_replaced(self, mock_auth, mock_sleep, config, conn):
        """Exchange reports status='rejected' → new replacement row inserted."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        save_order(entry, conn)
        save_order(sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {"status": "rejected"}
        mock_exchange.fetch_positions.return_value = _open_position()
        mock_exchange.amount_to_precision.return_value = "0.1"
        mock_exchange.create_order.return_value = {"id": "sl_exch2"}
        mock_auth.return_value = mock_exchange

        updated = sync_order_status(config, conn)
        assert updated == 1

        assert get_order(sl["order_id"], conn)["status"] == "CANCELLED"
        new_sl = _latest_placed(conn, "s1", "stop_loss")
        assert new_sl is not None
        assert new_sl["exchange_order_id"] == "sl_exch2"

    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_unknown_status_left_placed(self, mock_auth, config, conn, caplog):
        """Exchange returns unrecognized status → order stays PLACED, warning logged."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        save_order(entry, conn)
        save_order(sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {"status": "wtf_status"}
        mock_auth.return_value = mock_exchange

        import logging
        with caplog.at_level(logging.WARNING, logger="scanner.execution.monitor"):
            updated = sync_order_status(config, conn)

        assert updated == 0

        sl_result = get_order(sl["order_id"], conn)
        assert sl_result["status"] == "PLACED"
        assert any("unknown exchange status" in r.message for r in caplog.records)


class TestPartialSLTP2Fill:
    """Partial SL/TP2 fills should re-place orders at residual quantity."""

    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_sl_partial_fill_replaces_sl_and_tp2(self, mock_auth, mock_cancel, mock_sleep,
                                                  config, conn):
        """SL partially fills → residual gets new SL + TP2 resized."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        tp2 = _make_order(scan_id="s1", order_type="tp2", status="PLACED",
                          exchange_order_id="tp2_exch1", trigger_price=68000, quantity=0.1)
        save_order(entry, conn)
        save_order(sl, conn)
        save_order(tp2, conn)

        mock_exchange = MagicMock()
        # SL reports closed with partial fill (0.06 of 0.1)
        mock_exchange.fetch_order.side_effect = lambda oid, sym: (
            {"status": "closed", "average": 59750, "filled": 0.06} if oid == "sl_exch1"
            else {"status": "open"}
        )
        mock_exchange.create_order.side_effect = [
            {"id": "sl_exch2"},   # SL replacement
            {"id": "tp2_exch2"},  # TP2 replacement
        ]
        mock_exchange.amount_to_precision.return_value = "0.04"
        mock_auth.return_value = mock_exchange

        sync_order_status(config, conn)

        # Old SL row is FILLED with partial fill (historical record)
        old_sl = get_order(sl["order_id"], conn)
        assert old_sl["status"] == "FILLED"
        assert old_sl["quantity"] == pytest.approx(0.06)

        # New SL row inserted at residual qty
        new_sl = _latest_placed(conn, "s1", "stop_loss")
        assert new_sl is not None
        assert new_sl["quantity"] == pytest.approx(0.04)
        assert new_sl["exchange_order_id"] == "sl_exch2"

        # Old TP2 row is CANCELLED; new TP2 row inserted at residual
        old_tp2 = get_order(tp2["order_id"], conn)
        assert old_tp2["status"] == "CANCELLED"
        new_tp2 = _latest_placed(conn, "s1", "tp2")
        assert new_tp2 is not None
        assert new_tp2["quantity"] == pytest.approx(0.04)
        assert new_tp2["exchange_order_id"] == "tp2_exch2"

    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_tp2_partial_fill_replaces_tp2_and_sl(self, mock_auth, mock_cancel, mock_sleep,
                                                   config, conn):
        """TP2 partially fills → residual gets new TP2 + SL resized."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.05)
        # TP1 already filled
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="FILLED",
                          exchange_order_id="tp1_exch1", quantity=0.025)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.025)
        tp2 = _make_order(scan_id="s1", order_type="tp2", status="PLACED",
                          exchange_order_id="tp2_exch1", trigger_price=68000, quantity=0.025)
        save_order(entry, conn)
        save_order(tp1, conn)
        save_order(sl, conn)
        save_order(tp2, conn)

        mock_exchange = MagicMock()
        # TP2 partially fills (0.015 of 0.025)
        mock_exchange.fetch_order.side_effect = lambda oid, sym: (
            {"status": "closed", "average": 68000, "filled": 0.015} if oid == "tp2_exch1"
            else {"status": "open"}
        )
        mock_exchange.create_order.side_effect = [
            {"id": "sl_exch2"},   # SL replacement
            {"id": "tp2_exch2"},  # TP2 replacement
        ]
        mock_exchange.amount_to_precision.return_value = "0.01"
        mock_auth.return_value = mock_exchange

        sync_order_status(config, conn)

        # Old SL row CANCELLED; new SL row inserted
        assert get_order(sl["order_id"], conn)["status"] == "CANCELLED"
        new_sl = _latest_placed(conn, "s1", "stop_loss")
        assert new_sl is not None
        assert new_sl["quantity"] == pytest.approx(0.01)
        assert new_sl["exchange_order_id"] == "sl_exch2"

        # Old TP2 row FILLED (partial); new TP2 row inserted for residual
        old_tp2 = get_order(tp2["order_id"], conn)
        assert old_tp2["status"] == "FILLED"
        assert old_tp2["quantity"] == pytest.approx(0.015)
        new_tp2 = _latest_placed(conn, "s1", "tp2")
        assert new_tp2 is not None
        assert new_tp2["quantity"] == pytest.approx(0.01)
        assert new_tp2["exchange_order_id"] == "tp2_exch2"

    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_sl_partial_fill_no_active_tp2(self, mock_auth, mock_cancel, mock_sleep,
                                            config, conn):
        """SL partially fills with no TP2 → only SL re-placed."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        save_order(entry, conn)
        save_order(sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {
            "status": "closed", "average": 59750, "filled": 0.07,
        }
        mock_exchange.create_order.return_value = {"id": "sl_exch2"}
        mock_exchange.amount_to_precision.return_value = "0.03"
        mock_auth.return_value = mock_exchange

        sync_order_status(config, conn)

        # Old SL FILLED (partial) + new SL inserted for residual
        assert get_order(sl["order_id"], conn)["status"] == "FILLED"
        new_sl = _latest_placed(conn, "s1", "stop_loss")
        assert new_sl is not None
        assert new_sl["quantity"] == pytest.approx(0.03)
        assert new_sl["exchange_order_id"] == "sl_exch2"

        # Only one create_order call (SL only, no TP2)
        assert mock_exchange.create_order.call_count == 1

    @patch("scanner.execution.alerts.requests.post")
    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_sl_partial_fill_replacement_fails_sends_alert(self, mock_auth, mock_cancel,
                                                            mock_sleep, mock_post, config, conn):
        """SL partial fill, replacement fails after retries → urgent alert sent."""
        config.discord_urgent_webhook_url = "https://discord.com/urgent"
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        save_order(entry, conn)
        save_order(sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {
            "status": "closed", "average": 59750, "filled": 0.07,
        }
        mock_exchange.amount_to_precision.return_value = "0.03"
        mock_exchange.create_order.side_effect = Exception("exchange down")
        mock_auth.return_value = mock_exchange
        mock_post.return_value = MagicMock(status_code=204)

        sync_order_status(config, conn)

        # Urgent alert sent
        mock_post.assert_called_once()
        call_body = mock_post.call_args[1]["json"]["content"]
        assert "URGENT" in call_body
        assert "BTC" in call_body

    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_full_sl_fill_still_cancels_remaining(self, mock_auth, mock_cancel, config, conn):
        """Full SL fill (no partial) still takes the existing full-close path — no regression."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="PLACED",
                          exchange_order_id="tp1_exch1", quantity=0.05)
        save_order(entry, conn)
        save_order(sl, conn)
        save_order(tp1, conn)

        mock_exchange = MagicMock()
        # SL fully filled (filled == quantity)
        mock_exchange.fetch_order.side_effect = lambda oid, sym: (
            {"status": "closed", "average": 59750, "filled": 0.1} if oid == "sl_exch1"
            else {"status": "open"}
        )
        mock_auth.return_value = mock_exchange

        sync_order_status(config, conn)

        # SL filled
        sl_result = get_order(sl["order_id"], conn)
        assert sl_result["status"] == "FILLED"

        # TP1 cancelled (full close path)
        tp1_result = get_order(tp1["order_id"], conn)
        assert tp1_result["status"] == "CANCELLED"

        # No replacement orders placed
        mock_exchange.create_order.assert_not_called()


class TestLivePositionQtyCrossCheck:
    """Replacement orders cross-check qty against exchange position."""

    @patch("scanner.execution.exit_fills.get_open_positions")
    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_replacement_uses_exchange_qty_on_mismatch(self, mock_auth, mock_cancel,
                                                        mock_sleep, mock_positions, config, conn):
        """DB says 0.04 residual, exchange says 0.03 → SL placed at 0.03."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        save_order(entry, conn)
        save_order(sl, conn)

        mock_positions.return_value = [
            {"coin": "BTC", "direction": "LONG", "size": 0.03, "entry_price": 62000,
             "unrealized_pnl": 0, "leverage": 5},
        ]

        mock_exchange = MagicMock()
        # SL partially fills 0.06 of 0.1 → DB residual = 0.04, exchange = 0.03
        mock_exchange.fetch_order.return_value = {
            "status": "closed", "average": 59750, "filled": 0.06,
        }
        mock_exchange.create_order.return_value = {"id": "sl_exch2"}
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange

        sync_order_status(config, conn)

        # New SL placed at exchange qty (0.03), not DB residual (0.04)
        new_sl = _latest_placed(conn, "s1", "stop_loss")
        assert new_sl is not None
        assert new_sl["quantity"] == pytest.approx(0.03)

    @patch("scanner.execution.exit_fills.get_open_positions", side_effect=Exception("API timeout"))
    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_replacement_falls_back_to_db_on_error(self, mock_auth, mock_cancel,
                                                     mock_sleep, mock_positions, config, conn):
        """Exchange position fetch raises → SL placed at DB-derived qty."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        save_order(entry, conn)
        save_order(sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {
            "status": "closed", "average": 59750, "filled": 0.06,
        }
        mock_exchange.create_order.return_value = {"id": "sl_exch2"}
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange

        sync_order_status(config, conn)

        # Falls back to DB residual (0.1 - 0.06 = 0.04)
        new_sl = _latest_placed(conn, "s1", "stop_loss")
        assert new_sl is not None
        assert new_sl["quantity"] == pytest.approx(0.04)

    @patch("scanner.execution.exit_fills.get_open_positions")
    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_replacement_logs_mismatch(self, mock_auth, mock_cancel, mock_sleep,
                                        mock_positions, config, conn, caplog):
        """Qty mismatch >1% → warning logged."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        save_order(entry, conn)
        save_order(sl, conn)

        mock_positions.return_value = [
            {"coin": "BTC", "direction": "LONG", "size": 0.03, "entry_price": 62000,
             "unrealized_pnl": 0, "leverage": 5},
        ]

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {
            "status": "closed", "average": 59750, "filled": 0.06,
        }
        mock_exchange.create_order.return_value = {"id": "sl_exch2"}
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange

        import logging
        with caplog.at_level(logging.WARNING, logger="scanner.execution.monitor"):
            sync_order_status(config, conn)

        assert any("qty mismatch" in r.message for r in caplog.records)


class TestMonitorDiscordAlerts:
    """Monitor CLI sends Discord alerts for missing_sl and phantom_sl."""

    @patch("scanner.execution.executor._send_urgent_alert")
    @patch("scanner.execution.monitor.get_exchange_open_orders", return_value=[])
    @patch("scanner.execution.monitor.sync_order_status", return_value=0)
    def test_missing_sl_sends_discord_alert(self, mock_sync, mock_orders, mock_urgent, config, conn):
        """missing_sl issue triggers Discord urgent alert."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED")
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="CANCELLED")
        save_order(entry, conn)
        save_order(sl, conn)

        # Inline the _cmd_monitor logic with our fixtures
        from scanner.execution.monitor import check_positions

        issues = check_positions(config, conn)
        critical = [i for i in issues if i["type"] in ("missing_sl", "phantom_sl")]
        assert len(critical) == 1

        from scanner.execution.executor import _send_urgent_alert
        for issue in critical:
            _send_urgent_alert(issue["message"], config)

        mock_urgent.assert_called_once()
        assert "no active stop loss" in mock_urgent.call_args[0][0]

    @patch("scanner.execution.executor._send_urgent_alert")
    @patch("scanner.execution.monitor.get_exchange_open_orders", return_value=[])
    @patch("scanner.execution.monitor.sync_order_status", return_value=0)
    def test_phantom_sl_sends_discord_alert(self, mock_sync, mock_orders, mock_urgent, config, conn):
        """phantom_sl issue triggers Discord urgent alert."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED")
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1")
        save_order(entry, conn)
        save_order(sl, conn)

        from scanner.execution.monitor import check_positions

        issues = check_positions(config, conn)
        critical = [i for i in issues if i["type"] in ("missing_sl", "phantom_sl")]
        assert len(critical) == 1

        from scanner.execution.executor import _send_urgent_alert
        for issue in critical:
            _send_urgent_alert(issue["message"], config)

        mock_urgent.assert_called_once()
        assert "DB but not on exchange" in mock_urgent.call_args[0][0]

    @patch("scanner.execution.executor._send_urgent_alert")
    @patch("scanner.execution.monitor.get_exchange_open_orders")
    @patch("scanner.execution.monitor.sync_order_status", return_value=0)
    def test_healthy_position_no_discord_alert(self, mock_sync, mock_orders, mock_urgent, config, conn):
        """Healthy position does not trigger Discord alert."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED")
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1")
        save_order(entry, conn)
        save_order(sl, conn)
        mock_orders.return_value = [{"id": "sl_exch1", "status": "open"}]

        from scanner.execution.monitor import check_positions

        issues = check_positions(config, conn)
        critical = [i for i in issues if i["type"] in ("missing_sl", "phantom_sl")]
        assert len(critical) == 0
        mock_urgent.assert_not_called()


class TestPositionTimeout:
    """enforce_position_timeouts market-closes positions past timeout_hours."""

    def _make_old_entry(self, conn, scan_id="s1", coin="BTC", direction="LONG",
                        hours_ago=80, quantity=0.1):
        """Create a filled entry order that's `hours_ago` hours old."""
        from datetime import datetime, timedelta, timezone
        filled_at = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
        entry = _make_order(
            scan_id=scan_id, coin=coin, direction=direction,
            order_type="entry", status="FILLED", quantity=quantity,
            entry_price=62000, filled_at=filled_at,
        )
        save_order(entry, conn)
        return entry

    @patch("scanner.execution.alerts.requests.post")
    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_happy_path_times_out_old_position(self, mock_auth, mock_cancel, mock_post, config, conn):
        """Entry filled 80h ago (timeout=72) → market close + timeout_close saved + SL cancelled + alert."""
        config.position_timeout_hours = 72
        config.discord_ops_webhook_url = "https://discord.com/ops"
        self._make_old_entry(conn, hours_ago=80)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        save_order(sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.create_order.return_value = {
            "id": "timeout_exch1", "status": "closed", "filled": 0.1, "average": 62500}
        mock_exchange.amount_to_precision.return_value = "0.1"
        mock_auth.return_value = mock_exchange
        mock_post.return_value = MagicMock(status_code=204)

        result = enforce_position_timeouts(config, conn)
        assert result == 1

        # timeout_close order saved
        scan_orders = get_orders_for_scan("s1", conn)
        timeout_orders = [o for o in scan_orders if o["order_type"] == "timeout_close"]
        assert len(timeout_orders) == 1
        assert timeout_orders[0]["status"] == "FILLED"

        # SL cancelled
        sl_result = get_order(sl["order_id"], conn)
        assert sl_result["status"] == "CANCELLED"

        # Discord embed sent with R value
        mock_post.assert_called()
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert "TIMEOUT" in embed["title"]
        assert "BTC" in embed["title"]
        field_values = {f["name"]: f["value"] for f in embed["fields"]}
        # entry=62000, fill=62500, sl=59800 → LONG: (62500-62000)/(62000-59800) = +0.23R
        assert "+0.23R" in field_values["R Value"]

    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_fresh_position_skipped(self, mock_auth, config, conn):
        """Entry filled 24h ago (timeout=72) → not timed out."""
        config.position_timeout_hours = 72
        self._make_old_entry(conn, hours_ago=24)

        mock_exchange = MagicMock()
        mock_auth.return_value = mock_exchange

        result = enforce_position_timeouts(config, conn)
        assert result == 0
        mock_exchange.create_order.assert_not_called()

    @patch("scanner.execution.alerts.requests.post")
    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_remaining_qty_after_tp1(self, mock_auth, mock_cancel, mock_post, config, conn):
        """Entry 0.1, TP1 filled 0.05 → timeout closes remaining 0.05."""
        config.position_timeout_hours = 72
        self._make_old_entry(conn, hours_ago=80, quantity=0.1)
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="FILLED", quantity=0.05)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.05)
        save_order(tp1, conn)
        save_order(sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.create_order.return_value = {
            "id": "timeout_exch1", "status": "closed", "filled": 0.05, "average": 63000}
        mock_exchange.amount_to_precision.return_value = "0.05"
        mock_auth.return_value = mock_exchange
        mock_post.return_value = MagicMock(status_code=204)

        result = enforce_position_timeouts(config, conn)
        assert result == 1

        # Verify it closed 0.05, not 0.1
        call_args = mock_exchange.create_order.call_args
        assert call_args[0][3] == 0.05  # qty argument

    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_retry_success(self, mock_auth, mock_cancel, mock_sleep, config, conn):
        """Market close fails twice, succeeds third → order saved."""
        config.position_timeout_hours = 72
        self._make_old_entry(conn, hours_ago=80)

        mock_exchange = MagicMock()
        mock_exchange.create_order.side_effect = [
            Exception("timeout"), Exception("timeout"),
            {"id": "timeout_exch1", "status": "closed", "filled": 0.1, "average": 62000},
        ]
        mock_exchange.amount_to_precision.return_value = "0.1"
        mock_auth.return_value = mock_exchange

        result = enforce_position_timeouts(config, conn)
        assert result == 1
        assert mock_exchange.create_order.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("scanner.execution.alerts.requests.post")
    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_all_retries_fail_sends_urgent(self, mock_auth, mock_sleep, mock_post, config, conn):
        """All 3 retries fail → no order saved, urgent alert sent."""
        config.position_timeout_hours = 72
        config.discord_urgent_webhook_url = "https://discord.com/urgent"
        self._make_old_entry(conn, hours_ago=80)

        mock_exchange = MagicMock()
        mock_exchange.create_order.side_effect = Exception("exchange down")
        mock_exchange.amount_to_precision.return_value = "0.1"
        mock_auth.return_value = mock_exchange
        mock_post.return_value = MagicMock(status_code=204)

        result = enforce_position_timeouts(config, conn)
        assert result == 0

        # No timeout_close order saved
        scan_orders = get_orders_for_scan("s1", conn)
        timeout_orders = [o for o in scan_orders if o["order_type"] == "timeout_close"]
        assert len(timeout_orders) == 0

        # Urgent alert sent
        mock_post.assert_called()
        call_body = mock_post.call_args[1]["json"]["content"]
        assert "URGENT" in call_body

    def test_disabled_returns_zero(self, config, conn):
        """timeout=0 → returns 0, no exchange calls."""
        config.position_timeout_hours = 0
        result = enforce_position_timeouts(config, conn)
        assert result == 0

    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_uses_filled_at_for_age(self, mock_auth, config, conn):
        """Age calculated from filled_at, not created_at."""
        from datetime import datetime, timedelta, timezone
        config.position_timeout_hours = 72

        # created_at is 100h ago, but filled_at is only 24h ago
        created = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
        filled = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        entry = _make_order(
            scan_id="s1", order_type="entry", status="FILLED",
            quantity=0.1, entry_price=62000,
            created_at=created, filled_at=filled,
        )
        save_order(entry, conn)

        mock_exchange = MagicMock()
        mock_auth.return_value = mock_exchange

        result = enforce_position_timeouts(config, conn)
        assert result == 0  # 24h < 72h, not timed out
        mock_exchange.create_order.assert_not_called()


class TestCalcR:
    """R (risk multiple) calculation."""

    def test_long_winner(self):
        """LONG: entry=100, exit=110, sl=95 → +2.0R."""
        assert _calc_r(100.0, 110.0, 95.0, "LONG") == pytest.approx(2.0)

    def test_long_loser(self):
        """LONG: entry=100, exit=95, sl=95 → -1.0R."""
        assert _calc_r(100.0, 95.0, 95.0, "LONG") == pytest.approx(-1.0)

    def test_short_winner(self):
        """SHORT: entry=100, exit=90, sl=105 → +2.0R."""
        assert _calc_r(100.0, 90.0, 105.0, "SHORT") == pytest.approx(2.0)

    def test_short_loser(self):
        """SHORT: entry=100, exit=105, sl=105 → -1.0R."""
        assert _calc_r(100.0, 105.0, 105.0, "SHORT") == pytest.approx(-1.0)

    def test_zero_risk_returns_none(self):
        """SL at entry price → zero risk → None."""
        assert _calc_r(100.0, 110.0, 100.0, "LONG") is None


class TestFillAlert:
    """SL/TP fill Discord embed notifications."""

    @patch("scanner.execution.alerts.requests.post")
    def test_fill_alert_sl(self, mock_post, conn):
        """SL fill, no original-SL row in DB → fallback to the just-filled
        SL's trigger (KL#2(iv): re-point preserves the SL-fill fallback)."""
        mock_post.return_value = MagicMock(status_code=204)
        config = ExecutionConfig(discord_ops_webhook_url="https://discord.com/ops")
        entry = {"coin": "BTC", "direction": "LONG", "entry_price": 62000}
        order = {"order_type": "stop_loss", "trigger_price": 59800}

        result = _send_fill_alert(entry, order, 59800.0, config, conn)

        assert result is True
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert "SL FILLED" in embed["title"]
        assert "BTC" in embed["title"]
        assert embed["color"] == 0xEF5350  # red
        field_values = {f["name"]: f["value"] for f in embed["fields"]}
        assert "59,800.00" in field_values["Fill Price"]
        assert "-1.00R" in field_values["R Value"]

    @patch("scanner.execution.alerts.requests.post")
    def test_fill_alert_tp(self, mock_post, conn):
        """TP1 fill → green embed, R value from the DB original-SL anchor
        (KL#2(iv): single owner storage.original_stop_price)."""
        mock_post.return_value = MagicMock(status_code=204)
        config = ExecutionConfig(discord_ops_webhook_url="https://discord.com/ops")
        save_order(_make_order(scan_id="s_tp", coin="ETH", direction="SHORT",
                               order_type="stop_loss", status="PLACED",
                               trigger_price=3500), conn)
        entry = {"scan_id": "s_tp", "coin": "ETH", "direction": "SHORT", "entry_price": 3400}
        order = {"order_type": "tp1"}

        result = _send_fill_alert(entry, order, 3200.50, config, conn)

        assert result is True
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert "TP1 FILLED" in embed["title"]
        assert "ETH" in embed["title"]
        assert embed["color"] == 0x26A69A  # green
        field_values = {f["name"]: f["value"] for f in embed["fields"]}
        assert "+2.00R" in field_values["R Value"]

    @patch("scanner.execution.alerts.requests.post")
    def test_fill_alert_tp_no_original_sl(self, mock_post, conn):
        """TP fill, no original-SL row in DB → no R field, no crash (TP has
        no fallback — only SL fills fall back to the just-filled trigger)."""
        mock_post.return_value = MagicMock(status_code=204)
        config = ExecutionConfig(discord_ops_webhook_url="https://discord.com/ops")
        entry = {"coin": "ETH", "direction": "SHORT", "entry_price": 3400}
        order = {"order_type": "tp1"}

        result = _send_fill_alert(entry, order, 3200.50, config, conn)

        assert result is True
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert "TP1 FILLED" in embed["title"]
        field_names = [f["name"] for f in embed["fields"]]
        assert "R Value" not in field_names

    @patch("scanner.execution.alerts.requests.post")
    def test_fill_alert_sl_drift_flatten_uses_original_sl(self, mock_post, conn):
        """Drift-flatten case: SL was moved to entry, then filled at entry.
        Without using the original SL as risk anchor, risk = 0 → R=None →
        R Value field omitted from alert (XRP/KBONK 2026-05-01 pattern).
        With the fix, R reports +0.00R (correct breakeven outcome)."""
        mock_post.return_value = MagicMock(status_code=204)
        config = ExecutionConfig(discord_ops_webhook_url="https://discord.com/ops")
        entry = {"scan_id": "s_drift", "coin": "XRP", "direction": "LONG", "entry_price": 1.3704}
        # Order that just filled is the post-drift SL (trigger == entry)
        order = {"order_type": "stop_loss", "trigger_price": 1.3704}
        # DB order chain: original SL (1.3439, earlier created_at) then the
        # drift-moved SL (1.3704). storage.original_stop_price ORDER BY
        # created_at ASC LIMIT 1 → the ORIGINAL 1.3439 (the risk anchor).
        save_order(_make_order(scan_id="s_drift", coin="XRP", order_type="stop_loss",
                               status="CANCELLED", trigger_price=1.3439,
                               created_at="2026-04-30T12:06:40+00:00"), conn)
        save_order(_make_order(scan_id="s_drift", coin="XRP", order_type="stop_loss",
                               status="PLACED", trigger_price=1.3704,
                               created_at="2026-05-01T00:10:51+00:00"), conn)

        result = _send_fill_alert(entry, order, 1.3704, config, conn)

        assert result is True
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        field_values = {f["name"]: f["value"] for f in embed["fields"]}
        assert "R Value" in field_values
        assert "+0.00R" in field_values["R Value"]

    @patch("scanner.execution.alerts.requests.post")
    def test_fill_alert_no_entry_price(self, mock_post, conn):
        """No entry_price on entry dict → no R field, no crash."""
        mock_post.return_value = MagicMock(status_code=204)
        config = ExecutionConfig(discord_ops_webhook_url="https://discord.com/ops")
        entry = {"coin": "BTC", "direction": "LONG"}
        order = {"order_type": "stop_loss", "trigger_price": 59800}

        result = _send_fill_alert(entry, order, 59800.0, config, conn)

        assert result is True
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert "SL FILLED" in embed["title"]
        field_names = [f["name"] for f in embed["fields"]]
        assert "R Value" not in field_names

    def test_fill_alert_no_webhook(self, conn):
        """No ops webhook → returns False, no request."""
        config = ExecutionConfig(discord_ops_webhook_url="")
        entry = {"coin": "BTC", "direction": "LONG"}
        order = {"order_type": "stop_loss"}

        result = _send_fill_alert(entry, order, 59800.0, config, conn)
        assert result is False


class TestReplacementOrderTypes:
    """Replacement orders (TP1 fill, partial fill) use type='market'."""

    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_tp1_sl_replacement_uses_market_type(self, mock_auth, mock_cancel, mock_sleep,
                                                   config, conn):
        """SL replacement after TP1 fill uses type='market'."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="PLACED",
                          exchange_order_id="tp1_exch1", quantity=0.05)
        save_order(entry, conn)
        save_order(sl, conn)
        save_order(tp1, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.side_effect = lambda oid, sym: (
            {"status": "closed", "average": 65000} if oid == "tp1_exch1"
            else {"status": "open"}
        )
        mock_exchange.amount_to_precision.return_value = "0.05"
        mock_exchange.create_order.return_value = {"id": "sl_exch2"}
        mock_auth.return_value = mock_exchange

        sync_order_status(config, conn)

        # SL replacement call should use "market" type
        call_args = mock_exchange.create_order.call_args
        assert call_args[0][1] == "market"
        assert call_args[0][5]["stopLossPrice"] == 59800

    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.exit_fills.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_partial_sl_replacement_uses_market_type(self, mock_auth, mock_cancel, mock_sleep,
                                                       config, conn):
        """SL + TP2 replacements after partial SL fill both use type='market'."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        tp2 = _make_order(scan_id="s1", order_type="tp2", status="PLACED",
                          exchange_order_id="tp2_exch1", trigger_price=68000, quantity=0.1)
        save_order(entry, conn)
        save_order(sl, conn)
        save_order(tp2, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.side_effect = lambda oid, sym: (
            {"status": "closed", "average": 59750, "filled": 0.06} if oid == "sl_exch1"
            else {"status": "open"}
        )
        mock_exchange.create_order.side_effect = [
            {"id": "sl_exch2"},
            {"id": "tp2_exch2"},
        ]
        mock_exchange.amount_to_precision.return_value = "0.04"
        mock_auth.return_value = mock_exchange

        sync_order_status(config, conn)

        # Both replacement calls should use "market"
        assert mock_exchange.create_order.call_count == 2
        for call in mock_exchange.create_order.call_args_list:
            assert call[0][1] == "market"


class TestRejectedExitOrderReplacement:
    """Auto-replace rejected/cancelled/expired exit orders."""

    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_rejected_sl_gets_replaced(self, mock_auth, mock_sleep, config, conn):
        """SL rejected on exchange → auto-replaced with same trigger."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        save_order(entry, conn)
        save_order(sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {"status": "rejected"}
        mock_exchange.fetch_positions.return_value = _open_position()
        mock_exchange.amount_to_precision.return_value = "0.1"
        mock_exchange.create_order.return_value = {"id": "sl_exch2"}
        mock_auth.return_value = mock_exchange

        sync_order_status(config, conn)

        # Old SL row CANCELLED; replacement inserted as new PLACED row
        assert get_order(sl["order_id"], conn)["status"] == "CANCELLED"
        new_sl = _latest_placed(conn, "s1", "stop_loss")
        assert new_sl is not None
        assert new_sl["exchange_order_id"] == "sl_exch2"
        # Verify it used stopLossPrice param
        call_args = mock_exchange.create_order.call_args
        assert call_args[0][5]["stopLossPrice"] == 59800.0

    @patch("scanner.execution.alerts.requests.post")
    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_rejected_tp1_not_replaced(self, mock_auth, mock_sleep, mock_post, config, conn):
        """TP1 rejected → stays CANCELLED, no replacement attempted."""
        config.discord_errors_webhook_url = "https://discord.com/errors"
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="PLACED",
                          exchange_order_id="tp1_exch1", trigger_price=65000, quantity=0.05)
        save_order(entry, conn)
        save_order(tp1, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {"status": "rejected"}
        mock_auth.return_value = mock_exchange
        mock_post.return_value = MagicMock(status_code=204)

        sync_order_status(config, conn)

        tp1_result = get_order(tp1["order_id"], conn)
        assert tp1_result["status"] == "CANCELLED"
        mock_exchange.create_order.assert_not_called()

    @patch("scanner.execution.alerts.requests.post")
    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_expired_tp2_not_replaced(self, mock_auth, mock_sleep, mock_post, config, conn):
        """TP2 expired → stays CANCELLED, no replacement attempted."""
        config.discord_errors_webhook_url = "https://discord.com/errors"
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        # TP1 already filled
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="FILLED", quantity=0.05)
        tp2 = _make_order(scan_id="s1", order_type="tp2", status="PLACED",
                          exchange_order_id="tp2_exch1", trigger_price=68000, quantity=0.05)
        save_order(entry, conn)
        save_order(tp1, conn)
        save_order(tp2, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {"status": "expired"}
        mock_auth.return_value = mock_exchange
        mock_post.return_value = MagicMock(status_code=204)

        sync_order_status(config, conn)

        tp2_result = get_order(tp2["order_id"], conn)
        assert tp2_result["status"] == "CANCELLED"
        mock_exchange.create_order.assert_not_called()

    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_canceled_zero_fill_sl_gets_replaced(self, mock_auth, mock_sleep, config, conn):
        """SL canceled (zero fill) → new replacement row inserted."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        save_order(entry, conn)
        save_order(sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {"status": "canceled", "filled": 0}
        mock_exchange.fetch_positions.return_value = _open_position()
        mock_exchange.amount_to_precision.return_value = "0.1"
        mock_exchange.create_order.return_value = {"id": "sl_exch2"}
        mock_auth.return_value = mock_exchange

        sync_order_status(config, conn)

        assert get_order(sl["order_id"], conn)["status"] == "CANCELLED"
        new_sl = _latest_placed(conn, "s1", "stop_loss")
        assert new_sl is not None
        assert new_sl["exchange_order_id"] == "sl_exch2"

    @patch("scanner.execution.alerts.requests.post")
    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_rejected_sl_replacement_fails_sends_urgent(self, mock_auth, mock_sleep,
                                                          mock_post, config, conn):
        """SL rejected, all replacement retries fail → URGENT alert sent."""
        config.discord_urgent_webhook_url = "https://discord.com/urgent"
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        save_order(entry, conn)
        save_order(sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {"status": "rejected"}
        mock_exchange.fetch_positions.return_value = _open_position()
        mock_exchange.amount_to_precision.return_value = "0.1"
        mock_exchange.create_order.side_effect = Exception("exchange down")
        mock_auth.return_value = mock_exchange
        mock_post.return_value = MagicMock(status_code=204)

        sync_order_status(config, conn)

        # URGENT alert sent for SL failure
        mock_post.assert_called()
        call_body = mock_post.call_args[1]["json"]["content"]
        assert "URGENT" in call_body
        assert "BTC" in call_body

    @patch("scanner.execution.alerts.requests.post")
    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_rejected_tp_sends_errors_alert(self, mock_auth, mock_sleep,
                                             mock_post, config, conn):
        """TP1 rejected → errors alert sent immediately, no replacement."""
        config.discord_errors_webhook_url = "https://discord.com/errors"
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="PLACED",
                          exchange_order_id="tp1_exch1", trigger_price=65000, quantity=0.05)
        save_order(entry, conn)
        save_order(tp1, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {"status": "rejected"}
        mock_exchange.fetch_positions.return_value = _open_position()
        mock_auth.return_value = mock_exchange
        mock_post.return_value = MagicMock(status_code=204)

        sync_order_status(config, conn)

        # Errors channel alert sent, no replacement attempted
        mock_post.assert_called()
        call_body = mock_post.call_args[1]["json"]["content"]
        assert "TP1 rejected by exchange" in call_body
        assert "BTC" in call_body
        mock_exchange.create_order.assert_not_called()

    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_rejected_entry_not_replaced(self, mock_auth, config, conn):
        """Non-exit order (entry) rejected → stays CANCELLED, no replacement."""
        # Need a FILLED entry so sync_order_status picks up the scan
        filled_entry = _make_order(scan_id="s1", order_type="entry", status="FILLED",
                                   exchange_order_id="entry_exch1", quantity=0.1)
        # Simulate a second PLACED "entry" order that gets rejected (edge case)
        extra = _make_order(scan_id="s1", order_type="entry", status="PLACED",
                            exchange_order_id="extra_exch1", quantity=0.1)
        save_order(filled_entry, conn)
        save_order(extra, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.side_effect = lambda oid, sym: (
            {"status": "rejected"} if oid == "extra_exch1"
            else {"status": "closed", "average": 62000}
        )
        mock_auth.return_value = mock_exchange

        sync_order_status(config, conn)

        extra_result = get_order(extra["order_id"], conn)
        assert extra_result["status"] == "CANCELLED"
        # No replacement attempted for non-exit order
        mock_exchange.create_order.assert_not_called()

    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_rejected_sl_skipped_if_position_closed(self, mock_auth, mock_sleep, config, conn):
        """SL rejected but position already fully closed → skip replacement.

        get_open_entry_orders excludes fully-closed positions (entry qty == filled exit qty),
        so the SL rejection is never even checked. Verify no replacement attempted.
        """
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        # TP2 already filled the full position
        tp2 = _make_order(scan_id="s1", order_type="tp2", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        save_order(entry, conn)
        save_order(tp2, conn)
        save_order(sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {"status": "rejected"}
        mock_auth.return_value = mock_exchange

        sync_order_status(config, conn)

        # SL stays PLACED — fully closed position not visited by sync_order_status
        sl_result = get_order(sl["order_id"], conn)
        assert sl_result["status"] == "PLACED"
        mock_exchange.create_order.assert_not_called()
        # fetch_order never called — entry excluded from open_entries
        mock_exchange.fetch_order.assert_not_called()

    @patch("scanner.execution.alerts.requests.post")
    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_tp_cancel_after_sl_fill_skips_alert(self, mock_auth, mock_sleep,
                                                  mock_post, config, conn):
        """SL fill auto-cancels TPs on exchange — DB not yet synced → no false alert.

        Regression for 2026-04-19 XRP: drift moved SL to BE, price triggered the
        new SL, HL auto-cancelled reduceOnly TPs on position close. sync_order_status
        visited the TPs before the SL fill was recorded in the DB (TPs have earlier
        created_at); DB remaining_qty was still > 0 but the position was flat.
        Verify the exchange position check skips the rejection alert.
        """
        config.discord_errors_webhook_url = "https://discord.com/errors"
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="PLACED",
                          exchange_order_id="tp1_exch1", trigger_price=65000, quantity=0.05)
        save_order(entry, conn)
        save_order(tp1, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {"status": "canceled", "filled": 0}
        # Position closed on exchange — no entry in fetch_positions
        mock_exchange.fetch_positions.return_value = []
        mock_auth.return_value = mock_exchange

        sync_order_status(config, conn)

        tp1_result = get_order(tp1["order_id"], conn)
        assert tp1_result["status"] == "CANCELLED"
        # No TP rejection alert — position was already closed
        mock_post.assert_not_called()
        mock_exchange.create_order.assert_not_called()


class TestGetRemainingQtyPrecision:
    """IEEE 754 floating-point residuals must clamp to 0.0."""

    def test_clamps_fp_residual_to_zero(self, conn):
        """BCH scenario: 0.043 + 0.042 must yield remaining_qty == 0.0, not 1.4e-17."""
        entry = _make_order(scan_id="bch1", order_type="entry", status="FILLED", quantity=0.085)
        tp1 = _make_order(scan_id="bch1", order_type="tp1", status="FILLED", quantity=0.043)
        timeout = _make_order(scan_id="bch1", order_type="timeout_close", status="FILLED", quantity=0.042)
        save_order(entry, conn)
        save_order(tp1, conn)
        save_order(timeout, conn)

        remaining = get_remaining_qty(entry, conn)
        assert remaining == 0.0

    def test_genuine_partial_returns_remainder(self, conn):
        """Real partial exit returns actual remaining quantity."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        tp1 = _make_order(scan_id="s1", order_type="tp1", status="FILLED", quantity=0.05)
        save_order(entry, conn)
        save_order(tp1, conn)

        remaining = get_remaining_qty(entry, conn)
        assert remaining == pytest.approx(0.05)

    def test_emergency_close_reduces_remaining(self, conn):
        """Emergency close is counted as a filled exit."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        ec = _make_order(scan_id="s1", order_type="emergency_close", status="FILLED", quantity=0.1)
        save_order(entry, conn)
        save_order(ec, conn)

        remaining = get_remaining_qty(entry, conn)
        assert remaining == 0.0


class TestAttemptSlRecovery:
    """Tests for attempt_sl_recovery — monitor places missing SL."""

    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_recovery_with_failed_sl_record(self, mock_auth, config, conn):
        """FAILED SL in DB -> recovery uses its trigger_price."""
        from scanner.execution.monitor import attempt_sl_recovery

        entry = _make_order(scan_id="scan1", order_type="entry", status="FILLED",
                            quantity=0.1, fill_price=62000)
        failed_sl = _make_order(scan_id="scan1", order_type="stop_loss", status="FAILED",
                                trigger_price=59800, quantity=0.1)
        save_order(entry, conn)
        save_order(failed_sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.create_order.return_value = {"id": "new-sl-123"}
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange

        issue = {"scan_id": "scan1", "coin": "BTC", "direction": "LONG",
                 "type": "missing_sl", "message": "test"}
        result = attempt_sl_recovery(issue, config, conn)

        assert result is True
        orders = get_orders_for_scan("scan1", conn)
        placed_sls = [o for o in orders if o["order_type"] == "stop_loss" and o["status"] == "PLACED"]
        assert len(placed_sls) == 1
        assert placed_sls[0]["trigger_price"] == 59800

        # Old FAILED SL marked CANCELLED
        failed_sls = [o for o in orders if o["order_type"] == "stop_loss" and o["status"] == "CANCELLED"]
        assert len(failed_sls) == 1

    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_recovery_fails_all_retries(self, mock_auth, config, conn):
        """SL placement fails 3x -> returns False."""
        from scanner.execution.monitor import attempt_sl_recovery

        entry = _make_order(scan_id="scan1", order_type="entry", status="FILLED",
                            quantity=0.1, fill_price=62000)
        failed_sl = _make_order(scan_id="scan1", order_type="stop_loss", status="FAILED",
                                trigger_price=59800, quantity=0.1)
        save_order(entry, conn)
        save_order(failed_sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.create_order.side_effect = Exception("rate limit")
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange

        issue = {"scan_id": "scan1", "coin": "BTC", "direction": "LONG",
                 "type": "missing_sl", "message": "test"}
        result = attempt_sl_recovery(issue, config, conn)

        assert result is False
        # No PLACED SL in DB
        orders = get_orders_for_scan("scan1", conn)
        placed_sls = [o for o in orders if o["order_type"] == "stop_loss" and o["status"] == "PLACED"]
        assert len(placed_sls) == 0

    @patch("scanner.execution.exchange.get_open_positions", return_value=[])
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_recovery_records_manual_close_when_position_gone(
        self, mock_auth, mock_positions, config, conn,
    ):
        """If the exchange has no matching position, recovery records a
        manual_close instead of looping on reduceOnly rejections."""
        from scanner.execution.monitor import attempt_sl_recovery

        entry = _make_order(scan_id="scan1", order_type="entry", status="FILLED",
                            quantity=0.1, fill_price=62000)
        failed_sl = _make_order(scan_id="scan1", order_type="stop_loss",
                                status="FAILED",
                                trigger_price=59800, quantity=0.1)
        save_order(entry, conn)
        save_order(failed_sl, conn)

        issue = {"scan_id": "scan1", "coin": "BTC", "direction": "LONG",
                 "type": "missing_sl", "message": "test"}
        result = attempt_sl_recovery(issue, config, conn)

        assert result is True
        orders = get_orders_for_scan("scan1", conn)
        manual_closes = [o for o in orders if o["order_type"] == "manual_close"]
        assert len(manual_closes) == 1
        assert manual_closes[0]["status"] == "FILLED"
        assert manual_closes[0]["quantity"] == 0.1
        # No new PLACED SL was created (we didn't try to place one)
        placed_sls = [o for o in orders if o["order_type"] == "stop_loss" and o["status"] == "PLACED"]
        assert len(placed_sls) == 0

    def test_recovery_position_already_closed(self, config, conn):
        """Position fully exited -> recovery returns True (no action needed)."""
        from scanner.execution.monitor import attempt_sl_recovery

        entry = _make_order(scan_id="scan1", order_type="entry", status="FILLED",
                            quantity=0.1, fill_price=62000)
        ec = _make_order(scan_id="scan1", order_type="emergency_close", status="FILLED",
                         quantity=0.1)
        save_order(entry, conn)
        save_order(ec, conn)

        issue = {"scan_id": "scan1", "coin": "BTC", "direction": "LONG",
                 "type": "missing_sl", "message": "test"}
        result = attempt_sl_recovery(issue, config, conn)
        assert result is True

    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_recovery_supersedes_phantom_placed_sl(self, mock_auth, config, conn):
        """phantom_sl: PLACED SL row gets marked CANCELLED, fresh SL inserted."""
        from scanner.execution.monitor import attempt_sl_recovery

        entry = _make_order(scan_id="scan1", order_type="entry", status="FILLED",
                            quantity=0.1, fill_price=62000)
        phantom_sl = _make_order(scan_id="scan1", order_type="stop_loss",
                                 status="PLACED",
                                 exchange_order_id="stale_sl_exch",
                                 trigger_price=59800, quantity=0.1)
        save_order(entry, conn)
        save_order(phantom_sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.create_order.return_value = {"id": "new-sl-456"}
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange

        issue = {"scan_id": "scan1", "coin": "BTC", "direction": "LONG",
                 "type": "phantom_sl", "message": "test"}
        result = attempt_sl_recovery(issue, config, conn)

        assert result is True
        orders = get_orders_for_scan("scan1", conn)
        # Old phantom SL now CANCELLED
        assert get_order(phantom_sl["order_id"], conn)["status"] == "CANCELLED"
        # Exactly one PLACED SL — the replacement
        placed_sls = [o for o in orders if o["order_type"] == "stop_loss" and o["status"] == "PLACED"]
        assert len(placed_sls) == 1
        assert placed_sls[0]["exchange_order_id"] == "new-sl-456"
        assert placed_sls[0]["trigger_price"] == 59800

    def test_recovery_no_trigger_price(self, config, conn):
        """No FAILED SL and no scan data -> returns False."""
        from scanner.execution.monitor import attempt_sl_recovery

        entry = _make_order(scan_id="scan1", order_type="entry", status="FILLED",
                            quantity=0.1, fill_price=62000)
        save_order(entry, conn)

        issue = {"scan_id": "scan1", "coin": "BTC", "direction": "LONG",
                 "type": "missing_sl", "message": "test"}
        result = attempt_sl_recovery(issue, config, conn)
        assert result is False

    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_recovery_rate_limited_breaks_immediately(self, mock_auth, config, conn):
        """Rate limit on first attempt -> breaks immediately, returns False."""
        from scanner.execution.monitor import attempt_sl_recovery

        entry = _make_order(scan_id="scan1", order_type="entry", status="FILLED",
                            quantity=0.1, fill_price=62000)
        failed_sl = _make_order(scan_id="scan1", order_type="stop_loss", status="FAILED",
                                trigger_price=59800, quantity=0.1)
        save_order(entry, conn)
        save_order(failed_sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.create_order.side_effect = Exception(
            "Too many cumulative requests sent (12530 > 12268)"
        )
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange

        issue = {"scan_id": "scan1", "coin": "BTC", "direction": "LONG",
                 "type": "missing_sl", "message": "test"}
        result = attempt_sl_recovery(issue, config, conn)

        assert result is False
        # Only 1 attempt, not 3 — rate limit broke retry loop
        assert mock_exchange.create_order.call_count == 1
        # No PLACED SL in DB
        orders = get_orders_for_scan("scan1", conn)
        placed_sls = [o for o in orders if o["order_type"] == "stop_loss" and o["status"] == "PLACED"]
        assert len(placed_sls) == 0


class TestRateLimitInReplacement:
    """Rate limit errors abort retries immediately to conserve API budget."""

    @patch("scanner.execution.alerts.requests.post")
    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_sl_replacement_rate_limited_breaks_and_alerts(self, mock_auth, mock_sleep,
                                                            mock_post, config, conn):
        """SL replacement rate-limited → 1 attempt, URGENT alert, no further retries."""
        config.discord_urgent_webhook_url = "https://discord.com/urgent"
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        save_order(entry, conn)
        save_order(sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {"status": "rejected"}
        mock_exchange.fetch_positions.return_value = _open_position()
        mock_exchange.amount_to_precision.return_value = "0.1"
        mock_exchange.create_order.side_effect = Exception(
            "Too many cumulative requests sent (12530 > 12268)"
        )
        mock_auth.return_value = mock_exchange
        mock_post.return_value = MagicMock(status_code=204)

        sync_order_status(config, conn)

        # Only 1 replacement attempt (not 3)
        assert mock_exchange.create_order.call_count == 1
        # No sleep calls — broke immediately
        mock_sleep.assert_not_called()
        # URGENT alert sent
        mock_post.assert_called()
        call_body = mock_post.call_args[1]["json"]["content"]
        assert "URGENT" in call_body
