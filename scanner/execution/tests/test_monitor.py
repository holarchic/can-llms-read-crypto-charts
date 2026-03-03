"""Tests for position monitoring + order status sync."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from scanner.execution.config import ExecutionConfig
from scanner.execution.monitor import check_positions, sync_order_status
from scanner.execution.storage import (
    ensure_orders_table,
    get_order,
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

    @patch("scanner.execution.monitor.cancel_order", return_value=True)
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
    def test_detects_missing_sl(self, config, conn):
        """Open position without active SL -> issue reported."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED")
        # SL was cancelled — no active SL
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="CANCELLED")
        save_order(entry, conn)
        save_order(sl, conn)

        issues = check_positions(config, conn)
        assert len(issues) == 1
        assert issues[0]["type"] == "missing_sl"

    def test_healthy_position(self, config, conn):
        """Position with active SL -> no issues."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED")
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED")
        save_order(entry, conn)
        save_order(sl, conn)

        issues = check_positions(config, conn)
        assert len(issues) == 0

    def test_no_positions(self, config, conn):
        """No open positions -> no issues."""
        issues = check_positions(config, conn)
        assert len(issues) == 0


class TestPartialExitHandling:
    @patch("scanner.execution.monitor.cancel_order", return_value=True)
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

        # SL should have updated quantity and new exchange order ID
        sl_result = get_order(sl["order_id"], conn)
        assert sl_result["quantity"] == 0.05
        assert sl_result["exchange_order_id"] == "sl_exch2"

        # TP2 should still be PLACED (not cancelled)
        tp2_result = get_order(tp2["order_id"], conn)
        assert tp2_result["status"] == "PLACED"

        # Old SL should have been cancelled on exchange
        mock_cancel.assert_called_once()

    @patch("scanner.execution.monitor.cancel_order", return_value=True)
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

    @patch("scanner.execution.monitor.cancel_order", return_value=True)
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
