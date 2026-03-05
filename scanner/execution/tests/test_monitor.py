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
    def test_check_positions_exchange_error_skips_verification(self, mock_orders, config, conn):
        """Exchange raises → still checks DB, no phantom_sl issues."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED")
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1")
        save_order(entry, conn)
        save_order(sl, conn)

        issues = check_positions(config, conn)
        assert len(issues) == 0  # SL exists in DB, exchange check skipped

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


class TestStaleListSafety:
    """Exit fill breaks the inner loop to avoid stale-list corruption."""

    @patch("scanner.execution.monitor.cancel_order", return_value=True)
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

        # SL should have replacement exchange_order_id, NOT be marked CANCELLED
        sl_result = get_order(sl["order_id"], conn)
        assert sl_result["exchange_order_id"] == "sl_exch2"
        assert sl_result["status"] != "CANCELLED"


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

    @patch("scanner.execution.monitor.cancel_order", return_value=True)
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

        # SL replaced with remaining = 0.1 - 0.03 = 0.07
        sl_result = get_order(sl["order_id"], conn)
        assert sl_result["quantity"] == 0.07
        assert sl_result["exchange_order_id"] == "sl_exch2"

    @patch("scanner.execution.monitor.cancel_order", return_value=True)
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

        sync_order_status(config, conn)

        # SL should be FILLED with actual fill quantity
        sl_result = get_order(sl["order_id"], conn)
        assert sl_result["status"] == "FILLED"
        assert sl_result["quantity"] == 0.08
        assert sl_result["fill_price"] == 59750

        # TP1 should be cancelled (SL = full close path)
        tp1_result = get_order(tp1["order_id"], conn)
        assert tp1_result["status"] == "CANCELLED"

    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_canceled_with_zero_fill_stays_cancelled(self, mock_auth, config, conn):
        """SL canceled with filled=0 → DB shows CANCELLED. Existing behavior preserved."""
        entry = _make_order(scan_id="s1", order_type="entry", status="FILLED", quantity=0.1)
        sl = _make_order(scan_id="s1", order_type="stop_loss", status="PLACED",
                         exchange_order_id="sl_exch1", trigger_price=59800, quantity=0.1)
        save_order(entry, conn)
        save_order(sl, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_order.return_value = {"status": "canceled", "filled": 0}
        mock_auth.return_value = mock_exchange

        updated = sync_order_status(config, conn)
        assert updated == 1

        sl_result = get_order(sl["order_id"], conn)
        assert sl_result["status"] == "CANCELLED"
        assert sl_result["fill_price"] is None

    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.monitor.cancel_order", return_value=True)
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

        # SL should be PLACED with new exchange order ID
        sl_result = get_order(sl["order_id"], conn)
        assert sl_result["status"] == "PLACED"
        assert sl_result["exchange_order_id"] == "sl_exch2"
        assert mock_exchange.create_order.call_count == 3
        assert mock_sleep.call_count == 2  # backoff between retries

    @patch("scanner.execution.monitor.requests.post")
    @patch("scanner.execution.monitor.time.sleep")
    @patch("scanner.execution.monitor.cancel_order", return_value=True)
    @patch("scanner.execution.monitor.get_auth_exchange")
    def test_tp1_sl_replacement_exhausted_sends_urgent(self, mock_auth, mock_cancel,
                                                        mock_sleep, mock_post, config, conn):
        """All 3 SL replacement attempts fail → URGENT Discord alert sent."""
        config.discord_execution_webhook_url = "https://discord.com/test"
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

        # SL should be CANCELLED in DB — old one is gone, all retries failed
        sl_result = get_order(sl["order_id"], conn)
        assert sl_result["status"] == "CANCELLED"

        # URGENT alert should have been sent
        mock_post.assert_called_once()
        call_body = mock_post.call_args[1]["json"]["content"]
        assert "URGENT" in call_body
        assert "BTC" in call_body

        # check_positions() should flag the missing SL
        issues = check_positions(config, conn)
        assert any(i["type"] == "missing_sl" and i["coin"] == "BTC" for i in issues)
