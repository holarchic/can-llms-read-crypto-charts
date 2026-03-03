"""Tests for execution analytics."""

import json
import sqlite3

import pytest

from scanner.execution.analytics import (
    TradeRecord,
    compare_sim_vs_live,
    compute_live_metrics,
    get_closed_trades,
    rolling_expectancy,
)
from scanner.execution.storage import ensure_orders_table, new_order_id, save_order


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_orders_table(c)
    # Also create scans table for sim_vs_live comparison
    c.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            scan_id TEXT PRIMARY KEY,
            machine_block TEXT
        )
    """)
    return c


def _insert_trade(conn, scan_id="s1", coin="BTC", direction="LONG",
                   entry_fill=62000, exit_fill=59800, exit_type="stop_loss",
                   quantity=0.1, stop_price=59800, leverage=5,
                   execution_mode="testnet", machine_block=None):
    """Insert a complete trade (entry + exit) into orders table."""
    entry_oid = new_order_id()
    exit_oid = new_order_id()
    sl_oid = new_order_id()

    # Entry order
    save_order({
        "order_id": entry_oid,
        "scan_id": scan_id,
        "coin": coin,
        "direction": direction,
        "order_type": "entry",
        "status": "FILLED",
        "entry_price": entry_fill,
        "fill_price": entry_fill,
        "quantity": quantity,
        "leverage": leverage,
        "exchange": "hyperliquid",
        "execution_mode": execution_mode,
        "placed_at": "2026-02-24T00:00:00+00:00",
        "filled_at": "2026-02-24T00:00:01+00:00",
        "created_at": "2026-02-24T00:00:00+00:00",
    }, conn)

    # SL order (for stop price reference)
    save_order({
        "order_id": sl_oid,
        "scan_id": scan_id,
        "coin": coin,
        "direction": direction,
        "order_type": "stop_loss",
        "status": "FILLED" if exit_type == "stop_loss" else "CANCELLED",
        "trigger_price": stop_price,
        "fill_price": exit_fill if exit_type == "stop_loss" else None,
        "quantity": quantity,
        "exchange": "hyperliquid",
        "execution_mode": execution_mode,
        "filled_at": "2026-02-24T12:00:00+00:00" if exit_type == "stop_loss" else None,
        "created_at": "2026-02-24T00:00:00+00:00",
    }, conn)

    # Exit order (if TP)
    if exit_type != "stop_loss":
        save_order({
            "order_id": exit_oid,
            "scan_id": scan_id,
            "coin": coin,
            "direction": direction,
            "order_type": exit_type,
            "status": "FILLED",
            "trigger_price": exit_fill,
            "fill_price": exit_fill,
            "quantity": quantity,
            "exchange": "hyperliquid",
            "execution_mode": execution_mode,
            "filled_at": "2026-02-24T12:00:00+00:00",
            "created_at": "2026-02-24T00:00:00+00:00",
        }, conn)

    # Machine block in scans table
    if machine_block:
        conn.execute(
            "INSERT OR REPLACE INTO scans (scan_id, machine_block) VALUES (?, ?)",
            (scan_id, json.dumps(machine_block)),
        )
        conn.commit()


class TestGetClosedTrades:
    def test_joins_entry_and_exit(self, conn):
        _insert_trade(conn, scan_id="s1", entry_fill=62000, exit_fill=59800,
                       exit_type="stop_loss", stop_price=59800)
        trades = get_closed_trades(conn)
        assert len(trades) == 1
        assert trades[0].coin == "BTC"
        assert trades[0].entry_fill == 62000
        assert trades[0].exit_fill == 59800
        assert trades[0].exit_type == "SL"

    def test_filters_by_execution_mode(self, conn):
        _insert_trade(conn, scan_id="s1", execution_mode="testnet")
        _insert_trade(conn, scan_id="s2", execution_mode="live", coin="ETH",
                       entry_fill=3000, exit_fill=2800, stop_price=2800)
        testnet = get_closed_trades(conn, execution_mode="testnet")
        live = get_closed_trades(conn, execution_mode="live")
        assert len(testnet) == 1
        assert len(live) == 1
        assert testnet[0].coin == "BTC"
        assert live[0].coin == "ETH"

    def test_open_trade_not_included(self, conn):
        """Entry filled but no exit filled -> not a closed trade."""
        save_order({
            "order_id": new_order_id(),
            "scan_id": "s_open",
            "coin": "SOL",
            "direction": "LONG",
            "order_type": "entry",
            "status": "FILLED",
            "entry_price": 100,
            "fill_price": 100,
            "quantity": 10,
            "exchange": "hyperliquid",
            "execution_mode": "testnet",
            "created_at": "2026-02-24T00:00:00+00:00",
        }, conn)
        trades = get_closed_trades(conn)
        assert len(trades) == 0

    def test_realized_r_calculation(self, conn):
        """LONG: buy at 62000, SL hit at 59800 -> -1R."""
        _insert_trade(conn, scan_id="s1", direction="LONG",
                       entry_fill=62000, exit_fill=59800,
                       exit_type="stop_loss", stop_price=59800, quantity=0.1)
        trades = get_closed_trades(conn)
        assert len(trades) == 1
        # risk = |62000 - 59800| * 0.1 = 220
        # pnl = (59800 - 62000) * 0.1 = -220
        # R = -220 / 220 = -1.0
        assert abs(trades[0].realized_r - (-1.0)) < 0.01


    def test_multi_exit_pnl(self, conn):
        """TP1 + TP2 blended PnL: sum across both exit legs."""
        entry_oid = new_order_id()
        # Entry: LONG 0.1 BTC at 62000
        save_order({
            "order_id": entry_oid, "scan_id": "s_multi", "coin": "BTC",
            "direction": "LONG", "order_type": "entry", "status": "FILLED",
            "entry_price": 62000, "fill_price": 62000, "quantity": 0.1,
            "leverage": 5, "exchange": "hyperliquid", "execution_mode": "testnet",
            "placed_at": "2026-02-24T00:00:00+00:00",
            "filled_at": "2026-02-24T00:00:01+00:00",
            "created_at": "2026-02-24T00:00:00+00:00",
        }, conn)
        # SL order (cancelled — both TPs hit)
        save_order({
            "order_id": new_order_id(), "scan_id": "s_multi", "coin": "BTC",
            "direction": "LONG", "order_type": "stop_loss", "status": "CANCELLED",
            "trigger_price": 59800, "quantity": 0.1,
            "exchange": "hyperliquid", "execution_mode": "testnet",
            "created_at": "2026-02-24T00:00:00+00:00",
        }, conn)
        # TP1: 0.05 at 65000
        save_order({
            "order_id": new_order_id(), "scan_id": "s_multi", "coin": "BTC",
            "direction": "LONG", "order_type": "tp1", "status": "FILLED",
            "trigger_price": 65000, "fill_price": 65000, "quantity": 0.05,
            "exchange": "hyperliquid", "execution_mode": "testnet",
            "filled_at": "2026-02-24T06:00:00+00:00",
            "created_at": "2026-02-24T00:00:00+00:00",
        }, conn)
        # TP2: 0.05 at 68000
        save_order({
            "order_id": new_order_id(), "scan_id": "s_multi", "coin": "BTC",
            "direction": "LONG", "order_type": "tp2", "status": "FILLED",
            "trigger_price": 68000, "fill_price": 68000, "quantity": 0.05,
            "exchange": "hyperliquid", "execution_mode": "testnet",
            "filled_at": "2026-02-24T12:00:00+00:00",
            "created_at": "2026-02-24T00:00:00+00:00",
        }, conn)

        trades = get_closed_trades(conn)
        assert len(trades) == 1
        # PnL = (65000-62000)*0.05 + (68000-62000)*0.05 = 150 + 300 = 450
        assert abs(trades[0].realized_pnl - 450) < 0.01
        assert trades[0].exit_type == "TP2"  # Last exit determines label

    def test_tp1_then_sl_pnl(self, conn):
        """TP1 wins, SL loses on remainder → net PnL."""
        entry_oid = new_order_id()
        save_order({
            "order_id": entry_oid, "scan_id": "s_mixed", "coin": "BTC",
            "direction": "LONG", "order_type": "entry", "status": "FILLED",
            "entry_price": 62000, "fill_price": 62000, "quantity": 0.1,
            "leverage": 5, "exchange": "hyperliquid", "execution_mode": "testnet",
            "placed_at": "2026-02-24T00:00:00+00:00",
            "filled_at": "2026-02-24T00:00:01+00:00",
            "created_at": "2026-02-24T00:00:00+00:00",
        }, conn)
        # SL: fills remaining 0.05 at 59800
        save_order({
            "order_id": new_order_id(), "scan_id": "s_mixed", "coin": "BTC",
            "direction": "LONG", "order_type": "stop_loss", "status": "FILLED",
            "trigger_price": 59800, "fill_price": 59800, "quantity": 0.05,
            "exchange": "hyperliquid", "execution_mode": "testnet",
            "filled_at": "2026-02-24T12:00:00+00:00",
            "created_at": "2026-02-24T00:00:00+00:00",
        }, conn)
        # TP1: 0.05 at 65000
        save_order({
            "order_id": new_order_id(), "scan_id": "s_mixed", "coin": "BTC",
            "direction": "LONG", "order_type": "tp1", "status": "FILLED",
            "trigger_price": 65000, "fill_price": 65000, "quantity": 0.05,
            "exchange": "hyperliquid", "execution_mode": "testnet",
            "filled_at": "2026-02-24T06:00:00+00:00",
            "created_at": "2026-02-24T00:00:00+00:00",
        }, conn)

        trades = get_closed_trades(conn)
        assert len(trades) == 1
        # PnL = (65000-62000)*0.05 + (59800-62000)*0.05 = 150 + (-110) = 40
        assert abs(trades[0].realized_pnl - 40) < 0.01
        assert trades[0].exit_type == "SL"  # Last exit by filled_at

    def test_partial_exit_not_in_closed_trades(self, conn):
        """Position with only TP1 filled (partial) should NOT appear in closed trades."""
        save_order({
            "order_id": new_order_id(), "scan_id": "s_partial", "coin": "BTC",
            "direction": "LONG", "order_type": "entry", "status": "FILLED",
            "entry_price": 62000, "fill_price": 62000, "quantity": 0.1,
            "leverage": 5, "exchange": "hyperliquid", "execution_mode": "testnet",
            "filled_at": "2026-02-24T00:00:01+00:00",
            "created_at": "2026-02-24T00:00:00+00:00",
        }, conn)
        # Only TP1 filled (half)
        save_order({
            "order_id": new_order_id(), "scan_id": "s_partial", "coin": "BTC",
            "direction": "LONG", "order_type": "tp1", "status": "FILLED",
            "trigger_price": 65000, "fill_price": 65000, "quantity": 0.05,
            "exchange": "hyperliquid", "execution_mode": "testnet",
            "filled_at": "2026-02-24T06:00:00+00:00",
            "created_at": "2026-02-24T00:00:00+00:00",
        }, conn)
        # SL still placed (not filled)
        save_order({
            "order_id": new_order_id(), "scan_id": "s_partial", "coin": "BTC",
            "direction": "LONG", "order_type": "stop_loss", "status": "PLACED",
            "trigger_price": 59800, "quantity": 0.05,
            "exchange": "hyperliquid", "execution_mode": "testnet",
            "created_at": "2026-02-24T00:00:00+00:00",
        }, conn)

        trades = get_closed_trades(conn)
        assert len(trades) == 0


class TestComputeLiveMetrics:
    def test_basic_metrics(self):
        trades = [
            TradeRecord("s1", "BTC", "LONG", 62000, 65000, "TP1", 300, 1.36, 220, 24, 0, 0, "testnet"),
            TradeRecord("s2", "ETH", "SHORT", 3000, 3200, "SL", -200, -1.0, 200, 12, 0, 0, "testnet"),
        ]
        m = compute_live_metrics(trades)
        assert m["count"] == 2
        assert m["win_rate"] == 0.5
        assert m["total_pnl"] == 100

    def test_no_trades(self):
        m = compute_live_metrics([])
        assert m["count"] == 0
        assert m["win_rate"] == 0
        assert m["expectancy_r"] == 0

    def test_profit_factor(self):
        trades = [
            TradeRecord("s1", "BTC", "LONG", 62000, 65000, "TP1", 300, 1.36, 220, 24, 0, 0, "testnet"),
            TradeRecord("s2", "ETH", "LONG", 3000, 2900, "SL", -100, -1.0, 100, 12, 0, 0, "testnet"),
        ]
        m = compute_live_metrics(trades)
        assert m["profit_factor"] == 3.0  # 300/100

    def test_fill_rate_default_zero(self):
        """fill_rate is 0 by default (caller populates with CONFIRM context)."""
        m = compute_live_metrics([])
        assert m["fill_rate"] == 0


class TestCompareSimVsLive:
    def test_matches_correctly(self, conn):
        mb = {
            "direction": "LONG",
            "entry_low": 61800,
            "entry_high": 62200,
            "stop_loss": 59800,
            "tp1": 65000,
            "leverage": 5,
        }
        _insert_trade(conn, scan_id="s1", entry_fill=62000, exit_fill=65000,
                       exit_type="tp1", stop_price=59800, machine_block=mb)

        trades = get_closed_trades(conn)
        comparisons = compare_sim_vs_live(trades, conn)
        assert len(comparisons) == 1
        assert comparisons[0]["coin"] == "BTC"
        # sim_r = (65000-62000)/2200 = 1.3636
        assert abs(comparisons[0]["sim_r"] - 1.3636) < 0.01

    def test_no_machine_block(self, conn):
        _insert_trade(conn, scan_id="s1")
        trades = get_closed_trades(conn)
        comparisons = compare_sim_vs_live(trades, conn)
        assert len(comparisons) == 0


class TestRollingExpectancy:
    def test_window_calculation(self):
        trades = [
            TradeRecord(f"s{i}", "BTC", "LONG", 62000, 65000, "TP1",
                        300 if i % 2 == 0 else -220,
                        1.36 if i % 2 == 0 else -1.0,
                        220, 24, 0, 0, "testnet")
            for i in range(25)
        ]
        result = rolling_expectancy(trades, window=20)
        assert len(result) > 0
        assert result[0][0] == 20  # starts at window size

    def test_too_few_trades(self):
        trades = [
            TradeRecord("s1", "BTC", "LONG", 62000, 65000, "TP1", 300, 1.36, 220, 24, 0, 0, "testnet")
        ]
        result = rolling_expectancy(trades, window=20)
        assert len(result) == 0
