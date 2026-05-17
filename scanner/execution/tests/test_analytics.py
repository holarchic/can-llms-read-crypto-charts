"""Tests for execution analytics."""

import json
import sqlite3

import pytest

from scanner.execution.analytics import compute_live_metrics, rolling_expectancy
from scanner.execution.storage import (
    ensure_orders_table,
    get_closed_trades,
    llm_cost_summary,
    new_order_id,
    save_order,
)
from scanner.models import TradeRecord


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_orders_table(c)
    # scans table — needed for machine_block slippage lookup + llm_cost_summary.
    c.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            scan_id TEXT PRIMARY KEY,
            machine_block TEXT,
            cost_usd REAL DEFAULT 0,
            created_at TEXT
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
    def test_drift_close_counted_as_exit(self, conn):
        """drift_close is a valid exit type — must show up in closed trades."""
        entry_oid = new_order_id()
        save_order({
            "order_id": entry_oid, "scan_id": "s_drift", "coin": "BTC",
            "direction": "LONG", "order_type": "entry", "status": "FILLED",
            "entry_price": 100.0, "fill_price": 100.0, "quantity": 0.1,
            "leverage": 5, "exchange": "hyperliquid", "execution_mode": "testnet",
            "placed_at": "2026-02-24T00:00:00+00:00",
            "filled_at": "2026-02-24T00:00:01+00:00",
            "created_at": "2026-02-24T00:00:00+00:00",
        }, conn)
        # Need a (cancelled) SL for stop_price reference
        save_order({
            "order_id": new_order_id(), "scan_id": "s_drift", "coin": "BTC",
            "direction": "LONG", "order_type": "stop_loss", "status": "CANCELLED",
            "trigger_price": 95.0, "quantity": 0.1,
            "exchange": "hyperliquid", "execution_mode": "testnet",
            "created_at": "2026-02-24T00:00:00+00:00",
        }, conn)
        save_order({
            "order_id": new_order_id(), "scan_id": "s_drift", "coin": "BTC",
            "direction": "LONG", "order_type": "drift_close", "status": "FILLED",
            "fill_price": 98.0, "quantity": 0.1,
            "exchange": "hyperliquid", "execution_mode": "testnet",
            "filled_at": "2026-02-25T00:00:00+00:00",
            "created_at": "2026-02-25T00:00:00+00:00",
        }, conn)

        trades = get_closed_trades(conn)
        assert len(trades) == 1
        assert trades[0].exit_fill == 98.0

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

    def test_missing_exit_price_trade_skipped(self, conn, caplog):
        """Manual-close row with NULL fill_price and NULL trigger_price must
        not produce a phantom (entry - 0) * qty PnL.

        Regression for the 2026-04-17 DOT incident: close-manual.py inserted
        a timeout_close row with no price fields; analytics fell through to
        `or 0` and reported a fake +$227.50 win on a SHORT at ~1.32.
        """
        save_order({
            "order_id": new_order_id(), "scan_id": "s_nullfill", "coin": "DOT",
            "direction": "SHORT", "order_type": "entry", "status": "FILLED",
            "entry_price": 1.3273, "fill_price": 1.3273, "quantity": 171.4,
            "leverage": 5, "exchange": "hyperliquid", "execution_mode": "live",
            "filled_at": "2026-04-16T16:00:00+00:00",
            "created_at": "2026-04-16T16:00:00+00:00",
        }, conn)
        # Cancelled SL (for risk anchor)
        save_order({
            "order_id": new_order_id(), "scan_id": "s_nullfill", "coin": "DOT",
            "direction": "SHORT", "order_type": "stop_loss", "status": "CANCELLED",
            "trigger_price": 1.3523, "quantity": 171.4,
            "exchange": "hyperliquid", "execution_mode": "live",
            "created_at": "2026-04-16T16:00:00+00:00",
        }, conn)
        # Manual-close exit with no fill_price OR trigger_price
        save_order({
            "order_id": new_order_id(), "scan_id": "s_nullfill", "coin": "DOT",
            "direction": "SHORT", "order_type": "timeout_close", "status": "FILLED",
            "quantity": 171.4,
            "exchange": "hyperliquid", "execution_mode": "live",
            "filled_at": "2026-04-17T08:17:00+00:00",
            "created_at": "2026-04-17T08:17:00+00:00",
        }, conn)

        import logging
        with caplog.at_level(logging.WARNING, logger="scanner.execution.analytics"):
            trades = get_closed_trades(conn, execution_mode="live")

        assert trades == []
        assert any("no fill_price or trigger_price" in r.message for r in caplog.records)

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
        assert m.count == 2
        assert m.win_rate == 0.5
        assert m.total_pnl == 100

    def test_no_trades(self):
        m = compute_live_metrics([])
        assert m.count == 0
        assert m.win_rate == 0
        assert m.expectancy_r == 0

    def test_profit_factor(self):
        trades = [
            TradeRecord("s1", "BTC", "LONG", 62000, 65000, "TP1", 300, 1.36, 220, 24, 0, 0, "testnet"),
            TradeRecord("s2", "ETH", "LONG", 3000, 2900, "SL", -100, -1.0, 100, 12, 0, 0, "testnet"),
        ]
        m = compute_live_metrics(trades)
        assert m.profit_factor == 3.0  # 300/100

    def test_fill_rate_default_zero(self):
        """fill_rate is 0 by default (caller populates with CONFIRM context)."""
        m = compute_live_metrics([])
        assert m.fill_rate == 0

    def test_drawdown_all_losses(self):
        """All-loss sequence should report non-zero drawdown using total_risk."""
        trades = [
            TradeRecord("s1", "BTC", "LONG", 62000, 59800, "SL", -200, -1.0, 200, 12, 0, 0, "testnet"),
            TradeRecord("s2", "ETH", "SHORT", 3000, 3200, "SL", -200, -1.0, 200, 12, 0, 0, "testnet"),
            TradeRecord("s3", "SOL", "LONG", 100, 90, "SL", -200, -1.0, 200, 12, 0, 0, "testnet"),
        ]
        m = compute_live_metrics(trades)
        # total_risk = 600, total loss = 600, max_dd_pct = 600/600*100 = 100
        assert m.max_drawdown_pct > 0
        assert m.max_drawdown_pct == 100.0

    def test_drawdown_loss_then_win(self):
        """Loss then bigger win → peak-based drawdown (regression check)."""
        trades = [
            TradeRecord("s1", "BTC", "LONG", 62000, 59800, "SL", -200, -1.0, 200, 12, 0, 0, "testnet"),
            TradeRecord("s2", "ETH", "LONG", 3000, 3500, "TP1", 500, 2.5, 200, 12, 0, 0, "testnet"),
        ]
        m = compute_live_metrics(trades)
        # After trade 1: running=-200, peak=0, dd=200 (no positive peak yet)
        # After trade 2: running=300, peak=300, dd=0
        # Peak goes positive after trade 2, so peak=300
        # max_dd=200 happened before peak went positive → uses total_risk=400
        # Actually: peak never > 0 during dd=200 window, but peak=300 at end
        # Let's trace: t1: running=-200, peak=0 (not >0), dd=0-(-200)=200, max_dd=200
        #              t2: running=300, peak=300, dd=0
        # peak=300 > 0, so max_dd_pct = 200/300*100 = 66.67
        assert m.max_drawdown_pct == pytest.approx(66.67, abs=0.01)

    def test_funding_r_normalized_by_total_risk(self):
        """funding_r = total_funding / total_risk (not avg risk)."""
        trades = [
            TradeRecord(f"s{i}", "BTC", "LONG", 62000, 65000, "TP1", 300, 1.36, 100, 24, 5, 0, "testnet")
            for i in range(10)
        ]
        m = compute_live_metrics(trades)
        # total_funding = 50, total_risk = 1000
        # correct: 50/1000 = 0.05
        assert m.total_funding_r == pytest.approx(0.05, abs=0.001)


class TestLlmCostSummary:
    def test_sums_cost_within_window(self, conn):
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        rows = [
            ("s_in_3d", 0.50, (now - timedelta(days=3)).isoformat()),
            ("s_in_5d", 0.25, (now - timedelta(days=5)).isoformat()),
            ("s_out_10d", 1.00, (now - timedelta(days=10)).isoformat()),
        ]
        for sid, cost, ts in rows:
            conn.execute(
                "INSERT INTO scans (scan_id, cost_usd, created_at) VALUES (?, ?, ?)",
                (sid, cost, ts),
            )
        conn.commit()

        assert llm_cost_summary(conn, 7) == pytest.approx(0.75)
        assert llm_cost_summary(conn, 28) == pytest.approx(1.75)

    def test_empty_returns_zero(self, conn):
        assert llm_cost_summary(conn, 7) == 0.0


class TestCloseWindowFilter:
    """`days=` filters by final exit time, not entry created_at."""

    def test_old_entry_with_recent_close_appears(self, conn):
        """Entry from 60 days ago, exit filled 5 days ago — in the 7d window."""
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        entry_dt = (now - timedelta(days=60)).isoformat()
        exit_dt = (now - timedelta(days=5)).isoformat()

        save_order({
            "order_id": new_order_id(), "scan_id": "s_old_entry", "coin": "BTC",
            "direction": "LONG", "order_type": "entry", "status": "FILLED",
            "entry_price": 62000, "fill_price": 62000, "quantity": 0.1,
            "leverage": 5, "exchange": "hyperliquid", "execution_mode": "live",
            "placed_at": entry_dt, "filled_at": entry_dt, "created_at": entry_dt,
        }, conn)
        save_order({
            "order_id": new_order_id(), "scan_id": "s_old_entry", "coin": "BTC",
            "direction": "LONG", "order_type": "stop_loss", "status": "CANCELLED",
            "trigger_price": 59800, "quantity": 0.1,
            "exchange": "hyperliquid", "execution_mode": "live",
            "created_at": entry_dt,
        }, conn)
        save_order({
            "order_id": new_order_id(), "scan_id": "s_old_entry", "coin": "BTC",
            "direction": "LONG", "order_type": "tp1", "status": "FILLED",
            "trigger_price": 65000, "fill_price": 65000, "quantity": 0.1,
            "exchange": "hyperliquid", "execution_mode": "live",
            "filled_at": exit_dt, "created_at": exit_dt,
        }, conn)

        # 7d window: included (final exit at -5d)
        assert len(get_closed_trades(conn, execution_mode="live", days=7)) == 1
        # 3d window: excluded (final exit at -5d > 3d ago)
        assert len(get_closed_trades(conn, execution_mode="live", days=3)) == 0
        # 28d window: included
        assert len(get_closed_trades(conn, execution_mode="live", days=28)) == 1

    def test_recent_entry_no_filled_exit_excluded(self, conn):
        """Entry from 2 days ago, no exit yet — should not appear in any window."""
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        entry_dt = (now - timedelta(days=2)).isoformat()

        save_order({
            "order_id": new_order_id(), "scan_id": "s_open_recent", "coin": "SOL",
            "direction": "LONG", "order_type": "entry", "status": "FILLED",
            "entry_price": 100, "fill_price": 100, "quantity": 10,
            "exchange": "hyperliquid", "execution_mode": "live",
            "placed_at": entry_dt, "filled_at": entry_dt, "created_at": entry_dt,
        }, conn)
        # SL placed but not filled
        save_order({
            "order_id": new_order_id(), "scan_id": "s_open_recent", "coin": "SOL",
            "direction": "LONG", "order_type": "stop_loss", "status": "PLACED",
            "trigger_price": 95, "quantity": 10,
            "exchange": "hyperliquid", "execution_mode": "live",
            "created_at": entry_dt,
        }, conn)

        assert get_closed_trades(conn, execution_mode="live", days=7) == []
        assert get_closed_trades(conn, execution_mode="live", days=28) == []


class TestQualityWarnings:
    def test_flags_missing_funding(self):
        """All trades have funding_paid=0 → warning present."""
        trades = [
            TradeRecord("s1", "BTC", "LONG", 62000, 65000, "TP1", 300, 1.36, 220, 24, 0, 0, "testnet"),
            TradeRecord("s2", "ETH", "LONG", 3000, 3200, "TP1", 200, 1.0, 200, 12, 0, 0, "testnet"),
        ]
        m = compute_live_metrics(trades)
        assert any("Funding costs not captured" in w for w in m.quality_warnings)

    def test_no_funding_warning_when_clean(self):
        """Trades with funding_paid > 0 → no funding warning."""
        trades = [
            TradeRecord("s1", "BTC", "LONG", 62000, 65000, "TP1", 300, 1.36, 220, 24, 5.0, 0, "testnet"),
            TradeRecord("s2", "ETH", "LONG", 3000, 3200, "TP1", 200, 1.0, 200, 12, 3.0, 0, "testnet"),
        ]
        m = compute_live_metrics(trades)
        assert not any("Funding costs not captured" in w for w in m.quality_warnings)
        assert not any("missing funding" in w for w in m.quality_warnings)

    def test_flags_missing_fill_prices(self):
        """Trades with entry_fill=0 → warning about missing fill prices."""
        trades = [
            TradeRecord("s1", "BTC", "LONG", 0, 65000, "TP1", 300, 1.36, 220, 24, 5.0, 0, "testnet"),
        ]
        m = compute_live_metrics(trades)
        assert any("missing fill prices" in w for w in m.quality_warnings)
        assert any("using trigger prices" in w for w in m.quality_warnings)

    def test_empty_trades_has_db_note(self):
        """No trades → still has DB-only note."""
        m = compute_live_metrics([])
        assert any("not reconciled" in w for w in m.quality_warnings)

    def test_metrics_include_data_source(self):
        """Metrics dict always includes data_source=db_only."""
        m = compute_live_metrics([])
        assert m.data_source == "db_only"
        trades = [
            TradeRecord("s1", "BTC", "LONG", 62000, 65000, "TP1", 300, 1.36, 220, 24, 5.0, 0, "testnet"),
        ]
        m2 = compute_live_metrics(trades)
        assert m2.data_source == "db_only"

    def test_quality_warnings_always_include_db_note(self):
        """DB-only note is always the first warning."""
        trades = [
            TradeRecord("s1", "BTC", "LONG", 62000, 65000, "TP1", 300, 1.36, 220, 24, 5.0, 0, "testnet"),
        ]
        m = compute_live_metrics(trades)
        assert m.quality_warnings[0] == "All metrics estimated from DB — not reconciled with exchange"

    def test_quality_warnings_partial_funding(self):
        """Some trades missing funding → partial funding warning."""
        trades = [
            TradeRecord("s1", "BTC", "LONG", 62000, 65000, "TP1", 300, 1.36, 220, 24, 5.0, 0, "testnet"),
            TradeRecord("s2", "ETH", "LONG", 3000, 3200, "TP1", 200, 1.0, 200, 12, 0, 0, "testnet"),
        ]
        m = compute_live_metrics(trades)
        assert any("1/2 trades missing funding data" in w for w in m.quality_warnings)


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
