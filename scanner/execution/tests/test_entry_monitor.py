"""Tests for conditional entry monitor — trigger evaluation, storage, orchestration."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from scanner.execution.config import ExecutionConfig
from scanner.execution.entry_monitor import (
    evaluate_entry_trigger,
    is_1h_boundary,
    save_conditional_entry,
)
from scanner.execution.storage import (
    ensure_orders_table,
    get_active_pending_entries,
    has_active_pending,
    increment_candles_checked,
    save_pending_entry,
    update_last_checked_hour,
    update_pending_status,
)


@pytest.fixture
def ce_config():
    """ExecutionConfig with conditional entry enabled."""
    return ExecutionConfig(
        enabled=True,
        conditional_entry_enabled=True,
        mode="testnet",
        private_key="0xtest_key",
        wallet_address="0xtest_wallet",
    )


@pytest.fixture
def conn():
    """In-memory SQLite with orders + pending_entries tables."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_orders_table(c)
    return c


def _make_pending(coin="BTC", direction="LONG", trigger="candle_close_above",
                  trigger_price=100.0, inv_price=95.0, expiry=12, **overrides):
    """Build a pending entry dict for testing."""
    entry = {
        "pending_id": f"p_{coin.lower()}",
        "scan_id": f"s_{coin.lower()}",
        "coin": coin,
        "direction": direction,
        "machine_block": json.dumps({
            "entry_low": 99.0, "entry_high": 101.0, "stop_loss": 95.0,
            "tp1": 110.0, "tp1_exit_pct": 50, "direction": direction,
            "confidence": 7, "leverage": 5,
            "entry_trigger": trigger, "entry_trigger_price": trigger_price,
            "invalidation_price": inv_price, "expiry_candles": expiry,
        }),
        "entry_trigger": trigger,
        "entry_trigger_price": trigger_price,
        "invalidation_price": inv_price,
        "expiry_candles": expiry,
        "exchange": "hyperliquid",
        "execution_mode": "testnet",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    entry.update(overrides)
    return entry


# ---------------------------------------------------------------------------
# Pure function tests — is_1h_boundary
# ---------------------------------------------------------------------------

class TestIs1hBoundary:

    def test_at_boundary_early(self):
        with patch("scanner.execution.entry_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 15, 12, 3, tzinfo=timezone.utc)
            assert is_1h_boundary() is True

    def test_at_boundary_edge(self):
        with patch("scanner.execution.entry_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 15, 12, 14, tzinfo=timezone.utc)
            assert is_1h_boundary() is True

    def test_past_window(self):
        with patch("scanner.execution.entry_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 15, 12, 16, tzinfo=timezone.utc)
            assert is_1h_boundary() is False

    def test_mid_hour(self):
        with patch("scanner.execution.entry_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 15, 12, 45, tzinfo=timezone.utc)
            assert is_1h_boundary() is False


# ---------------------------------------------------------------------------
# Pure function tests — evaluate_entry_trigger
# ---------------------------------------------------------------------------

class TestEvaluateEntryTrigger:

    def test_candle_close_above_triggered(self):
        event, action = evaluate_entry_trigger(
            "candle_close_above", 100.0, 95.0, "LONG", 101.0)
        assert event == "triggered"
        assert action == "execute"

    def test_candle_close_above_exact(self):
        event, action = evaluate_entry_trigger(
            "candle_close_above", 100.0, 95.0, "LONG", 100.0)
        assert event == "triggered"

    def test_candle_close_above_not_triggered(self):
        event, action = evaluate_entry_trigger(
            "candle_close_above", 100.0, 95.0, "LONG", 99.5)
        assert event is None
        assert action == "none"

    def test_candle_close_below_triggered(self):
        event, action = evaluate_entry_trigger(
            "candle_close_below", 100.0, 105.0, "SHORT", 99.0)
        assert event == "triggered"
        assert action == "execute"

    def test_candle_close_below_not_triggered(self):
        event, action = evaluate_entry_trigger(
            "candle_close_below", 100.0, 105.0, "SHORT", 100.5)
        assert event is None
        assert action == "none"

    def test_invalidation_long_fires(self):
        event, action = evaluate_entry_trigger(
            "candle_close_above", 100.0, 95.0, "LONG", 94.0)
        assert event == "invalidated"
        assert action == "kill"

    def test_invalidation_long_exact(self):
        event, action = evaluate_entry_trigger(
            "candle_close_above", 100.0, 95.0, "LONG", 95.0)
        assert event == "invalidated"

    def test_invalidation_short_fires(self):
        event, action = evaluate_entry_trigger(
            "candle_close_below", 100.0, 105.0, "SHORT", 106.0)
        assert event == "invalidated"
        assert action == "kill"

    def test_invalidation_takes_priority(self):
        """When both trigger and invalidation would fire, invalidation wins."""
        # LONG: close <= invalidation AND close >= trigger (impossible in practice
        # but test priority by using equal prices)
        event, action = evaluate_entry_trigger(
            "candle_close_above", 95.0, 95.0, "LONG", 95.0)
        assert event == "invalidated"
        assert action == "kill"

    def test_neither_fires(self):
        event, action = evaluate_entry_trigger(
            "candle_close_above", 100.0, 90.0, "LONG", 97.0)
        assert event is None
        assert action == "none"


# ---------------------------------------------------------------------------
# Storage tests
# ---------------------------------------------------------------------------

class TestPendingStorage:

    def test_save_and_retrieve(self, conn):
        entry = _make_pending()
        save_pending_entry(entry, conn)

        active = get_active_pending_entries(conn, "hyperliquid", "testnet")
        assert len(active) == 1
        assert active[0]["coin"] == "BTC"
        assert active[0]["status"] == "ACTIVE"

    def test_get_active_filters_status(self, conn):
        entry = _make_pending()
        save_pending_entry(entry, conn)
        update_pending_status("p_btc", "TRIGGERED", conn)

        active = get_active_pending_entries(conn, "hyperliquid", "testnet")
        assert len(active) == 0

    def test_update_status_sets_resolved_at(self, conn):
        entry = _make_pending()
        save_pending_entry(entry, conn)
        update_pending_status("p_btc", "INVALIDATED", conn)

        row = conn.execute("SELECT status, resolved_at FROM pending_entries WHERE pending_id = 'p_btc'").fetchone()
        assert row["status"] == "INVALIDATED"
        assert row["resolved_at"] is not None

    def test_has_active_pending_true(self, conn):
        save_pending_entry(_make_pending(), conn)
        assert has_active_pending("BTC", conn, "hyperliquid", "testnet") is True

    def test_has_active_pending_false(self, conn):
        assert has_active_pending("BTC", conn, "hyperliquid", "testnet") is False

    def test_has_active_pending_resolved_not_counted(self, conn):
        save_pending_entry(_make_pending(), conn)
        update_pending_status("p_btc", "EXPIRED", conn)
        assert has_active_pending("BTC", conn, "hyperliquid", "testnet") is False

    def test_increment_candles_checked(self, conn):
        save_pending_entry(_make_pending(), conn)
        new_count = increment_candles_checked("p_btc", conn)
        assert new_count == 1
        new_count = increment_candles_checked("p_btc", conn)
        assert new_count == 2

    def test_duplicate_scan_id_rejected(self, conn):
        save_pending_entry(_make_pending(), conn)
        with pytest.raises(sqlite3.IntegrityError):
            save_pending_entry(_make_pending(pending_id="p_btc2"), conn)


# ---------------------------------------------------------------------------
# Integration tests — check_pending_entries orchestrator
# ---------------------------------------------------------------------------

class TestCheckPendingEntries:

    def test_disabled_returns_zero(self, conn):
        from scanner.execution.entry_monitor import check_pending_entries

        config = ExecutionConfig(
            enabled=True, conditional_entry_enabled=False,
            mode="testnet", private_key="0xtest", wallet_address="0xtest",
        )
        assert check_pending_entries(config, conn) == 0

    @patch("scanner.execution.entry_monitor.is_1h_boundary", return_value=False)
    def test_non_boundary_returns_zero(self, mock_boundary, ce_config, conn):
        from scanner.execution.entry_monitor import check_pending_entries
        assert check_pending_entries(ce_config, conn) == 0

    @patch("scanner.execution.entry_monitor.is_1h_boundary", return_value=True)
    def test_no_active_entries_returns_zero(self, mock_boundary, ce_config, conn):
        from scanner.execution.entry_monitor import check_pending_entries
        assert check_pending_entries(ce_config, conn) == 0

    @patch("scanner.execution.entry_monitor._send_trigger_alert", return_value=True)
    @patch("scanner.execution.entry_monitor._send_resolution_alert", return_value=True)
    @patch("scanner.execution.entry_monitor.is_1h_boundary", return_value=True)
    def test_trigger_fires_executes(self, mock_boundary, mock_res_alert, mock_trig_alert, ce_config, conn):
        from scanner.execution.entry_monitor import check_pending_entries

        save_pending_entry(_make_pending(trigger_price=100.0, inv_price=90.0), conn)

        candle = {"timestamp": datetime.now(timezone.utc).isoformat(),
                  "open": 99.0, "high": 102.0, "low": 98.0, "close": 101.0}

        with patch("scanner.execution.entry_monitor._fetch_1h_candle", return_value=candle), \
             patch("scanner.execution.entry_monitor.has_open_position", return_value=False), \
             patch("scanner.execution.entry_monitor._execute_triggered_entry", return_value=1) as mock_exec:
            acted = check_pending_entries(ce_config, conn)

        assert acted == 1
        mock_exec.assert_called_once()

    @patch("scanner.execution.entry_monitor._send_resolution_alert", return_value=True)
    @patch("scanner.execution.entry_monitor.is_1h_boundary", return_value=True)
    def test_invalidation_kills_entry(self, mock_boundary, mock_alert, ce_config, conn):
        from scanner.execution.entry_monitor import check_pending_entries

        save_pending_entry(_make_pending(trigger_price=100.0, inv_price=95.0), conn)

        candle = {"timestamp": datetime.now(timezone.utc).isoformat(),
                  "open": 96.0, "high": 97.0, "low": 93.0, "close": 94.0}

        with patch("scanner.execution.entry_monitor._fetch_1h_candle", return_value=candle), \
             patch("scanner.execution.entry_monitor.has_open_position", return_value=False):
            acted = check_pending_entries(ce_config, conn)

        assert acted == 1
        row = conn.execute("SELECT status FROM pending_entries WHERE pending_id = 'p_btc'").fetchone()
        assert row["status"] == "INVALIDATED"

    @patch("scanner.execution.entry_monitor._send_resolution_alert", return_value=True)
    @patch("scanner.execution.entry_monitor.is_1h_boundary", return_value=True)
    def test_expiry_after_max_candles(self, mock_boundary, mock_alert, ce_config, conn):
        from scanner.execution.entry_monitor import check_pending_entries

        save_pending_entry(_make_pending(trigger_price=100.0, inv_price=90.0, expiry=3), conn)
        # Set candles_checked to 2 (one before expiry)
        conn.execute("UPDATE pending_entries SET candles_checked = 2 WHERE pending_id = 'p_btc'")
        conn.commit()

        # Candle that neither triggers nor invalidates
        candle = {"timestamp": datetime.now(timezone.utc).isoformat(),
                  "open": 97.0, "high": 98.0, "low": 96.0, "close": 97.0}

        with patch("scanner.execution.entry_monitor._fetch_1h_candle", return_value=candle), \
             patch("scanner.execution.entry_monitor.has_open_position", return_value=False):
            acted = check_pending_entries(ce_config, conn)

        assert acted == 1
        row = conn.execute("SELECT status FROM pending_entries WHERE pending_id = 'p_btc'").fetchone()
        assert row["status"] == "EXPIRED"

    @patch("scanner.execution.entry_monitor._send_resolution_alert", return_value=True)
    @patch("scanner.execution.entry_monitor.is_1h_boundary", return_value=True)
    def test_position_exists_invalidates(self, mock_boundary, mock_alert, ce_config, conn):
        from scanner.execution.entry_monitor import check_pending_entries

        save_pending_entry(_make_pending(), conn)

        with patch("scanner.execution.entry_monitor.has_open_position", return_value=True):
            acted = check_pending_entries(ce_config, conn)

        assert acted == 1
        row = conn.execute("SELECT status FROM pending_entries WHERE pending_id = 'p_btc'").fetchone()
        assert row["status"] == "INVALIDATED"

    @patch("scanner.execution.entry_monitor.is_1h_boundary", return_value=True)
    def test_stale_candle_skipped(self, mock_boundary, ce_config, conn):
        from scanner.execution.entry_monitor import check_pending_entries

        save_pending_entry(_make_pending(), conn)

        old_time = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        candle = {"timestamp": old_time,
                  "open": 99.0, "high": 102.0, "low": 98.0, "close": 101.0}

        with patch("scanner.execution.entry_monitor._fetch_1h_candle", return_value=candle), \
             patch("scanner.execution.entry_monitor.has_open_position", return_value=False):
            acted = check_pending_entries(ce_config, conn)

        assert acted == 0
        row = conn.execute("SELECT status FROM pending_entries WHERE pending_id = 'p_btc'").fetchone()
        assert row["status"] == "ACTIVE"


# ---------------------------------------------------------------------------
# save_conditional_entry tests
# ---------------------------------------------------------------------------

class TestSaveConditionalEntry:

    @patch("scanner.execution.entry_monitor.fetch_current_price", return_value=2300.0)
    @patch("scanner.execution.entry_monitor._send_pending_alert", return_value=True)
    def test_saves_entry(self, mock_alert, mock_price, ce_config, conn):
        from scanner.models import ScanResult

        result = ScanResult(
            scan_id="test123", coin="ETH",
            timestamp=datetime.now(timezone.utc),
            setup_signal=None, decision="CONFIRM",
            confluence_score=None, analysis_text=None,
            parsed_decision=None, chart_paths=[], cost_usd=0.0,
            machine_block={
                "entry_low": 2300, "entry_high": 2350, "stop_loss": 2200,
                "tp1": 2500, "tp1_exit_pct": 50, "direction": "LONG",
                "confidence": 7, "leverage": 5,
                "entry_trigger": "candle_close_above",
                "entry_trigger_price": 2350.0,
                "invalidation_price": 2180.0,
                "expiry_candles": 12,
            },
        )

        saved = save_conditional_entry(result, ce_config, conn)
        assert saved is True

        active = get_active_pending_entries(conn, "hyperliquid", "testnet")
        assert len(active) == 1
        assert active[0]["coin"] == "ETH"
        assert active[0]["entry_trigger_price"] == 2350.0

    @patch("scanner.execution.entry_monitor.fetch_current_price", return_value=97.0)
    @patch("scanner.execution.entry_monitor._send_pending_alert", return_value=True)
    def test_dedup_blocks_second(self, mock_alert, mock_price, ce_config, conn):
        from scanner.models import ScanResult

        mb = {
            "entry_low": 99, "entry_high": 101, "stop_loss": 95,
            "tp1": 110, "tp1_exit_pct": 50, "direction": "LONG",
            "confidence": 7, "leverage": 5,
            "entry_trigger": "candle_close_above",
            "entry_trigger_price": 100.0,
            "invalidation_price": 93.0,
            "expiry_candles": 12,
        }

        r1 = ScanResult(scan_id="s1", coin="BTC", timestamp=datetime.now(timezone.utc),
                        setup_signal=None, decision="CONFIRM", confluence_score=None,
                        analysis_text=None, parsed_decision=None, chart_paths=[],
                        cost_usd=0.0, machine_block=mb)
        r2 = ScanResult(scan_id="s2", coin="BTC", timestamp=datetime.now(timezone.utc),
                        setup_signal=None, decision="CONFIRM", confluence_score=None,
                        analysis_text=None, parsed_decision=None, chart_paths=[],
                        cost_usd=0.0, machine_block=mb)

        assert save_conditional_entry(r1, ce_config, conn) is True
        assert save_conditional_entry(r2, ce_config, conn) is False

    def test_expiry_capped(self, ce_config, conn):
        from scanner.models import ScanResult

        ce_config.conditional_entry_max_expiry = 10

        mb = {
            "entry_low": 99, "entry_high": 101, "stop_loss": 95,
            "tp1": 110, "tp1_exit_pct": 50, "direction": "LONG",
            "confidence": 7, "leverage": 5,
            "entry_trigger": "candle_close_above",
            "entry_trigger_price": 100.0,
            "invalidation_price": 93.0,
            "expiry_candles": 24,
        }

        result = ScanResult(scan_id="s1", coin="BTC", timestamp=datetime.now(timezone.utc),
                            setup_signal=None, decision="CONFIRM", confluence_score=None,
                            analysis_text=None, parsed_decision=None, chart_paths=[],
                            cost_usd=0.0, machine_block=mb)

        with patch("scanner.execution.entry_monitor._send_pending_alert", return_value=True), \
             patch("scanner.execution.entry_monitor.fetch_current_price", return_value=97.0):
            save_conditional_entry(result, ce_config, conn)

        row = conn.execute("SELECT expiry_candles FROM pending_entries").fetchone()
        assert row["expiry_candles"] == 10


# ---------------------------------------------------------------------------
# Dedup: once-per-candle guard tests
# ---------------------------------------------------------------------------

class TestOncePerCandleDedup:

    @patch("scanner.execution.entry_monitor._send_resolution_alert", return_value=True)
    @patch("scanner.execution.entry_monitor.is_1h_boundary", return_value=True)
    def test_second_call_same_hour_skips(self, mock_boundary, mock_alert, ce_config, conn):
        """Two calls with the same candle hour should only increment once."""
        from scanner.execution.entry_monitor import check_pending_entries

        save_pending_entry(_make_pending(trigger_price=100.0, inv_price=90.0, expiry=12), conn)

        candle = {"timestamp": "2026-04-16T14:00:00+00:00",
                  "open": 97.0, "high": 98.0, "low": 96.0, "close": 97.0}

        with patch("scanner.execution.entry_monitor._fetch_1h_candle", return_value=candle), \
             patch("scanner.execution.entry_monitor.has_open_position", return_value=False):
            check_pending_entries(ce_config, conn)  # First call: processes
            check_pending_entries(ce_config, conn)  # Second call: skips (same hour)

        row = conn.execute(
            "SELECT candles_checked, last_checked_hour FROM pending_entries WHERE pending_id = 'p_btc'"
        ).fetchone()
        assert row["candles_checked"] == 1
        assert row["last_checked_hour"] == "2026-04-16T14"

    @patch("scanner.execution.entry_monitor.is_1h_boundary", return_value=True)
    def test_different_hour_processes_again(self, mock_boundary, ce_config, conn):
        """Two calls with different candle hours should increment twice."""
        from scanner.execution.entry_monitor import check_pending_entries

        save_pending_entry(_make_pending(trigger_price=100.0, inv_price=90.0, expiry=12), conn)

        candle_14 = {"timestamp": "2026-04-16T14:00:00+00:00",
                     "open": 97.0, "high": 98.0, "low": 96.0, "close": 97.0}
        candle_15 = {"timestamp": "2026-04-16T15:00:00+00:00",
                     "open": 97.0, "high": 98.0, "low": 96.0, "close": 97.5}

        with patch("scanner.execution.entry_monitor.has_open_position", return_value=False):
            with patch("scanner.execution.entry_monitor._fetch_1h_candle", return_value=candle_14):
                check_pending_entries(ce_config, conn)
            with patch("scanner.execution.entry_monitor._fetch_1h_candle", return_value=candle_15):
                check_pending_entries(ce_config, conn)

        row = conn.execute(
            "SELECT candles_checked FROM pending_entries WHERE pending_id = 'p_btc'"
        ).fetchone()
        assert row["candles_checked"] == 2

    @patch("scanner.execution.entry_monitor.is_1h_boundary", return_value=True)
    def test_null_last_checked_hour_processes(self, mock_boundary, ce_config, conn):
        """Pre-migration entry with NULL last_checked_hour processes on first call."""
        from scanner.execution.entry_monitor import check_pending_entries

        save_pending_entry(_make_pending(trigger_price=100.0, inv_price=90.0, expiry=12), conn)
        # Verify starts as NULL
        row = conn.execute(
            "SELECT last_checked_hour FROM pending_entries WHERE pending_id = 'p_btc'"
        ).fetchone()
        assert row["last_checked_hour"] is None

        candle = {"timestamp": "2026-04-16T14:00:00+00:00",
                  "open": 97.0, "high": 98.0, "low": 96.0, "close": 97.0}

        with patch("scanner.execution.entry_monitor._fetch_1h_candle", return_value=candle), \
             patch("scanner.execution.entry_monitor.has_open_position", return_value=False):
            check_pending_entries(ce_config, conn)

        row = conn.execute(
            "SELECT candles_checked, last_checked_hour FROM pending_entries WHERE pending_id = 'p_btc'"
        ).fetchone()
        assert row["candles_checked"] == 1
        assert row["last_checked_hour"] == "2026-04-16T14"


# ---------------------------------------------------------------------------
# _fetch_1h_candle — completed candle test
# ---------------------------------------------------------------------------

class TestFetch1hCandle:

    def test_returns_completed_not_in_progress(self):
        """Should return iloc[-2] (completed candle), not iloc[-1] (in-progress)."""
        import pandas as pd

        from scanner.execution.entry_monitor import _fetch_1h_candle

        data = {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [1000.0, 1100.0, 1200.0],
        }
        index = pd.to_datetime([
            "2026-04-16T13:00:00+00:00",
            "2026-04-16T14:00:00+00:00",
            "2026-04-16T15:00:00+00:00",
        ])
        df = pd.DataFrame(data, index=index)

        with patch("scanner.data.fetch_candles", return_value=df):
            candle = _fetch_1h_candle("BTC")

        assert candle is not None
        assert candle["close"] == 101.5  # completed, not 102.5 (in-progress)
        assert "2026-04-16T14" in candle["timestamp"]

    def test_returns_none_for_single_candle(self):
        """Need at least 2 candles to have a completed one at iloc[-2]."""
        import pandas as pd

        from scanner.execution.entry_monitor import _fetch_1h_candle

        data = {"Open": [100.0], "High": [101.0], "Low": [99.0],
                "Close": [100.5], "Volume": [1000.0]}
        index = pd.to_datetime(["2026-04-16T15:00:00+00:00"])
        df = pd.DataFrame(data, index=index)

        with patch("scanner.data.fetch_candles", return_value=df):
            assert _fetch_1h_candle("BTC") is None


# ---------------------------------------------------------------------------
# Storage: update_last_checked_hour
# ---------------------------------------------------------------------------

class TestUpdateLastCheckedHour:

    def test_sets_hour_key(self, conn):
        save_pending_entry(_make_pending(), conn)
        update_last_checked_hour("p_btc", "2026-04-16T14", conn)
        row = conn.execute(
            "SELECT last_checked_hour FROM pending_entries WHERE pending_id = 'p_btc'"
        ).fetchone()
        assert row["last_checked_hour"] == "2026-04-16T14"

    def test_overwrites_previous(self, conn):
        save_pending_entry(_make_pending(), conn)
        update_last_checked_hour("p_btc", "2026-04-16T14", conn)
        update_last_checked_hour("p_btc", "2026-04-16T15", conn)
        row = conn.execute(
            "SELECT last_checked_hour FROM pending_entries WHERE pending_id = 'p_btc'"
        ).fetchone()
        assert row["last_checked_hour"] == "2026-04-16T15"
