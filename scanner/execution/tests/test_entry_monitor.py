"""Tests for conditional entry monitor — trigger evaluation, storage, orchestration."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from scanner.execution.config import ExecutionConfig
from scanner.execution.entry_monitor import (
    _reconstruct_scan_result,
    evaluate_deviation_reclaim,
    evaluate_entry_trigger,
    evaluate_limit_at_trigger,
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


class TestEvaluateDeviationReclaim:
    """Ruleset §4.4: excess + reclaim two-part pattern."""

    def test_long_excess_then_reclaim_fires(self):
        # Price: 100 (at level), 99 (excess below support), 100.5 (reclaim)
        event, action = evaluate_deviation_reclaim(
            entry_trigger_price=100.0, invalidation_price=97.0,
            direction="LONG", candle_closes=[100.0, 99.0, 100.5],
        )
        assert event == "triggered"
        assert action == "execute"

    def test_long_no_reclaim_yet(self):
        # Price goes below but never back above
        event, action = evaluate_deviation_reclaim(
            entry_trigger_price=100.0, invalidation_price=97.0,
            direction="LONG", candle_closes=[100.0, 99.0, 98.5, 99.5],
        )
        assert event is None
        assert action == "none"

    def test_long_no_excess_yet(self):
        # Price stayed above support the whole time — no excess, no trigger
        event, action = evaluate_deviation_reclaim(
            entry_trigger_price=100.0, invalidation_price=97.0,
            direction="LONG", candle_closes=[100.5, 101.0, 100.2],
        )
        assert event is None
        assert action == "none"

    def test_short_excess_then_reclaim_fires(self):
        # Resistance at 100: price breaks to 101 (excess above), closes back at 99.5
        event, action = evaluate_deviation_reclaim(
            entry_trigger_price=100.0, invalidation_price=103.0,
            direction="SHORT", candle_closes=[100.0, 101.0, 99.5],
        )
        assert event == "triggered"
        assert action == "execute"

    def test_short_no_reclaim_yet(self):
        event, action = evaluate_deviation_reclaim(
            entry_trigger_price=100.0, invalidation_price=103.0,
            direction="SHORT", candle_closes=[100.5, 101.0, 101.5],
        )
        assert event is None
        assert action == "none"

    def test_long_invalidation_overrides_reclaim(self):
        # Excess AND invalidation hit simultaneously — invalidation wins
        event, action = evaluate_deviation_reclaim(
            entry_trigger_price=100.0, invalidation_price=97.0,
            direction="LONG", candle_closes=[100.0, 96.5, 100.2],
        )
        assert event == "invalidated"
        assert action == "kill"

    def test_short_invalidation_overrides_reclaim(self):
        event, action = evaluate_deviation_reclaim(
            entry_trigger_price=100.0, invalidation_price=103.0,
            direction="SHORT", candle_closes=[100.0, 103.5, 99.0],
        )
        assert event == "invalidated"
        assert action == "kill"

    def test_empty_closes_returns_none(self):
        event, action = evaluate_deviation_reclaim(
            entry_trigger_price=100.0, invalidation_price=97.0,
            direction="LONG", candle_closes=[],
        )
        assert event is None
        assert action == "none"

    def test_long_excess_extends_then_reclaims(self):
        # Extended excess before reclaim — should still fire on reclaim
        event, action = evaluate_deviation_reclaim(
            entry_trigger_price=100.0, invalidation_price=95.0,
            direction="LONG",
            candle_closes=[100.0, 99.0, 98.0, 97.5, 100.5],
        )
        assert event == "triggered"
        assert action == "execute"


class TestEvaluateLimitAt:
    """Ruleset §4.2 Trigger 1 — spot-price touch."""

    def test_long_below_trigger_fires(self):
        event, action = evaluate_limit_at_trigger(
            entry_trigger_price=100.0, invalidation_price=95.0,
            direction="LONG", current_price=99.5,
        )
        assert event == "triggered"
        assert action == "execute"

    def test_long_at_trigger_exactly(self):
        event, action = evaluate_limit_at_trigger(
            entry_trigger_price=100.0, invalidation_price=95.0,
            direction="LONG", current_price=100.0,
        )
        assert event == "triggered"

    def test_long_above_trigger_waits(self):
        event, action = evaluate_limit_at_trigger(
            entry_trigger_price=100.0, invalidation_price=95.0,
            direction="LONG", current_price=101.5,
        )
        assert event is None
        assert action == "none"

    def test_short_above_trigger_fires(self):
        event, action = evaluate_limit_at_trigger(
            entry_trigger_price=100.0, invalidation_price=105.0,
            direction="SHORT", current_price=100.5,
        )
        assert event == "triggered"
        assert action == "execute"

    def test_short_at_trigger_exactly(self):
        event, action = evaluate_limit_at_trigger(
            entry_trigger_price=100.0, invalidation_price=105.0,
            direction="SHORT", current_price=100.0,
        )
        assert event == "triggered"

    def test_short_below_trigger_waits(self):
        event, action = evaluate_limit_at_trigger(
            entry_trigger_price=100.0, invalidation_price=105.0,
            direction="SHORT", current_price=99.5,
        )
        assert event is None
        assert action == "none"

    def test_long_invalidation_overrides_trigger(self):
        # Price at trigger AND at invalidation — invalidation wins
        event, action = evaluate_limit_at_trigger(
            entry_trigger_price=100.0, invalidation_price=95.0,
            direction="LONG", current_price=94.5,
        )
        assert event == "invalidated"
        assert action == "kill"

    def test_short_invalidation_overrides_trigger(self):
        event, action = evaluate_limit_at_trigger(
            entry_trigger_price=100.0, invalidation_price=105.0,
            direction="SHORT", current_price=106.0,
        )
        assert event == "invalidated"
        assert action == "kill"


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

    def test_claim_pending_first_wins(self, conn):
        """claim_pending atomically flips ACTIVE → TRIGGERED; only the first caller wins."""
        from scanner.execution.storage import claim_pending

        save_pending_entry(_make_pending(), conn)
        assert claim_pending("p_btc", "TRIGGERED", conn) is True
        # Second attempt: status is already TRIGGERED → no ACTIVE row matches
        assert claim_pending("p_btc", "TRIGGERED", conn) is False

        row = conn.execute(
            "SELECT status FROM pending_entries WHERE pending_id = 'p_btc'",
        ).fetchone()
        assert row["status"] == "TRIGGERED"

    def test_claim_pending_nonexistent_returns_false(self, conn):
        from scanner.execution.storage import claim_pending

        assert claim_pending("nope", "TRIGGERED", conn) is False


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

    @patch("scanner.execution.entry_monitor.get_open_positions", return_value=[])
    @patch("scanner.execution.entry_monitor.fetch_current_price", return_value=2300.0)
    @patch("scanner.execution.entry_monitor._send_pending_alert", return_value=True)
    def test_saves_entry(self, mock_alert, mock_price, mock_pos, ce_config, conn):
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
                "invalidation_price": 2250.0,
                "expiry_candles": 12,
            },
        )

        saved = save_conditional_entry(result, ce_config, conn)
        assert saved is True

        active = get_active_pending_entries(conn, "hyperliquid", "testnet")
        assert len(active) == 1
        assert active[0]["coin"] == "ETH"
        assert active[0]["entry_trigger_price"] == 2350.0

    @patch("scanner.execution.entry_monitor.get_open_positions", return_value=[])
    @patch("scanner.execution.entry_monitor.fetch_current_price", return_value=97.0)
    @patch("scanner.execution.entry_monitor._send_pending_alert", return_value=True)
    def test_dedup_blocks_second(self, mock_alert, mock_price, mock_pos, ce_config, conn):
        from scanner.models import ScanResult

        mb = {
            "entry_low": 99, "entry_high": 101, "stop_loss": 95,
            "tp1": 110, "tp1_exit_pct": 50, "direction": "LONG",
            "confidence": 7, "leverage": 5,
            "entry_trigger": "candle_close_above",
            "entry_trigger_price": 100.0,
            "invalidation_price": 96.0,
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

    @patch("scanner.execution.entry_monitor.get_open_positions", return_value=[])
    @patch("scanner.execution.entry_monitor.fetch_current_price", return_value=97.0)
    @patch("scanner.execution.entry_monitor._send_pending_alert", return_value=True)
    def test_saves_screener_direction(self, mock_alert, mock_price, mock_pos, ce_config, conn):
        """screener_direction (from setup_signal) persisted so counter-trend
        flag survives the scan → trigger gap."""
        from scanner.models import (
            Direction,
            MarketContext,
            ScanResult,
            SetupSignal,
            SetupType,
            Structure,
        )

        ctx = MarketContext(
            structure=Structure.BULLISH, swings=[], sr_levels=[], msbs=[],
            range_bounds=None, atr=5.0, current_price=100.0, fib_levels=None,
        )
        setup = SetupSignal(
            coin="BTC", setup_type=SetupType.A, direction=Direction.LONG,
            confidence=0.7, key_level=100.0, context=ctx,
            description="same-way level",
        )
        # LLM picks SHORT — counter-trend vs screener's LONG
        result = ScanResult(
            scan_id="s1", coin="BTC", timestamp=datetime.now(timezone.utc),
            setup_signal=setup, decision="CONFIRM", confluence_score=None,
            analysis_text=None, parsed_decision=None, chart_paths=[],
            cost_usd=0.0, machine_block={
                "entry_low": 99, "entry_high": 101, "stop_loss": 105,
                "tp1": 90, "tp1_exit_pct": 50, "direction": "SHORT",
                "confidence": 7, "leverage": 5,
                "entry_trigger": "candle_close_below",
                "entry_trigger_price": 99.0, "invalidation_price": 102.0,
                "expiry_candles": 12,
            },
        )

        save_conditional_entry(result, ce_config, conn)

        row = conn.execute(
            "SELECT screener_direction FROM pending_entries WHERE scan_id = ?",
            ("s1",),
        ).fetchone()
        assert row["screener_direction"] == "LONG"

    def test_expiry_capped(self, ce_config, conn):
        from scanner.models import ScanResult

        ce_config.conditional_entry_max_expiry = 10

        mb = {
            "entry_low": 99, "entry_high": 101, "stop_loss": 95,
            "tp1": 110, "tp1_exit_pct": 50, "direction": "LONG",
            "confidence": 7, "leverage": 5,
            "entry_trigger": "candle_close_above",
            "entry_trigger_price": 100.0,
            "invalidation_price": 96.0,
            "expiry_candles": 24,
        }

        result = ScanResult(scan_id="s1", coin="BTC", timestamp=datetime.now(timezone.utc),
                            setup_signal=None, decision="CONFIRM", confluence_score=None,
                            analysis_text=None, parsed_decision=None, chart_paths=[],
                            cost_usd=0.0, machine_block=mb)

        with patch("scanner.execution.entry_monitor._send_pending_alert", return_value=True), \
             patch("scanner.execution.entry_monitor.fetch_current_price", return_value=97.0), \
             patch("scanner.execution.entry_monitor.get_open_positions", return_value=[]):
            save_conditional_entry(result, ce_config, conn)

        row = conn.execute("SELECT expiry_candles FROM pending_entries").fetchone()
        assert row["expiry_candles"] == 10

    def test_skips_when_position_open_exchange(self, ce_config, conn):
        """Pending creation blocked when coin already has an open position
        (exchange truth path). LLM-confirmed signal still logged in scans
        upstream — only the pending row + alerts are suppressed here."""
        from scanner.models import ScanResult

        result = ScanResult(
            scan_id="s_dup", coin="BTC",
            timestamp=datetime.now(timezone.utc),
            setup_signal=None, decision="CONFIRM",
            confluence_score=None, analysis_text=None,
            parsed_decision=None, chart_paths=[], cost_usd=0.0,
            machine_block={
                "entry_low": 99, "entry_high": 101, "stop_loss": 95,
                "tp1": 110, "tp1_exit_pct": 50, "direction": "LONG",
                "confidence": 7, "leverage": 5,
                "entry_trigger": "candle_close_above",
                "entry_trigger_price": 100.0,
                "invalidation_price": 96.0,
                "expiry_candles": 12,
            },
        )

        positions = [{"coin": "BTC", "direction": "LONG", "size": 0.1,
                      "entry_price": 50000.0, "unrealized_pnl": 0.0, "leverage": 5}]
        with patch("scanner.execution.entry_monitor.get_open_positions",
                   return_value=positions), \
             patch("scanner.execution.entry_monitor._send_pending_alert") as mock_alert:
            saved = save_conditional_entry(result, ce_config, conn)

        assert saved is False
        assert not has_active_pending("BTC", conn, "hyperliquid", "testnet")
        mock_alert.assert_not_called()

    def test_position_fetch_failure_live_blocks(self, ce_config, conn):
        """In live mode, exchange-position fetch failure blocks pending
        creation (mirrors executor.py reconciliation_error behavior)."""
        from scanner.models import ScanResult

        ce_config.mode = "live"

        result = ScanResult(
            scan_id="s_live", coin="BTC",
            timestamp=datetime.now(timezone.utc),
            setup_signal=None, decision="CONFIRM",
            confluence_score=None, analysis_text=None,
            parsed_decision=None, chart_paths=[], cost_usd=0.0,
            machine_block={
                "entry_low": 99, "entry_high": 101, "stop_loss": 95,
                "tp1": 110, "tp1_exit_pct": 50, "direction": "LONG",
                "confidence": 7, "leverage": 5,
                "entry_trigger": "candle_close_above",
                "entry_trigger_price": 100.0,
                "invalidation_price": 96.0,
                "expiry_candles": 12,
            },
        )

        with patch("scanner.execution.entry_monitor.get_open_positions",
                   side_effect=RuntimeError("exchange unreachable")), \
             patch("scanner.execution.entry_monitor._send_pending_alert") as mock_alert:
            saved = save_conditional_entry(result, ce_config, conn)

        assert saved is False
        assert not has_active_pending("BTC", conn, "hyperliquid", "live")
        mock_alert.assert_not_called()

    @patch("scanner.execution.entry_monitor.fetch_current_price", return_value=97.0)
    @patch("scanner.execution.entry_monitor._send_pending_alert", return_value=True)
    def test_position_fetch_failure_testnet_falls_back_to_db(
        self, mock_alert, mock_price, ce_config, conn,
    ):
        """In testnet, exchange-fetch failure falls back to DB has_open_position.
        With empty DB the save proceeds normally."""
        from scanner.models import ScanResult

        result = ScanResult(
            scan_id="s_testnet", coin="BTC",
            timestamp=datetime.now(timezone.utc),
            setup_signal=None, decision="CONFIRM",
            confluence_score=None, analysis_text=None,
            parsed_decision=None, chart_paths=[], cost_usd=0.0,
            machine_block={
                "entry_low": 99, "entry_high": 101, "stop_loss": 95,
                "tp1": 110, "tp1_exit_pct": 50, "direction": "LONG",
                "confidence": 7, "leverage": 5,
                "entry_trigger": "candle_close_above",
                "entry_trigger_price": 100.0,
                "invalidation_price": 96.0,
                "expiry_candles": 12,
            },
        )

        with patch("scanner.execution.entry_monitor.get_open_positions",
                   side_effect=RuntimeError("transient ccxt error")):
            saved = save_conditional_entry(result, ce_config, conn)

        assert saved is True
        assert has_active_pending("BTC", conn, "hyperliquid", "testnet")

    def test_skips_invalid_machine_block(self, ce_config, conn):
        """Save-time backstop rejects mb's that violate validate_execution_params
        (LONG inv<SL or SHORT inv>SL). Without this, ZEC/NEAR-shape blocks
        become pendings that deterministically TRIGGER_FAILED at execute time."""
        from scanner.models import ScanResult

        # SHORT with invalidation_price (338) > stop_loss (335) — exact
        # ZEC 2026-04-29 shape.
        result = ScanResult(
            scan_id="s_invalid", coin="ZEC",
            timestamp=datetime.now(timezone.utc),
            setup_signal=None, decision="CONFIRM",
            confluence_score=None, analysis_text=None,
            parsed_decision=None, chart_paths=[], cost_usd=0.0,
            machine_block={
                "entry_low": 327, "entry_high": 329, "stop_loss": 335.0,
                "tp1": 310, "tp1_exit_pct": 50, "direction": "SHORT",
                "confidence": 7, "leverage": 5,
                "entry_trigger": "candle_close_below",
                "entry_trigger_price": 328.0,
                "invalidation_price": 338.0,
                "expiry_candles": 12,
            },
        )

        with patch("scanner.execution.entry_monitor.get_open_positions",
                   return_value=[]), \
             patch("scanner.execution.entry_monitor._send_pending_alert") as mock_alert:
            saved = save_conditional_entry(result, ce_config, conn)

        assert saved is False
        assert not has_active_pending("ZEC", conn, "hyperliquid", "testnet")
        mock_alert.assert_not_called()


# ---------------------------------------------------------------------------
# _reconstruct_scan_result tests — R:R preservation + counter-trend flag
# ---------------------------------------------------------------------------

class TestReconstructScanResult:

    def test_reanchors_sl_and_tp_by_trigger_delta(self):
        """When price has moved between scan and trigger, SL/TP shift by the
        same delta so R:R is preserved instead of collapsing."""
        pending = _make_pending(
            coin="BTC", direction="LONG",
            trigger="candle_close_above", trigger_price=100.0,
            inv_price=95.0,
        )
        mb = json.loads(pending["machine_block"])
        # Original plan: entry 99-101 (mid=100), SL 95, TP1 110 → 2R
        mb["entry_low"] = 99.0
        mb["entry_high"] = 101.0
        mb["stop_loss"] = 95.0
        mb["tp1"] = 110.0
        mb["tp2"] = 115.0
        pending["machine_block"] = json.dumps(mb)

        # Trigger fires at 105 (5 above original mid)
        result = _reconstruct_scan_result(pending, trigger_close=105.0)

        # Entry zone rebuilt around trigger
        new_mb = result.machine_block
        assert new_mb["entry_low"] == pytest.approx(105.0 * 0.995)
        assert new_mb["entry_high"] == pytest.approx(105.0 * 1.005)
        # SL/TP shifted by delta=+5 → R still = |105-100| = 5, reward = |115-105| = 10
        assert new_mb["stop_loss"] == pytest.approx(100.0)
        assert new_mb["tp1"] == pytest.approx(115.0)
        assert new_mb["tp2"] == pytest.approx(120.0)

    def test_setup_signal_carries_screener_direction(self):
        """Counter-trend detection works after trigger: reconstructed
        setup_signal.direction matches the stored screener_direction."""
        pending = _make_pending(coin="BTC", direction="SHORT")
        pending["screener_direction"] = "LONG"

        result = _reconstruct_scan_result(pending, trigger_close=99.0)

        assert result.setup_signal is not None
        assert result.setup_signal.direction.value == "LONG"
        # mb direction is still SHORT (LLM decision)
        assert result.machine_block["direction"] == "SHORT"

    def test_missing_screener_direction_leaves_setup_signal_none(self):
        """Legacy pending rows without screener_direction still reconstruct
        (setup_signal=None) — executor falls back to non-counter-trend."""
        pending = _make_pending(coin="BTC", direction="LONG")
        pending["screener_direction"] = None

        result = _reconstruct_scan_result(pending, trigger_close=100.0)

        assert result.setup_signal is None


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

        # Use a recent timestamp so the stale-candle guard doesn't skip it
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        ts = now.isoformat()
        candle = {"timestamp": ts,
                  "open": 97.0, "high": 98.0, "low": 96.0, "close": 97.0}

        with patch("scanner.execution.entry_monitor._fetch_1h_candle", return_value=candle), \
             patch("scanner.execution.entry_monitor.has_open_position", return_value=False):
            check_pending_entries(ce_config, conn)  # First call: processes
            check_pending_entries(ce_config, conn)  # Second call: skips (same hour)

        row = conn.execute(
            "SELECT candles_checked, last_checked_hour FROM pending_entries WHERE pending_id = 'p_btc'"
        ).fetchone()
        assert row["candles_checked"] == 1
        assert row["last_checked_hour"] == ts[:13]

    @patch("scanner.execution.entry_monitor.is_1h_boundary", return_value=True)
    def test_different_hour_processes_again(self, mock_boundary, ce_config, conn):
        """Two calls with different candle hours should increment twice."""
        from scanner.execution.entry_monitor import check_pending_entries

        save_pending_entry(_make_pending(trigger_price=100.0, inv_price=90.0, expiry=12), conn)

        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        prev = now - timedelta(hours=1)
        candle_prev = {"timestamp": prev.isoformat(),
                       "open": 97.0, "high": 98.0, "low": 96.0, "close": 97.0}
        candle_now = {"timestamp": now.isoformat(),
                      "open": 97.0, "high": 98.0, "low": 96.0, "close": 97.5}

        with patch("scanner.execution.entry_monitor.has_open_position", return_value=False):
            with patch("scanner.execution.entry_monitor._fetch_1h_candle", return_value=candle_prev):
                check_pending_entries(ce_config, conn)
            with patch("scanner.execution.entry_monitor._fetch_1h_candle", return_value=candle_now):
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

        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        ts = now.isoformat()
        candle = {"timestamp": ts,
                  "open": 97.0, "high": 98.0, "low": 96.0, "close": 97.0}

        with patch("scanner.execution.entry_monitor._fetch_1h_candle", return_value=candle), \
             patch("scanner.execution.entry_monitor.has_open_position", return_value=False):
            check_pending_entries(ce_config, conn)

        row = conn.execute(
            "SELECT candles_checked, last_checked_hour FROM pending_entries WHERE pending_id = 'p_btc'"
        ).fetchone()
        assert row["candles_checked"] == 1
        assert row["last_checked_hour"] == ts[:13]


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
