"""Tests for trade execution orchestrator."""

import sqlite3
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

from scanner.execution.executor import (
    execute_trade,
    validate_execution_params,
    ExecutionResult,
)
from scanner.execution.config import ExecutionConfig
from scanner.execution.storage import ensure_orders_table, get_orders_for_scan
from scanner.models import (
    ScanResult, SetupSignal, SetupType, Direction,
    MarketContext, Structure, SRLevel,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_orders_table(c)
    return c


@pytest.fixture
def exec_config():
    return ExecutionConfig(
        enabled=True,
        mode="testnet",
        private_key="0xtest",
        wallet_address="0xwallet",
        risk_pct=1.0,
        discord_execution_webhook_url="https://discord.com/test",
    )


@pytest.fixture
def scanner_config():
    from scanner.config import ScannerConfig
    return ScannerConfig()


def _make_machine_block(**overrides):
    base = {
        "direction": "LONG",
        "entry_low": 61800,
        "entry_high": 62200,
        "stop_loss": 59800,
        "tp1": 65000,
        "tp2": 68000,
        "leverage": 5,
        "tp1_exit_pct": 50,
        "tp2_exit_pct": 50,
    }
    base.update(overrides)
    return base


_UNSET = object()


def _make_scan_result(machine_block=_UNSET, coin="BTC", **overrides):
    mb = _make_machine_block() if machine_block is _UNSET else machine_block
    ctx = MarketContext(
        structure=Structure.BULLISH,
        swings=[], sr_levels=[], msbs=[],
        range_bounds=None, atr=500, current_price=62000,
        fib_levels=None,
    )
    signal = SetupSignal(
        coin=coin,
        setup_type=SetupType.A,
        direction=Direction.LONG,
        confidence=0.8,
        key_level=61800,
        context=ctx,
        description="Test signal",
        max_leverage=20,
    )
    defaults = {
        "scan_id": "test123",
        "coin": coin,
        "timestamp": datetime.now(timezone.utc),
        "setup_signal": signal,
        "decision": "CONFIRM",
        "confluence_score": "8/10",
        "analysis_text": "Test analysis",
        "parsed_decision": {},
        "chart_paths": [],
        "cost_usd": 0.09,
        "machine_block": mb,
    }
    defaults.update(overrides)
    return ScanResult(**defaults)


def _mock_exchange_success():
    """Return mock responses for entry, SL, TP orders."""
    return {
        "id": "exch_order_123",
        "average": 62050,
        "price": 62050,
        "status": "closed",
    }


class TestValidateExecutionParams:
    def test_valid_long(self):
        assert validate_execution_params(_make_machine_block()) == []

    def test_missing_direction(self):
        mb = _make_machine_block()
        del mb["direction"]
        errors = validate_execution_params(mb)
        assert any("direction" in e.lower() for e in errors)

    def test_invalid_direction(self):
        errors = validate_execution_params(_make_machine_block(direction="UP"))
        assert any("direction" in e.lower() for e in errors)

    def test_invalid_leverage(self):
        errors = validate_execution_params(_make_machine_block(leverage=-1))
        assert any("leverage" in e.lower() for e in errors)

    def test_validates_exit_pcts_over_100(self):
        errors = validate_execution_params(_make_machine_block(tp1_exit_pct=70, tp2_exit_pct=50))
        assert any("exit pcts sum" in e for e in errors)

    def test_validates_exit_pct_negative(self):
        errors = validate_execution_params(_make_machine_block(tp1_exit_pct=-10))
        assert any("tp1_exit_pct must be positive" in e for e in errors)

    def test_validates_tp2_exit_pct_negative(self):
        errors = validate_execution_params(_make_machine_block(tp2_exit_pct=-5))
        assert any("tp2_exit_pct must be non-negative" in e for e in errors)

    def test_valid_exit_pcts_no_errors(self):
        errors = validate_execution_params(_make_machine_block(tp1_exit_pct=60, tp2_exit_pct=40))
        assert errors == []


class TestExecuteTrade:
    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_happy_path_long(self, mock_killed, mock_price, mock_balance,
                             mock_auth, mock_discord, conn, exec_config, scanner_config):
        mock_exchange = MagicMock()
        mock_exchange.create_order.return_value = _mock_exchange_success()
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)

        assert result.executed is True
        assert result.entry_order_id != ""
        assert result.sl_order_id != ""
        assert result.position_size is not None

        orders = get_orders_for_scan("test123", conn)
        assert len(orders) >= 2  # entry + SL at minimum

        # entry_price should be actual fill price, not zone midpoint
        entry_order = next(o for o in orders if o["order_type"] == "entry")
        assert entry_order["entry_price"] == 62050  # from mock average, not (61800+62200)/2

    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_happy_path_short(self, mock_killed, mock_price, mock_balance,
                              mock_auth, mock_discord, conn, exec_config, scanner_config):
        mock_exchange = MagicMock()
        mock_exchange.create_order.return_value = _mock_exchange_success()
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        mb = _make_machine_block(direction="SHORT", stop_loss=64200, tp1=59000, tp2=56000)
        sr = _make_scan_result(machine_block=mb)

        result = execute_trade(sr, exec_config, scanner_config, conn)
        assert result.executed is True

    def test_skip_stale_signal(self, conn, exec_config, scanner_config):
        from datetime import timedelta
        old_timestamp = datetime.now(timezone.utc) - timedelta(minutes=35)
        sr = _make_scan_result(timestamp=old_timestamp)
        result = execute_trade(sr, exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.skip_reason == "stale_signal"

    def test_fresh_signal_passes_staleness(self, conn, exec_config, scanner_config):
        from datetime import timedelta
        fresh_timestamp = datetime.now(timezone.utc) - timedelta(minutes=10)
        sr = _make_scan_result(timestamp=fresh_timestamp, machine_block=None)
        result = execute_trade(sr, exec_config, scanner_config, conn)
        # Should pass staleness but fail on no_machine_block (next check)
        assert result.skip_reason == "no_machine_block"

    def test_skip_no_machine_block(self, conn, exec_config, scanner_config):
        sr = _make_scan_result(machine_block=None)
        result = execute_trade(sr, exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.skip_reason == "no_machine_block"

    def test_skip_invalid_params(self, conn, exec_config, scanner_config):
        sr = _make_scan_result(machine_block={"direction": "INVALID"})
        result = execute_trade(sr, exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.skip_reason == "invalid_params"

    @patch("scanner.execution.executor.fetch_current_price", return_value=56000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_skip_sanity_fail(self, mock_killed, mock_price, conn, exec_config, scanner_config):
        """Price too far from entry zone fails sanity check."""
        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.skip_reason == "sanity_fail"

    @patch("scanner.execution.executor.fetch_current_price", return_value=65000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_skip_price_outside_zone(self, mock_killed, mock_price, conn, exec_config, scanner_config):
        """Price beyond entry zone + direction buffer [61800, 62200] + 2% LONG buffer."""
        # 65000 is well above zone + buffer ceiling (~63440), so should still skip
        exec_config.entry_price_tolerance_pct = 10.0
        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.skip_reason == "price_outside_zone"

    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=True)
    def test_skip_kill_switch(self, mock_killed, mock_price, conn, exec_config, scanner_config):
        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.skip_reason == "kill_switch"

    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_skip_position_limit(self, mock_killed, mock_price, conn, exec_config, scanner_config):
        exec_config.max_concurrent_positions = 0  # limit is 0
        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.skip_reason == "position_limit"

    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_sl_retry_succeeds(self, mock_killed, mock_price, mock_balance,
                               mock_auth, mock_discord, conn, exec_config, scanner_config):
        """SL fails first two attempts, succeeds on third."""
        mock_exchange = MagicMock()
        # Entry succeeds, SL fails twice then succeeds
        mock_exchange.create_order.side_effect = [
            _mock_exchange_success(),  # entry
            Exception("timeout"),      # SL attempt 1
            Exception("timeout"),      # SL attempt 2
            _mock_exchange_success(),  # SL attempt 3
            _mock_exchange_success(),  # TP1
            _mock_exchange_success(),  # TP2
        ]
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is True
        assert result.sl_order_id != ""

    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_sl_retry_exhausted_urgent_alert(self, mock_killed, mock_price, mock_balance,
                                             mock_auth, mock_discord, conn, exec_config, scanner_config):
        """All SL retries fail -> URGENT alert, trade still marked executed."""
        mock_exchange = MagicMock()
        mock_exchange.create_order.side_effect = [
            _mock_exchange_success(),  # entry
            Exception("timeout"),      # SL attempt 1
            Exception("timeout"),      # SL attempt 2
            Exception("timeout"),      # SL attempt 3
            _mock_exchange_success(),  # TP1
            _mock_exchange_success(),  # TP2
        ]
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is True
        assert result.sl_order_id == ""  # No SL placed

        # Verify URGENT alert was sent
        urgent_calls = [c for c in mock_discord.call_args_list
                       if "URGENT" in str(c)]
        assert len(urgent_calls) > 0

    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_tp_failure_non_fatal(self, mock_killed, mock_price, mock_balance,
                                  mock_auth, mock_discord, conn, exec_config, scanner_config):
        """TP placement fails -> trade still executed, SL present."""
        mock_exchange = MagicMock()
        mock_exchange.create_order.side_effect = [
            _mock_exchange_success(),  # entry
            _mock_exchange_success(),  # SL
            Exception("TP failed"),    # TP1
            Exception("TP failed"),    # TP2
        ]
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is True
        assert result.sl_order_id != ""
        assert result.tp1_order_id == ""  # TP failed, non-fatal

    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_discord_alert_sent(self, mock_killed, mock_price, mock_balance,
                                mock_auth, mock_discord, conn, exec_config, scanner_config):
        mock_exchange = MagicMock()
        mock_exchange.create_order.return_value = _mock_exchange_success()
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert mock_discord.called
