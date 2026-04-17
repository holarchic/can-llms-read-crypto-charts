"""Tests for trade execution orchestrator."""

import sqlite3
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from ccxt.base.errors import DuplicateOrderId

from scanner.execution.config import ExecutionConfig
from scanner.execution.executor import (
    ExecutionResult,
    _place_stop_loss,
    _place_take_profit,
    current_open_notional,
    execute_trade,
    preflight_check,
    send_skip_alert,
    validate_execution_params,
)
from scanner.execution.storage import ensure_orders_table, get_orders_for_scan, save_order
from scanner.models import (
    Direction,
    MarketContext,
    ScanResult,
    SetupSignal,
    SetupType,
    Structure,
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
        btc_vol_gate_enabled=False,
        discord_ops_webhook_url="https://discord.com/ops",
        discord_urgent_webhook_url="https://discord.com/urgent",
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
        "confidence": 7,
    }
    base.update(overrides)
    return base


_UNSET = object()


def _make_scan_result(machine_block=_UNSET, coin="BTC", screener_direction=None, **overrides):
    mb = _make_machine_block() if machine_block is _UNSET else machine_block
    # Default screener direction matches machine_block to avoid unintended counter-trend
    if screener_direction is None:
        mb_dir = mb.get("direction", "LONG").upper() if mb else "LONG"
        screener_direction = Direction.LONG if mb_dir == "LONG" else Direction.SHORT
    ctx = MarketContext(
        structure=Structure.BULLISH,
        swings=[], sr_levels=[], msbs=[],
        range_bounds=None, atr=500, current_price=62000,
        fib_levels=None,
    )
    signal = SetupSignal(
        coin=coin,
        setup_type=SetupType.A,
        direction=screener_direction,
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


def _mock_exchange_success(filled=None):
    """Return mock responses for entry, SL, TP orders."""
    resp = {
        "id": "exch_order_123",
        "average": 62050,
        "price": 62050,
        "status": "closed",
    }
    if filled is not None:
        resp["filled"] = filled
    return resp


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

    def test_long_stop_above_entry(self):
        errors = validate_execution_params(_make_machine_block(stop_loss=62000))
        assert any("LONG stop" in e for e in errors)

    def test_short_stop_below_entry(self):
        mb = _make_machine_block(direction="SHORT", stop_loss=61000, tp1=59000, tp2=56000)
        errors = validate_execution_params(mb)
        assert any("SHORT stop" in e for e in errors)

    def test_long_tp1_below_entry(self):
        errors = validate_execution_params(_make_machine_block(tp1=61000))
        assert any("LONG tp1" in e for e in errors)

    def test_short_tp1_above_entry(self):
        mb = _make_machine_block(direction="SHORT", stop_loss=64000, tp1=62500, tp2=56000)
        errors = validate_execution_params(mb)
        assert any("SHORT tp1" in e for e in errors)

    def test_entry_low_above_entry_high(self):
        errors = validate_execution_params(_make_machine_block(entry_low=63000, entry_high=61000))
        assert any("entry_low" in e for e in errors)

    def test_long_tp2_below_tp1(self):
        errors = validate_execution_params(_make_machine_block(tp2=64000))
        assert any("LONG tp2" in e for e in errors)

    def test_valid_short(self):
        mb = _make_machine_block(direction="SHORT", stop_loss=64000, tp1=59000, tp2=56000)
        errors = validate_execution_params(mb)
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
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
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
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        mb = _make_machine_block(direction="SHORT", stop_loss=64200, tp1=59000, tp2=56000)
        sr = _make_scan_result(machine_block=mb)

        result = execute_trade(sr, exec_config, scanner_config, conn)
        assert result.executed is True

    def test_skip_stale_signal(self, conn, exec_config, scanner_config):
        from datetime import timedelta
        old_detected = datetime.now(timezone.utc) - timedelta(minutes=60)
        sr = _make_scan_result(detected_at=old_detected)
        result = execute_trade(sr, exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.skip_reason == "stale_signal"

    def test_skip_stale_signal_falls_back_to_timestamp(self, conn, exec_config, scanner_config):
        """When detected_at is None, staleness uses timestamp."""
        from datetime import timedelta
        old_timestamp = datetime.now(timezone.utc) - timedelta(minutes=60)
        sr = _make_scan_result(timestamp=old_timestamp, detected_at=None)
        result = execute_trade(sr, exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.skip_reason == "stale_signal"

    def test_fresh_signal_passes_staleness(self, conn, exec_config, scanner_config):
        from datetime import timedelta
        fresh_detected = datetime.now(timezone.utc) - timedelta(minutes=10)
        sr = _make_scan_result(detected_at=fresh_detected, machine_block=None)
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

    @patch("scanner.execution.executor.calculate_position")
    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_skip_insufficient_margin(self, mock_killed, mock_price, mock_balance,
                                       mock_auth, mock_discord, mock_calc,
                                       conn, exec_config, scanner_config):
        """Margin required exceeds available balance."""
        mock_pos = MagicMock()
        mock_pos.quantity = 0.1
        mock_pos.margin_required = 50  # >> balance of 10
        mock_calc.return_value = mock_pos
        mock_auth.return_value = MagicMock()

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.skip_reason == "insufficient_margin"

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
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is True
        assert result.sl_failed is False
        assert result.sl_order_id != ""

    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_sl_fails_emergency_close_succeeds(self, mock_killed, mock_price, mock_balance,
                                               mock_auth, mock_discord, conn, exec_config, scanner_config):
        """SL fails 3x -> emergency close succeeds -> early return, no TPs."""
        mock_exchange = MagicMock()
        mock_exchange.create_order.side_effect = [
            _mock_exchange_success(),  # entry
            Exception("timeout"),      # SL attempt 1
            Exception("timeout"),      # SL attempt 2
            Exception("timeout"),      # SL attempt 3
            # Emergency close — needs get_auth_exchange again
            _mock_exchange_success(),  # emergency close
        ]
        mock_exchange.fetch_ticker.return_value = {"last": 62000}
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is True
        assert result.sl_failed is True
        assert result.sl_order_id == ""

        # FAILED SL saved to DB
        orders = get_orders_for_scan("test123", conn)
        sl_orders = [o for o in orders if o["order_type"] == "stop_loss"]
        assert len(sl_orders) == 1
        assert sl_orders[0]["status"] == "FAILED"
        assert sl_orders[0]["trigger_price"] is not None

        # Emergency close saved to DB
        ec_orders = [o for o in orders if o["order_type"] == "emergency_close"]
        assert len(ec_orders) == 1
        assert ec_orders[0]["status"] == "FILLED"

        # No TP orders placed (early return)
        tp_orders = [o for o in orders if o["order_type"] in ("tp1", "tp2")]
        assert len(tp_orders) == 0

    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_sl_fails_emergency_close_also_fails(self, mock_killed, mock_price, mock_balance,
                                                 mock_auth, mock_discord, conn, exec_config, scanner_config):
        """SL fails 3x, emergency close fails 3x -> URGENT alert, returns early (no TPs)."""
        mock_exchange = MagicMock()
        mock_exchange.create_order.side_effect = [
            _mock_exchange_success(),  # entry
            Exception("timeout"),      # SL attempt 1
            Exception("timeout"),      # SL attempt 2
            Exception("timeout"),      # SL attempt 3
            # Emergency close fails
            Exception("exchange error"),  # emergency attempt 1
            Exception("exchange error"),  # emergency attempt 2
            Exception("exchange error"),  # emergency attempt 3
        ]
        mock_exchange.fetch_ticker.return_value = {"last": 62000}
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.sl_failed is True

        # FAILED SL saved, no emergency close
        orders = get_orders_for_scan("test123", conn)
        sl_orders = [o for o in orders if o["order_type"] == "stop_loss"]
        assert len(sl_orders) == 1
        assert sl_orders[0]["status"] == "FAILED"
        ec_orders = [o for o in orders if o["order_type"] == "emergency_close"]
        assert len(ec_orders) == 0

        # No TP orders placed (early return)
        tp_orders = [o for o in orders if o["order_type"] in ("tp1", "tp2")]
        assert len(tp_orders) == 0

        # URGENT alert mentions emergency close failure
        urgent_calls = [c for c in mock_discord.call_args_list
                       if "Emergency close also FAILED" in str(c)]
        assert len(urgent_calls) > 0

    @patch("scanner.execution.executor.time.sleep")
    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_sl_fails_emergency_close_duplicate_order_id(
        self, mock_killed, mock_price, mock_balance, mock_auth, mock_discord,
        mock_sleep, conn, exec_config, scanner_config,
    ):
        """SL fails 3x, emergency close attempt 1 times out, attempt 2 DuplicateOrderId → treated as filled."""
        mock_exchange = MagicMock()
        mock_exchange.create_order.side_effect = [
            _mock_exchange_success(),         # entry
            Exception("timeout"),             # SL attempt 1
            Exception("timeout"),             # SL attempt 2
            Exception("timeout"),             # SL attempt 3
            Exception("timeout"),             # emergency close attempt 1 (times out)
            DuplicateOrderId("duplicate"),    # emergency close attempt 2
        ]
        mock_exchange.fetch_ticker.return_value = {"last": 62000}
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is True
        assert result.sl_failed is True

        # Emergency close saved as FILLED despite DuplicateOrderId
        orders = get_orders_for_scan("test123", conn)
        ec_orders = [o for o in orders if o["order_type"] == "emergency_close"]
        assert len(ec_orders) == 1
        assert ec_orders[0]["status"] == "FILLED"
        assert ec_orders[0]["fill_price"] == 62000

        # No false "Emergency close FAILED" URGENT alert
        urgent_failed_calls = [c for c in mock_discord.call_args_list
                               if "Emergency close also FAILED" in str(c)]
        assert len(urgent_failed_calls) == 0

    @patch("scanner.execution.executor.time.sleep")
    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_sl_rate_limited_skips_to_emergency_close(self, mock_killed, mock_price, mock_balance,
                                                       mock_auth, mock_discord, mock_sleep,
                                                       conn, exec_config, scanner_config):
        """SL rate-limited on attempt 1 -> breaks immediately -> emergency close succeeds."""
        mock_exchange = MagicMock()
        mock_exchange.create_order.side_effect = [
            _mock_exchange_success(),  # entry
            Exception('hyperliquid {"status":"err","response":"Too many cumulative requests sent (12530 > 12268)"}'),
            _mock_exchange_success(),  # emergency close
        ]
        mock_exchange.fetch_ticker.return_value = {"last": 62000}
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is True
        assert result.sl_failed is True

        # Only 1 SL attempt (not 3) — rate limit broke the retry loop
        sl_calls = [c for c in mock_exchange.create_order.call_args_list
                    if "stopLossPrice" in str(c)]
        assert len(sl_calls) == 1

        # Emergency close saved
        orders = get_orders_for_scan("test123", conn)
        ec_orders = [o for o in orders if o["order_type"] == "emergency_close"]
        assert len(ec_orders) == 1
        assert ec_orders[0]["status"] == "FILLED"

    @patch("scanner.execution.executor.time.sleep")
    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_emergency_close_rate_limited_uses_longer_backoff(self, mock_killed, mock_price, mock_balance,
                                                              mock_auth, mock_discord, mock_sleep,
                                                              conn, exec_config, scanner_config):
        """Emergency close rate-limited -> uses 20s backoff instead of 5s."""
        rl_exc = Exception('Too many cumulative requests sent (12530 > 12268)')
        mock_exchange = MagicMock()
        mock_exchange.create_order.side_effect = [
            _mock_exchange_success(),  # entry
            rl_exc,                    # SL attempt 1 (rate limited, breaks)
            rl_exc,                    # emergency attempt 1
            rl_exc,                    # emergency attempt 2
            rl_exc,                    # emergency attempt 3
        ]
        mock_exchange.fetch_ticker.return_value = {"last": 62000}
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.sl_failed is True

        # Emergency close used 20s backoff (2 sleeps for 3 attempts)
        sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
        assert 20 in sleep_calls

    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_sl_duplicate_cloid_found_on_exchange(self, mock_killed, mock_price, mock_balance,
                                                    mock_auth, mock_discord, conn, exec_config, scanner_config):
        """SL attempt 1 times out, attempt 2 raises DuplicateOrderId -> query finds real order."""
        existing_sl = {"id": "real-sl-123", "clientOrderId": "test-cloid", "status": "open"}
        mock_exchange = MagicMock()
        mock_exchange.create_order.side_effect = [
            _mock_exchange_success(),        # entry
            Exception("timeout"),            # SL attempt 1
            DuplicateOrderId("duplicate"),   # SL attempt 2 — prior attempt actually succeeded
        ]
        mock_exchange.fetch_open_orders.return_value = [existing_sl]
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        # Patch make_cloid to return the matching clientOrderId
        with patch("scanner.execution.executor.make_cloid", return_value="test-cloid"):
            result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is True
        assert result.sl_order_id != ""

        # No URGENT alert — SL found on exchange
        urgent_calls = [c for c in mock_discord.call_args_list
                       if "URGENT" in str(c)]
        assert len(urgent_calls) == 0

    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_sl_duplicate_cloid_not_found_triggers_urgent(self, mock_killed, mock_price, mock_balance,
                                                           mock_auth, mock_discord, conn, exec_config, scanner_config):
        """SL DuplicateOrderId but order not found on exchange -> emergency close -> URGENT alert."""
        mock_exchange = MagicMock()
        mock_exchange.create_order.side_effect = [
            _mock_exchange_success(),        # entry
            Exception("timeout"),            # SL attempt 1
            DuplicateOrderId("duplicate"),   # SL attempt 2
            # Emergency close fails (mock runs out of side_effect)
        ]
        mock_exchange.fetch_open_orders.return_value = []  # Not found
        mock_exchange.fetch_ticker.return_value = {"last": 62000}
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is False  # SL + emergency close both failed → early return

        # URGENT alert — SL ambiguous / emergency close failure
        urgent_calls = [c for c in mock_discord.call_args_list
                       if "URGENT" in str(c)]
        assert len(urgent_calls) >= 1

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
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
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
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert mock_discord.called

    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_zero_fill_aborts(self, mock_killed, mock_price, mock_balance,
                              mock_auth, mock_discord, conn, exec_config, scanner_config):
        """Zero fill from IOC → CANCELLED entry, no SL/TP placed."""
        mock_exchange = MagicMock()
        mock_exchange.create_order.return_value = _mock_exchange_success(filled=0)
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)

        assert result.executed is False
        assert result.skip_reason == "entry_unfilled"
        assert result.sl_order_id == ""
        assert result.tp1_order_id == ""
        # Entry saved as CANCELLED
        orders = get_orders_for_scan("test123", conn)
        assert len(orders) == 1
        assert orders[0]["status"] == "CANCELLED"
        assert orders[0]["quantity"] == 0

    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_partial_fill_scales_sl_tp(self, mock_killed, mock_price, mock_balance,
                                       mock_auth, mock_discord, conn, exec_config, scanner_config):
        """Partial fill scales SL and TP quantities proportionally."""
        mock_exchange = MagicMock()
        # Entry returns half fill; SL, TP1, TP2 succeed
        # quantity ≈ 0.04545; we say exchange filled 0.02 (partial)
        filled = 0.02
        mock_exchange.create_order.side_effect = [
            _mock_exchange_success(filled=filled),  # entry
            _mock_exchange_success(),  # SL
            _mock_exchange_success(),  # TP1
            _mock_exchange_success(),  # TP2
        ]
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)

        assert result.executed is True
        orders = get_orders_for_scan("test123", conn)
        entry = next(o for o in orders if o["order_type"] == "entry")
        sl = next(o for o in orders if o["order_type"] == "stop_loss")
        tp1 = next(o for o in orders if o["order_type"] == "tp1")
        tp2 = next(o for o in orders if o["order_type"] == "tp2")

        # Entry and SL use filled_qty
        assert entry["quantity"] == filled
        assert sl["quantity"] == filled
        # TP1 + TP2 = filled_qty (50/50 split)
        assert abs(tp1["quantity"] + tp2["quantity"] - filled) < 1e-10
        assert abs(tp1["quantity"] - tp2["quantity"]) < 1e-10  # equal split

        # Ops alert sent for partial fill
        ops_calls = [c for c in mock_discord.call_args_list
                     if "Partial fill" in str(c)]
        assert len(ops_calls) > 0

    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_full_fill_uses_confirmed_qty(self, mock_killed, mock_price, mock_balance,
                                          mock_auth, mock_discord, conn, exec_config, scanner_config):
        """Full fill uses confirmed quantity from exchange, not requested."""
        mock_exchange = MagicMock()
        # Sizing gives ≈0.04545; exchange confirms exact same (full fill)
        # We compute expected qty from sizing: risk=100, stop_dist=2200
        expected_qty = 100.0 / 2200.0
        mock_exchange.create_order.return_value = _mock_exchange_success(filled=expected_qty)
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)

        assert result.executed is True
        orders = get_orders_for_scan("test123", conn)
        entry = next(o for o in orders if o["order_type"] == "entry")
        sl = next(o for o in orders if o["order_type"] == "stop_loss")

        assert abs(entry["quantity"] - expected_qty) < 1e-10
        assert abs(sl["quantity"] - expected_qty) < 1e-10


class TestReanchoredExitPrices:
    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_reanchored_logged_on_fill(self, mock_killed, mock_price, mock_balance,
                                       mock_auth, mock_discord, conn, exec_config,
                                       scanner_config, caplog):
        """Fill at different price from entry_mid → reanchored exits logged."""
        mock_exchange = MagicMock()
        # entry_mid = (61800+62200)/2 = 62000, but fill at 62300 (slippage)
        resp = _mock_exchange_success()
        resp["average"] = 62300
        resp["price"] = 62300
        mock_exchange.create_order.return_value = resp
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        import logging
        with caplog.at_level(logging.INFO, logger="scanner.execution.executor"):
            result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)

        assert result.executed is True
        reanchor_logs = [r for r in caplog.records if "reanchored" in r.message]
        assert len(reanchor_logs) == 1
        # delta = 62300 - 62000 = +300
        assert "+300" in reanchor_logs[0].message

    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_reanchored_prices_used_in_orders(self, mock_killed, mock_price, mock_balance,
                                               mock_auth, mock_discord, conn, exec_config,
                                               scanner_config):
        """SL/TP trigger_prices in saved orders reflect reanchored levels."""
        mock_exchange = MagicMock()
        resp = _mock_exchange_success()
        resp["average"] = 62300  # delta = +300
        resp["price"] = 62300
        mock_exchange.create_order.return_value = resp
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is True

        from scanner.execution.storage import get_order
        # SL: 59800 + 300 = 60100
        sl = get_order(result.sl_order_id, conn)
        assert float(sl["trigger_price"]) == 60100

        # TP1: 65000 + 300 = 65300
        tp1 = get_order(result.tp1_order_id, conn)
        assert float(tp1["trigger_price"]) == 65300

        # TP2: 68000 + 300 = 68300
        tp2 = get_order(result.tp2_order_id, conn)
        assert float(tp2["trigger_price"]) == 68300

    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_zero_delta_preserves_original_prices(self, mock_killed, mock_price, mock_balance,
                                                    mock_auth, mock_discord, conn, exec_config,
                                                    scanner_config, caplog):
        """Fill exactly at entry_mid → no reanchor log, original prices used."""
        mock_exchange = MagicMock()
        resp = _mock_exchange_success()
        resp["average"] = 62000  # exactly entry_mid
        resp["price"] = 62000
        mock_exchange.create_order.return_value = resp
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        import logging
        with caplog.at_level(logging.INFO, logger="scanner.execution.executor"):
            result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)

        assert result.executed is True
        reanchor_logs = [r for r in caplog.records if "reanchored" in r.message]
        assert len(reanchor_logs) == 0

        from scanner.execution.storage import get_order
        sl = get_order(result.sl_order_id, conn)
        assert float(sl["trigger_price"]) == 59800  # original


class TestClientOrderId:
    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_cloid_passed_to_all_orders(self, mock_killed, mock_price, mock_balance,
                                         mock_auth, mock_discord, conn, exec_config, scanner_config):
        """All create_order calls include clientOrderId in params."""
        mock_exchange = MagicMock()
        mock_exchange.create_order.return_value = _mock_exchange_success()
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_exchange.fetch_ticker.return_value = {"last": 62000}
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        execute_trade(_make_scan_result(), exec_config, scanner_config, conn)

        # Entry, SL, TP1, TP2 = 4 create_order calls
        for call in mock_exchange.create_order.call_args_list:
            params = call[0][-1] if len(call[0]) > 5 else call.kwargs.get("params", call[0][-1])
            assert "clientOrderId" in params, f"Missing clientOrderId in {call}"
            assert params["clientOrderId"].startswith("0x")


class TestRoundQuantity:
    def test_rounds_via_exchange_precision(self):
        from scanner.execution.executor import _round_quantity
        mock_exchange = MagicMock()
        mock_exchange.amount_to_precision.return_value = "0.66"
        result = _round_quantity(mock_exchange, "HYPE/USDC:USDC", 0.6666666)
        assert result == 0.66
        mock_exchange.amount_to_precision.assert_called_once_with("HYPE/USDC:USDC", 0.6666666)

    def test_passthrough_returns_float(self):
        from scanner.execution.executor import _round_quantity
        mock_exchange = MagicMock()
        mock_exchange.amount_to_precision.return_value = "0.142"
        result = _round_quantity(mock_exchange, "TAO/USDC:USDC", 0.14285714)
        assert result == 0.142


def _make_btc_4h_df(last_range: float, atr_approx: float, rows: int = 30):
    """Build a minimal BTC 4H DataFrame where the just-closed candle (iloc[-2])
    has the given range and ATR ≈ atr_approx. The final row is the in-progress
    candle and is dropped by the executor before range-check."""
    base = 60000.0
    data = {
        "Open": [base] * rows,
        "High": [base + atr_approx] * rows,
        "Low": [base] * rows,
        "Close": [base + atr_approx * 0.5] * rows,
        "Volume": [100.0] * rows,
    }
    # The "just-closed" candle is the second-to-last row — the last row is
    # the in-progress candle and gets dropped by the executor.
    data["High"][-2] = base + last_range
    data["Low"][-2] = base

    dates = pd.date_range("2024-01-01", periods=rows, freq="4h", tz="UTC")
    return pd.DataFrame(data, index=dates)


class TestBtcVolatilityGate:
    @pytest.fixture
    def vol_gate_config(self, exec_config):
        """exec_config with BTC vol gate explicitly enabled."""
        exec_config.btc_vol_gate_enabled = True
        return exec_config

    @patch("scanner.data.fetch_candles")
    def test_btc_vol_gate_blocks_extreme(self, mock_fetch, conn, vol_gate_config, scanner_config):
        """BTC 4H range > 2x ATR → skip with btc_volatility_gate."""
        # ATR ≈ 1000 (all candles have range=1000), last candle range = 2500 (2.5x)
        mock_fetch.return_value = _make_btc_4h_df(last_range=2500, atr_approx=1000)
        sr = _make_scan_result()
        result = execute_trade(sr, vol_gate_config, scanner_config, conn)
        assert result.executed is False
        assert result.skip_reason == "btc_volatility_gate"
        assert "BTC 4H range" in result.error

    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.data.fetch_candles")
    def test_btc_vol_gate_passes_normal(self, mock_fetch, mock_price, conn, vol_gate_config, scanner_config):
        """Normal BTC volatility (0.5x ATR) → execution continues past gate."""
        # ATR ≈ 1000, last candle range = 500 (0.5x) — should pass
        mock_fetch.return_value = _make_btc_4h_df(last_range=500, atr_approx=1000)
        sr = _make_scan_result(machine_block=None)
        result = execute_trade(sr, vol_gate_config, scanner_config, conn)
        # Should pass the gate and fail on next check (no_machine_block)
        assert result.skip_reason == "no_machine_block"

    @patch("scanner.data.fetch_candles", side_effect=Exception("API timeout"))
    def test_btc_vol_gate_fails_open(self, mock_fetch, conn, vol_gate_config, scanner_config):
        """fetch_candles raises → gate fails open, execution continues."""
        sr = _make_scan_result(machine_block=None)
        result = execute_trade(sr, vol_gate_config, scanner_config, conn)
        # Gate failed open → proceeds to next check
        assert result.skip_reason == "no_machine_block"


class TestExchangeReconciliation:
    """Step 0.6: exchange position reconciliation before DB-based limits."""

    @patch("scanner.execution.executor.count_open_positions", return_value=1)
    @patch("scanner.execution.executor.get_open_positions")
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_exchange_reconciliation_blocks_duplicate(self, mock_killed, mock_price,
                                                      mock_positions, mock_db_count,
                                                      conn, exec_config, scanner_config):
        """Exchange shows BTC position → skip as duplicate (any direction, net position)."""
        mock_positions.return_value = [
            {"coin": "BTC", "direction": "LONG", "size": 0.05, "entry_price": 61000,
             "unrealized_pnl": 100, "leverage": 5},
        ]
        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.skip_reason == "duplicate_position"

    @patch("scanner.execution.executor.count_open_positions", return_value=1)
    @patch("scanner.execution.executor.get_open_positions")
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_exchange_reconciliation_blocks_opposite_direction(self, mock_killed, mock_price,
                                                               mock_positions, mock_db_count,
                                                               conn, exec_config, scanner_config):
        """Exchange shows BTC SHORT → skip BTC LONG (HL net position would close the SHORT)."""
        mock_positions.return_value = [
            {"coin": "BTC", "direction": "SHORT", "size": 0.05, "entry_price": 63000,
             "unrealized_pnl": -50, "leverage": 5},
        ]
        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.skip_reason == "duplicate_position"

    @patch("scanner.execution.executor.count_open_positions", return_value=2)
    @patch("scanner.execution.executor.get_open_positions")
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_exchange_reconciliation_respects_limit(self, mock_killed, mock_price,
                                                     mock_positions, mock_db_count,
                                                     conn, exec_config, scanner_config):
        """Exchange shows max positions → skip even if DB disagrees."""
        exec_config.max_concurrent_positions = 2
        mock_positions.return_value = [
            {"coin": "ETH", "direction": "LONG", "size": 1.0, "entry_price": 3000,
             "unrealized_pnl": 50, "leverage": 3},
            {"coin": "SOL", "direction": "SHORT", "size": 10.0, "entry_price": 150,
             "unrealized_pnl": -20, "leverage": 5},
        ]
        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.skip_reason == "position_limit"

    @patch("scanner.execution.executor.get_open_positions", side_effect=Exception("API timeout"))
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_exchange_reconciliation_falls_back_to_db(self, mock_killed, mock_price,
                                                       mock_positions, conn, exec_config, scanner_config):
        """API raises → DB checks still run (position_limit from DB with max=0)."""
        exec_config.max_concurrent_positions = 0
        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.skip_reason == "position_limit"

    @patch("scanner.execution.executor.get_open_positions")
    def test_exchange_reconciliation_blocks_mismatch(self, mock_positions, conn, exec_config,
                                                      scanner_config):
        """Exchange vs DB count differs → trade blocked."""
        mock_positions.return_value = [
            {"coin": "ETH", "direction": "LONG", "size": 1.0, "entry_price": 3000,
             "unrealized_pnl": 50, "leverage": 3},
        ]
        # DB has 0 positions, exchange has 1 — mismatch
        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.skip_reason == "position_mismatch"
        assert "DB positions=0 vs exchange=1" in result.error


class TestFailPolicy:
    """Fail-open in testnet, fail-closed in live."""

    @patch("scanner.data.fetch_candles", side_effect=Exception("API timeout"))
    def test_btc_vol_gate_error_blocks_live(self, mock_fetch, conn, scanner_config):
        """BTC vol gate fetch failure in live mode → blocks trade."""
        live_config = ExecutionConfig(
            enabled=True, mode="live", private_key="0xtest",
            wallet_address="0xwallet", risk_pct=1.0,
            btc_vol_gate_enabled=True,
        )
        result = execute_trade(_make_scan_result(), live_config, scanner_config, conn)
        assert result.skip_reason == "btc_vol_gate_error"

    @patch("scanner.data.fetch_candles", side_effect=Exception("API timeout"))
    def test_btc_vol_gate_error_passes_testnet(self, mock_fetch, conn, exec_config, scanner_config):
        """BTC vol gate fetch failure in testnet → proceeds past gate."""
        exec_config.btc_vol_gate_enabled = True
        sr = _make_scan_result(machine_block=None)
        result = execute_trade(sr, exec_config, scanner_config, conn)
        assert result.skip_reason != "btc_vol_gate_error"

    @patch("scanner.execution.executor.get_open_positions", side_effect=Exception("API timeout"))
    def test_reconciliation_error_blocks_live(self, mock_positions, conn, scanner_config):
        """Exchange position fetch failure in live mode → blocks trade."""
        live_config = ExecutionConfig(
            enabled=True, mode="live", private_key="0xtest",
            wallet_address="0xwallet", risk_pct=1.0,
            btc_vol_gate_enabled=False,
        )
        result = execute_trade(_make_scan_result(), live_config, scanner_config, conn)
        assert result.skip_reason == "reconciliation_error"

    @patch("scanner.execution.executor.get_open_positions", side_effect=Exception("API timeout"))
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_reconciliation_error_passes_testnet(self, mock_killed, mock_price,
                                                  mock_positions, conn, exec_config, scanner_config):
        """Exchange position fetch failure in testnet → proceeds past gate."""
        exec_config.max_concurrent_positions = 0
        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        # Hits position_limit from DB checks, not reconciliation_error
        assert result.skip_reason == "position_limit"


class TestUnprotectedPositionGuard:
    """Block all entries if any open position lacks a PLACED stop loss."""

    @patch("scanner.execution.executor.get_open_positions")
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_unprotected_position_blocks_entry(self, mock_killed, mock_price,
                                                mock_positions, conn, exec_config, scanner_config):
        """Entry FILLED + SL CANCELLED → skip_reason='unprotected_position'."""
        # Exchange shows 1 position (matches DB count of 1)
        mock_positions.return_value = [
            {"coin": "ETH", "direction": "LONG", "size": 1.0, "entry_price": 3000,
             "unrealized_pnl": 0, "leverage": 3},
        ]
        save_order({
            "order_id": "e1", "scan_id": "s_existing", "coin": "ETH", "direction": "LONG",
            "order_type": "entry", "status": "FILLED", "entry_price": 3000,
            "quantity": 1.0, "exchange": "hyperliquid", "execution_mode": "testnet",
            "created_at": "2026-03-05T00:00:00+00:00",
        }, conn)
        save_order({
            "order_id": "sl1", "scan_id": "s_existing", "coin": "ETH", "direction": "LONG",
            "order_type": "stop_loss", "status": "CANCELLED",
            "quantity": 1.0, "exchange": "hyperliquid", "execution_mode": "testnet",
            "created_at": "2026-03-05T00:00:00+00:00",
        }, conn)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.skip_reason == "unprotected_position"

    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_open_positions", return_value=[])
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_protected_position_allows_entry(self, mock_killed, mock_price, mock_balance,
                                              mock_auth, mock_positions, mock_discord,
                                              conn, exec_config, scanner_config):
        """Entry FILLED + SL PLACED → proceeds past the guard."""
        save_order({
            "order_id": "e1", "scan_id": "s_existing", "coin": "ETH", "direction": "LONG",
            "order_type": "entry", "status": "FILLED", "entry_price": 3000,
            "quantity": 1.0, "exchange": "hyperliquid", "execution_mode": "testnet",
            "created_at": "2026-03-05T00:00:00+00:00",
        }, conn)
        save_order({
            "order_id": "sl1", "scan_id": "s_existing", "coin": "ETH", "direction": "LONG",
            "order_type": "stop_loss", "status": "PLACED",
            "quantity": 1.0, "exchange": "hyperliquid", "execution_mode": "testnet",
            "created_at": "2026-03-05T00:00:00+00:00",
        }, conn)

        mock_exchange = MagicMock()
        mock_exchange.create_order.return_value = _mock_exchange_success()
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.skip_reason != "unprotected_position"

    @patch("scanner.execution.executor.get_open_positions", return_value=[])
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_closed_position_not_unprotected(self, mock_killed, mock_price,
                                              mock_positions, conn, exec_config, scanner_config):
        """Entry FILLED + SL FILLED (position closed) → not considered unprotected."""
        save_order({
            "order_id": "e1", "scan_id": "s_closed", "coin": "ETH", "direction": "LONG",
            "order_type": "entry", "status": "FILLED", "entry_price": 3000,
            "quantity": 1.0, "exchange": "hyperliquid", "execution_mode": "testnet",
            "created_at": "2026-03-05T00:00:00+00:00",
        }, conn)
        save_order({
            "order_id": "sl1", "scan_id": "s_closed", "coin": "ETH", "direction": "LONG",
            "order_type": "stop_loss", "status": "FILLED", "fill_price": 2800,
            "quantity": 1.0, "exchange": "hyperliquid", "execution_mode": "testnet",
            "created_at": "2026-03-05T00:00:00+00:00",
            "filled_at": "2026-03-05T01:00:00+00:00",
        }, conn)

        # Position is fully closed → should proceed past unprotected guard
        sr = _make_scan_result(machine_block=None)
        result = execute_trade(sr, exec_config, scanner_config, conn)
        # Should fail on next check (no_machine_block), not unprotected_position
        assert result.skip_reason == "no_machine_block"


class TestCapitalBounds:
    """Zero balance guard and max notional per trade."""

    @patch("scanner.execution.executor.get_open_positions", return_value=[])
    @patch("scanner.execution.executor.fetch_balance", return_value=0)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_skip_zero_balance(self, mock_killed, mock_price, mock_balance,
                                mock_positions, conn, exec_config, scanner_config):
        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.skip_reason == "zero_balance"

    @patch("scanner.execution.executor.get_open_positions", return_value=[])
    @patch("scanner.execution.executor.fetch_balance", return_value=-100)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_skip_negative_balance(self, mock_killed, mock_price, mock_balance,
                                    mock_positions, conn, exec_config, scanner_config):
        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.skip_reason == "zero_balance"

    @patch("scanner.execution.executor.calculate_position")
    @patch("scanner.execution.executor.get_open_positions", return_value=[])
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_skip_notional_limit(self, mock_killed, mock_price, mock_balance,
                                  mock_positions, mock_calc, conn, exec_config, scanner_config):
        """Notional exceeds max → skip."""
        exec_config.max_notional_usd = 500.0
        mock_pos = MagicMock()
        mock_pos.quantity = 0.1
        mock_pos.margin_required = 100
        mock_pos.notional_value = 6200  # >> 500
        mock_calc.return_value = mock_pos

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.skip_reason == "notional_limit"

    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_open_positions", return_value=[])
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_notional_limit_disabled_by_default(self, mock_killed, mock_price, mock_balance,
                                                 mock_auth, mock_positions, mock_discord,
                                                 conn, exec_config, scanner_config):
        """max_notional_usd=0 (default) → no notional limit check."""
        assert exec_config.max_notional_usd == 0.0
        mock_exchange = MagicMock()
        mock_exchange.create_order.return_value = _mock_exchange_success()
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is True


class TestCurrentOpenNotional:
    @patch("scanner.execution.executor.get_auth_exchange")
    def test_uses_live_price(self, mock_auth, conn, exec_config):
        """Open position priced at current market → notional > entry-based."""
        # Insert a filled entry at 62000, qty 0.1
        save_order({
            "order_id": "not1", "scan_id": "s1", "coin": "BTC", "direction": "LONG",
            "order_type": "entry", "status": "FILLED", "entry_price": 62000,
            "quantity": 0.1, "exchange": "hyperliquid", "execution_mode": "testnet",
            "created_at": "2026-03-05T00:00:00+00:00",
        }, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker.return_value = {"last": 70000}
        mock_auth.return_value = mock_exchange

        result = current_open_notional(exec_config, conn)
        # 70000 * 0.1 = 7000 (vs entry-based 62000 * 0.1 = 6200)
        assert result == pytest.approx(7000.0)

    @patch("scanner.execution.executor.get_auth_exchange")
    def test_fallback_on_ticker_error(self, mock_auth, conn, exec_config):
        """Ticker fetch fails → uses entry_price as fallback."""
        save_order({
            "order_id": "not2", "scan_id": "s2", "coin": "ETH", "direction": "LONG",
            "order_type": "entry", "status": "FILLED", "entry_price": 3000,
            "quantity": 1.0, "exchange": "hyperliquid", "execution_mode": "testnet",
            "created_at": "2026-03-05T00:00:00+00:00",
        }, conn)

        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker.side_effect = Exception("API timeout")
        mock_auth.return_value = mock_exchange

        result = current_open_notional(exec_config, conn)
        assert result == pytest.approx(3000.0)  # 3000 * 1.0

    def test_no_positions_returns_zero(self, conn, exec_config):
        """No open positions → 0.0."""
        result = current_open_notional(exec_config, conn)
        assert result == 0.0

    def test_exposure_uses_exchange_positions_when_available(self, conn, exec_config):
        """Exchange positions provided → uses exchange truth, skips DB/ticker."""
        exchange_positions = [
            {"coin": "ETH", "direction": "LONG", "size": 2.0, "entry_price": 3000},
            {"coin": "SOL", "direction": "SHORT", "size": -50.0, "entry_price": 150},
        ]
        result = current_open_notional(exec_config, conn, exchange_positions=exchange_positions)
        # abs(2.0)*3000 + abs(-50.0)*150 = 6000 + 7500 = 13500
        assert result == pytest.approx(13500.0)

    def test_exposure_falls_back_to_db(self, conn, exec_config):
        """No exchange positions → falls back to DB (empty DB = 0.0)."""
        result = current_open_notional(exec_config, conn, exchange_positions=None)
        assert result == 0.0


class TestOpenOrderCap:
    """Step 7.6: open order cap prevents HL rejection at 1000 orders."""

    @patch("scanner.execution.executor.get_exchange_open_orders")
    @patch("scanner.execution.executor.get_open_positions", return_value=[])
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_skip_open_order_limit(self, mock_killed, mock_price, mock_positions,
                                    mock_orders, conn, exec_config, scanner_config):
        """200 open orders → skip with open_order_limit."""
        exec_config.max_open_orders = 200
        mock_orders.return_value = [{}] * 200
        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is False
        assert result.skip_reason == "open_order_limit"

    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_exchange_open_orders")
    @patch("scanner.execution.executor.get_open_positions", return_value=[])
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_open_order_below_threshold_passes(self, mock_killed, mock_price, mock_balance,
                                                mock_auth, mock_positions, mock_orders,
                                                mock_discord, conn, exec_config, scanner_config):
        """50 open orders < 200 cap → passes to next check."""
        exec_config.max_open_orders = 200
        mock_orders.return_value = [{}] * 50
        mock_exchange = MagicMock()
        mock_exchange.create_order.return_value = _mock_exchange_success()
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)
        assert result.executed is True

    @patch("scanner.execution.executor.get_exchange_open_orders", side_effect=Exception("API timeout"))
    @patch("scanner.execution.executor.get_open_positions", return_value=[])
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_order_count_error_blocks_live(self, mock_killed, mock_price, mock_positions,
                                            mock_orders, conn, scanner_config):
        """API raises in live mode → blocks trade."""
        live_config = ExecutionConfig(
            enabled=True, mode="live", private_key="0xtest",
            wallet_address="0xwallet", risk_pct=1.0,
            btc_vol_gate_enabled=False, max_open_orders=200,
        )
        result = execute_trade(_make_scan_result(), live_config, scanner_config, conn)
        assert result.skip_reason == "order_count_error"

    @patch("scanner.execution.executor.get_exchange_open_orders", side_effect=Exception("API timeout"))
    @patch("scanner.execution.executor.get_open_positions", return_value=[])
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_order_count_error_passes_testnet(self, mock_killed, mock_price, mock_positions,
                                               mock_orders, conn, exec_config, scanner_config):
        """API raises in testnet → proceeds past check."""
        exec_config.max_open_orders = 200
        sr = _make_scan_result(machine_block=None)
        result = execute_trade(sr, exec_config, scanner_config, conn)
        # Should pass the cap check and fail on next (no_machine_block via balance → ...)
        assert result.skip_reason != "order_count_error"


class TestSkipAlertSeverity:
    """Critical vs routine skip alert formatting."""

    @patch("scanner.execution.executor.requests.post")
    def test_skip_alert_critical_includes_everyone(self, mock_post, exec_config):
        """position_mismatch → @everyone + BLOCKED."""
        mock_post.return_value = MagicMock(status_code=204)
        er = ExecutionResult(coin="BTC", direction="LONG", skip_reason="position_mismatch",
                             error="DB=0 vs exchange=1")
        send_skip_alert(er, exec_config)
        call_args = mock_post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "@everyone" in body["content"]
        assert "BLOCKED" in body["content"]

    @patch("scanner.execution.executor.requests.post")
    def test_skip_alert_routine_embed(self, mock_post, exec_config):
        """position_limit → embed with SKIPPED title, no @everyone."""
        mock_post.return_value = MagicMock(status_code=204)
        er = ExecutionResult(coin="ETH", direction="SHORT", skip_reason="position_limit")
        send_skip_alert(er, exec_config)
        call_args = mock_post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "content" not in body  # embed, not plain text
        embed = body["embeds"][0]
        assert "SKIPPED" in embed["title"]
        assert "ETH" in embed["title"]
        assert embed["color"] == 0x999999
        field_values = {f["name"]: f["value"] for f in embed["fields"]}
        assert field_values["Reason"] == "position_limit"

    @patch("scanner.execution.executor.requests.post")
    def test_skip_alert_routine_with_detail(self, mock_post, exec_config):
        """Routine skip with error detail → Detail field in embed."""
        mock_post.return_value = MagicMock(status_code=204)
        er = ExecutionResult(coin="JUP", direction="SHORT", skip_reason="price_outside_zone",
                             error="Price 0.156 outside [0.16, 0.162]")
        send_skip_alert(er, exec_config)
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        field_values = {f["name"]: f["value"] for f in embed["fields"]}
        assert "price_outside_zone" in field_values["Reason"]
        assert "0.156" in field_values["Detail"]


class TestPreflightCheck:
    """Startup self-checks before first trade."""

    @patch("scanner.execution.executor.is_killed", return_value=False)
    @patch("scanner.execution.executor.fetch_balance", return_value=5000.0)
    def test_preflight_all_pass(self, mock_balance, mock_killed, exec_config):
        failures = preflight_check(exec_config)
        assert failures == []

    @patch("scanner.execution.executor.is_killed", return_value=False)
    @patch("scanner.execution.executor.fetch_balance", side_effect=Exception("auth failed"))
    def test_preflight_exchange_failure(self, mock_balance, mock_killed, exec_config):
        failures = preflight_check(exec_config)
        assert len(failures) == 1
        assert "Exchange auth/balance failed" in failures[0]

    @patch("scanner.execution.executor.is_killed", return_value=False)
    @patch("scanner.execution.executor.fetch_balance", return_value=0.0)
    def test_preflight_zero_balance(self, mock_balance, mock_killed, exec_config):
        failures = preflight_check(exec_config)
        assert len(failures) == 1
        assert "Zero balance" in failures[0]


class TestOrderTypeMarket:
    """SL and TP trigger orders must use type='market' for ccxt isMarket=True."""

    @patch("scanner.execution.executor.get_auth_exchange")
    def test_stop_loss_uses_market_type(self, mock_auth, exec_config):
        """_place_stop_loss passes type='market' with stopLossPrice in params."""
        mock_exchange = MagicMock()
        mock_exchange.create_order.return_value = {"id": "sl_123"}
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange

        result = _place_stop_loss(exec_config, "BTC/USDC:USDC", "sell", 0.1, 59800.0)

        assert result is not None
        call_args = mock_exchange.create_order.call_args
        assert call_args[0][1] == "market"
        assert call_args[1] is not None or call_args[0][5]["stopLossPrice"] == 59800.0

    @patch("scanner.execution.executor.get_auth_exchange")
    def test_take_profit_uses_market_type(self, mock_auth, exec_config):
        """_place_take_profit passes type='market' with takeProfitPrice in params."""
        mock_exchange = MagicMock()
        mock_exchange.create_order.return_value = {"id": "tp_123"}
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange

        result = _place_take_profit(exec_config, "BTC/USDC:USDC", "sell", 0.05, 65000.0)

        assert result is not None
        call_args = mock_exchange.create_order.call_args
        assert call_args[0][1] == "market"
        assert call_args[1] is not None or call_args[0][5]["takeProfitPrice"] == 65000.0


class TestBatchExitOrders:
    """Tests for batch exit order placement (SL + TP1 + TP2 in one API call)."""

    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_batch_success_places_all_exits(self, mock_killed, mock_price, mock_balance,
                                             mock_auth, mock_discord, conn, exec_config, scanner_config):
        """When batch succeeds, SL + TP1 + TP2 placed via create_orders, not create_order."""
        mock_exchange = MagicMock()
        # Entry via create_order
        mock_exchange.create_order.return_value = _mock_exchange_success()
        # Batch exit via create_orders — returns list of order responses
        mock_exchange.create_orders.return_value = [
            {"id": "sl-batch-1", "status": "open"},
            {"id": "tp1-batch-1", "status": "open"},
            {"id": "tp2-batch-1", "status": "open"},
        ]
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)

        assert result.executed is True
        assert result.sl_order_id != ""
        assert result.tp1_order_id != ""
        assert result.tp2_order_id != ""

        # create_orders called once (batch), create_order called once (entry only)
        assert mock_exchange.create_orders.call_count == 1
        assert mock_exchange.create_order.call_count == 1  # entry only

        # All orders saved to DB
        orders = get_orders_for_scan("test123", conn)
        order_types = {o["order_type"] for o in orders}
        assert {"entry", "stop_loss", "tp1", "tp2"} == order_types

    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_batch_failure_falls_back_to_sequential(self, mock_killed, mock_price, mock_balance,
                                                     mock_auth, mock_discord, conn, exec_config,
                                                     scanner_config):
        """When batch raises, falls back to sequential create_order calls."""
        mock_exchange = MagicMock()
        mock_exchange.create_orders.side_effect = Exception("batch API error")
        # Sequential fallback: entry, SL, TP1, TP2
        mock_exchange.create_order.return_value = _mock_exchange_success()
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)

        assert result.executed is True
        assert result.sl_order_id != ""
        # Sequential used: entry + SL + TP1 + TP2 = 4 create_order calls
        assert mock_exchange.create_order.call_count == 4

    @patch("scanner.execution.executor.requests.post")
    @patch("scanner.execution.executor.get_auth_exchange")
    @patch("scanner.execution.executor.fetch_balance", return_value=10000)
    @patch("scanner.execution.executor.fetch_current_price", return_value=62000)
    @patch("scanner.execution.executor.is_killed", return_value=False)
    def test_batch_partial_sl_missing_falls_back(self, mock_killed, mock_price, mock_balance,
                                                   mock_auth, mock_discord, conn, exec_config,
                                                   scanner_config):
        """Batch returns TP but no SL id → OID recovery finds nothing → falls back to sequential SL."""
        mock_exchange = MagicMock()
        # Batch returns: SL missing id, TP1 OK, TP2 OK
        mock_exchange.create_orders.return_value = [
            {"status": "error"},  # SL failed — no 'id'
            {"id": "tp1-batch-1", "status": "open"},
            {"id": "tp2-batch-1", "status": "open"},
        ]
        mock_exchange.fetch_open_orders.return_value = []  # OID recovery finds nothing
        # Sequential fallback for SL
        mock_exchange.create_order.side_effect = [
            _mock_exchange_success(),  # entry
            _mock_exchange_success(),  # SL sequential fallback
        ]
        mock_exchange.amount_to_precision.side_effect = lambda s, q: str(q)
        mock_auth.return_value = mock_exchange
        mock_discord.return_value = MagicMock(status_code=204)

        result = execute_trade(_make_scan_result(), exec_config, scanner_config, conn)

        assert result.executed is True
        assert result.sl_order_id != ""
        assert result.tp1_order_id != ""
        assert result.tp2_order_id != ""
        # SL placed sequentially (entry + SL = 2 create_order calls)
        assert mock_exchange.create_order.call_count == 2
