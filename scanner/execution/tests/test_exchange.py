"""Tests for authenticated exchange connection + queries."""

from unittest.mock import MagicMock, patch

import pytest

from scanner.execution.config import ExecutionConfig
from scanner.execution.exchange import (
    cancel_order,
    fetch_balance,
    fetch_current_price,
    get_auth_exchange,
    get_open_positions,
    reset_auth_exchange,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset auth exchange singleton between tests."""
    reset_auth_exchange()
    yield
    reset_auth_exchange()


@pytest.fixture
def testnet_config():
    return ExecutionConfig(
        mode="testnet",
        private_key="0xtest_key",
        wallet_address="0xtest_wallet",
    )


@pytest.fixture
def live_config():
    return ExecutionConfig(
        mode="live",
        private_key="0xlive_key",
        wallet_address="0xlive_wallet",
    )


class TestGetAuthExchange:
    @patch("scanner.execution.exchange.SafeHyperliquid")
    def test_testnet_sets_sandbox(self, mock_cls, testnet_config):
        mock_exchange = MagicMock()
        mock_cls.return_value = mock_exchange

        _result = get_auth_exchange(testnet_config)
        mock_exchange.set_sandbox_mode.assert_called_once_with(True)
        mock_exchange.load_markets.assert_called_once()

    @patch("scanner.execution.exchange.SafeHyperliquid")
    def test_live_no_sandbox(self, mock_cls, live_config):
        mock_exchange = MagicMock()
        mock_cls.return_value = mock_exchange

        _result = get_auth_exchange(live_config)
        mock_exchange.set_sandbox_mode.assert_not_called()

    @patch("scanner.execution.exchange.SafeHyperliquid")
    def test_singleton_cached(self, mock_cls, testnet_config):
        mock_exchange = MagicMock()
        mock_cls.return_value = mock_exchange

        first = get_auth_exchange(testnet_config)
        second = get_auth_exchange(testnet_config)
        assert first is second
        assert mock_cls.call_count == 1

    @patch("scanner.execution.exchange.SafeHyperliquid")
    def test_reset_clears_singleton(self, mock_cls, testnet_config):
        mock_exchange = MagicMock()
        mock_cls.return_value = mock_exchange

        get_auth_exchange(testnet_config)
        reset_auth_exchange()
        get_auth_exchange(testnet_config)
        assert mock_cls.call_count == 2

    @patch("scanner.execution.exchange.SafeHyperliquid")
    def test_different_configs_get_different_instances(self, mock_cls, testnet_config, live_config):
        mock_cls.side_effect = [MagicMock(), MagicMock()]

        first = get_auth_exchange(testnet_config)
        second = get_auth_exchange(live_config)
        assert first is not second
        assert mock_cls.call_count == 2


class TestFetchCurrentPrice:
    @patch("scanner.execution.exchange.get_auth_exchange")
    def test_returns_last_price(self, mock_get_auth, testnet_config):
        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker.return_value = {"last": 62150.5}
        mock_get_auth.return_value = mock_exchange

        price = fetch_current_price("BTC", testnet_config)
        assert price == 62150.5


class TestFetchBalance:
    @patch("scanner.execution.exchange.get_auth_exchange")
    def test_returns_usdc_free(self, mock_get_auth, testnet_config):
        mock_exchange = MagicMock()
        mock_exchange.fetch_balance.return_value = {"USDC": {"free": 10543.21}}
        mock_get_auth.return_value = mock_exchange

        balance = fetch_balance(testnet_config)
        assert balance == 10543.21


class TestGetOpenPositions:
    @patch("scanner.execution.exchange.get_auth_exchange")
    def test_returns_positions(self, mock_get_auth, testnet_config):
        mock_exchange = MagicMock()
        mock_exchange.fetch_positions.return_value = [
            {
                "symbol": "BTC/USDC:USDC",
                "contracts": 0.1,
                "side": "long",
                "entryPrice": 62000,
                "unrealizedPnl": 150,
                "leverage": 5,
            },
            {
                "symbol": "ETH/USDC:USDC",
                "contracts": 0,  # closed — should be filtered
                "side": "long",
                "entryPrice": 3000,
                "unrealizedPnl": 0,
                "leverage": 3,
            },
        ]
        mock_get_auth.return_value = mock_exchange

        positions = get_open_positions(testnet_config)
        assert len(positions) == 1
        assert positions[0]["coin"] == "BTC"
        assert positions[0]["direction"] == "LONG"


class TestCancelOrder:
    @patch("scanner.execution.exchange.get_auth_exchange")
    def test_cancel_success(self, mock_get_auth, testnet_config):
        mock_exchange = MagicMock()
        mock_get_auth.return_value = mock_exchange

        assert cancel_order("order123", "BTC/USDC:USDC", testnet_config) is True
        mock_exchange.cancel_order.assert_called_once_with("order123", "BTC/USDC:USDC")

    @patch("scanner.execution.exchange.get_auth_exchange")
    def test_cancel_failure(self, mock_get_auth, testnet_config):
        mock_exchange = MagicMock()
        mock_exchange.cancel_order.side_effect = Exception("order not found")
        mock_get_auth.return_value = mock_exchange

        assert cancel_order("bad_id", "BTC/USDC:USDC", testnet_config) is False


