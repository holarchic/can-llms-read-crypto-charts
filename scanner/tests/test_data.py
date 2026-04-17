"""Tests for exchange data fetching — thread safety."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from scanner import data


class TestGetExchangeThreadSafe:
    def setup_method(self):
        """Reset module-level state between tests."""
        data._exchanges.clear()
        data._exchange_locks.clear()

    def test_get_exchange_thread_safe(self):
        """10 threads call get_exchange() simultaneously — only 1 load_markets() call."""
        mock_exchange = MagicMock()
        mock_cls = MagicMock(return_value=mock_exchange)

        with patch("scanner.data.ccxt") as mock_ccxt:
            mock_ccxt.hyperliquid = mock_cls
            with patch("scanner.data.get_exchange_meta", return_value={
                "ccxt_class": "hyperliquid",
            }):
                barrier = threading.Barrier(10)
                results = []

                def _call():
                    barrier.wait()
                    ex = data.get_exchange("hyperliquid")
                    results.append(ex)

                threads = [threading.Thread(target=_call) for _ in range(10)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

        assert len(results) == 10
        # All threads got the same instance
        assert all(r is mock_exchange for r in results)
        # load_markets called exactly once
        mock_exchange.load_markets.assert_called_once()

    def test_fetch_candles_concurrent(self):
        """Concurrent fetch_candles calls with mocked exchange don't raise."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = [
            [1000000, 1.0, 2.0, 0.5, 1.5, 100.0],
            [1000001, 1.5, 2.5, 1.0, 2.0, 200.0],
        ]
        # Pre-populate exchange + lock
        data._exchanges["hyperliquid"] = mock_exchange
        data._exchange_locks["hyperliquid"] = threading.Lock()

        errors = []

        def _fetch(coin):
            try:
                data.fetch_candles(coin, "4h", 2)
            except Exception as e:
                errors.append(e)

        with patch("scanner.data.get_exchange_meta", return_value={
            "ccxt_class": "hyperliquid",
            "symbol_template": "{coin}/USDC:USDC",
        }):
            with patch("scanner.data.time.sleep"):
                threads = [threading.Thread(target=_fetch, args=(f"COIN{i}",)) for i in range(5)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

        assert errors == []
        assert mock_exchange.fetch_ohlcv.call_count == 5


class TestGetTestnetCoins:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        """Reset the process-wide testnet coin cache between tests."""
        data._testnet_coins_cache.clear()
        yield
        data._testnet_coins_cache.clear()

    @patch("scanner.data.ccxt")
    @patch("scanner.data.get_exchange_meta", return_value={"ccxt_class": "hyperliquid"})
    def test_returns_coin_names(self, mock_meta, mock_ccxt):
        mock_exchange = MagicMock()
        mock_exchange.markets = {
            "BTC/USDC:USDC": {},
            "ETH/USDC:USDC": {},
            "SOL/USDC:USDC": {},
            "BTC/USDC": {},  # spot — no colon, should be skipped
        }
        mock_cls = MagicMock(return_value=mock_exchange)
        mock_ccxt.hyperliquid = mock_cls

        coins = data.get_testnet_coins("hyperliquid")

        assert coins == {"BTC", "ETH", "SOL"}
        mock_exchange.set_sandbox_mode.assert_called_once_with(True)
        mock_exchange.load_markets.assert_called_once()

    @patch("scanner.data.ccxt")
    @patch("scanner.data.get_exchange_meta", return_value={"ccxt_class": "hyperliquid"})
    def test_excludes_non_alpha_coins(self, mock_meta, mock_ccxt):
        mock_exchange = MagicMock()
        mock_exchange.markets = {
            "BTC/USDC:USDC": {},
            "K-PEPE/USDC:USDC": {},  # hyphen — non-alpha
            "1000SHIB/USDC:USDC": {},  # starts with digit — non-alpha
        }
        mock_cls = MagicMock(return_value=mock_exchange)
        mock_ccxt.hyperliquid = mock_cls

        coins = data.get_testnet_coins("hyperliquid")

        assert coins == {"BTC"}

    @patch("scanner.data.ccxt")
    @patch("scanner.data.get_exchange_meta", return_value={"ccxt_class": "hyperliquid"})
    def test_not_cached_in_singleton(self, mock_meta, mock_ccxt):
        """Testnet exchange should NOT pollute the module-level singleton."""
        mock_exchange = MagicMock()
        mock_exchange.markets = {"BTC/USDC:USDC": {}}
        mock_cls = MagicMock(return_value=mock_exchange)
        mock_ccxt.hyperliquid = mock_cls

        data.get_testnet_coins("hyperliquid")

        assert "hyperliquid_testnet" not in data._exchanges
        # Module singleton should be unaffected
        assert mock_exchange not in data._exchanges.values()
