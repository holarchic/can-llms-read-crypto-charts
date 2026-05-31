"""Tests for exchange data fetching — thread safety."""

import threading
from unittest.mock import MagicMock, patch

import ccxt
import pandas as pd
import pytest

from scanner import data


class TestGetExchangeThreadSafe:
    def setup_method(self):
        """Reset module-level state between tests."""
        data._exchanges.clear()
        data._exchange_locks.clear()
        data._exchange_failures.clear()

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


class TestFetchCandlesValidation:
    """`fetch_candles` validates the ccxt response at the untrusted ingestion boundary —
    an empty frame or a non-finite / non-positive O/H/L/C price raises (decision #34)."""

    def setup_method(self):
        data._exchanges.clear()
        data._exchange_locks.clear()
        data._exchange_failures.clear()

    def _fetch_with(self, rows):
        """Run fetch_candles against a mocked exchange returning ``rows``."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = rows
        data._exchanges["hyperliquid"] = mock_exchange
        data._exchange_locks["hyperliquid"] = threading.Lock()
        with patch("scanner.data.get_exchange_meta", return_value={
            "ccxt_class": "hyperliquid",
            "symbol_template": "{coin}/USDC:USDC",
        }):
            with patch("scanner.data.time.sleep"):
                return data.fetch_candles("BTC", "4h", 2)

    def test_fetch_candles_empty_raises(self):
        with pytest.raises(ValueError, match="empty OHLCV"):
            self._fetch_with([])

    def test_fetch_candles_nan_ohlc_raises(self):
        rows = [
            [1000000, 1.0, 2.0, 0.5, 1.5, 100.0],
            [1000001, 1.5, float("nan"), 1.0, 2.0, 200.0],  # NaN High
        ]
        with pytest.raises(ValueError, match="non-finite or non-positive"):
            self._fetch_with(rows)

    def test_fetch_candles_nonpositive_price_raises(self):
        """A non-positive O/H/L (not just Close) is malformed — Low <= 0 here."""
        rows = [
            [1000000, 1.0, 2.0, 0.5, 1.5, 100.0],
            [1000001, 1.5, 2.5, 0.0, 2.0, 200.0],  # Low == 0
        ]
        with pytest.raises(ValueError, match="non-finite or non-positive"):
            self._fetch_with(rows)

    def test_fetch_candles_valid_passes(self):
        rows = [
            [1000000, 1.0, 2.0, 0.5, 1.5, 100.0],
            [1000001, 1.5, 2.5, 1.0, 2.0, 200.0],
        ]
        df = self._fetch_with(rows)
        assert len(df) == 2
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


class TestFetchCandlesAsOf:
    """`fetch_candles(as_of=T)` clamps to a historical scan time (T-020.15, SPEC #53):
    keep bars with open-time <= T, verify the bar *containing* T is present (interval-aware
    freshness — NOT a median-width heuristic), return the last `count`. A stale tail raises."""

    def setup_method(self):
        data._exchanges.clear()
        data._exchange_locks.clear()
        data._exchange_failures.clear()

    @staticmethod
    def _rows(opens):
        """OHLCV rows (valid prices) at the given open-time Timestamps."""
        return [[int(o.timestamp() * 1000), 1.0, 2.0, 0.5, 1.5, 100.0] for o in opens]

    def _run(self, rows, as_of, interval, count):
        """Run fetch_candles against a mocked exchange returning ``rows`` for every call.
        Stores the mock on ``self.mock`` so call args (since vs most-recent) are assertable."""
        self.mock = MagicMock()
        self.mock.fetch_ohlcv.return_value = rows
        data._exchanges["hyperliquid"] = self.mock
        data._exchange_locks["hyperliquid"] = threading.Lock()
        with patch("scanner.data.get_exchange_meta", return_value={
            "ccxt_class": "hyperliquid",
            "symbol_template": "{coin}/USDC:USDC",
        }):
            with patch("scanner.data.time.sleep"):
                return data.fetch_candles("BTC", interval, count, as_of=as_of)

    def test_asof_clamps_and_keeps_bar_containing_T(self):
        """Bars straddling+overshooting T: returns the last `count` with open <= T, incl. the
        forming-at-T bar; the as-of path passes a `since` (historical window, not most-recent)."""
        as_of = pd.Timestamp("2026-05-20 12:00:00", tz="UTC")
        opens = pd.date_range("2026-05-20 09:00", "2026-05-20 14:00", freq="1h", tz="UTC")
        df = self._run(self._rows(opens), as_of, "1h", 3)
        assert len(df) == 3
        assert df.index.max() == as_of  # 12:00 bar kept (forming-at-T; caller drops it)
        assert data._advance_one_bar(df.index[-1], "1h") > as_of  # freshness holds
        assert self.mock.fetch_ohlcv.call_args.kwargs.get("since") is not None  # as-of path

    def test_asof_stale_tail_raises(self):
        """A tail ending well before the bar containing T fails freshness → widen-retry → raise
        (never a silent wrong-bar drop). fetch_ohlcv is called twice (initial + one widen)."""
        as_of = pd.Timestamp("2026-05-20 12:00:00", tz="UTC")
        opens = pd.date_range("2026-05-20 04:00", "2026-05-20 08:00", freq="1h", tz="UTC")
        with pytest.raises(ValueError, match="stale"):
            self._run(self._rows(opens), as_of, "1h", 3)
        assert self.mock.fetch_ohlcv.call_count == 2

    def test_asof_1M_month_boundary_stale_raises(self):
        """The month-boundary hole a median-width check misses: early-March as_of with the latest
        bar Feb-1 (March monthly bar MISSING). `_advance_one_bar(Feb-1)=Mar-1 <= Mar-3` ⇒ stale
        (a `T - last < median width` test would have wrongly passed at ~30d)."""
        as_of = pd.Timestamp("2026-03-03 00:00:00", tz="UTC")
        opens = pd.date_range("2025-01-01", "2026-02-01", freq="MS", tz="UTC")  # ... Jan, Feb (no Mar)
        with pytest.raises(ValueError, match="stale"):
            self._run(self._rows(opens), as_of, "1M", 12)

    def test_asof_1M_month_boundary_fresh_passes(self):
        """Same early-March as_of but the March bar IS present (open Mar-1) ⇒ fresh; the bar
        containing T is kept and `_advance_one_bar` steps a full calendar month to Apr-1."""
        as_of = pd.Timestamp("2026-03-03 00:00:00", tz="UTC")
        opens = pd.date_range("2025-01-01", "2026-03-01", freq="MS", tz="UTC")  # ... Feb, Mar
        df = self._run(self._rows(opens), as_of, "1M", 12)
        assert df.index.max() == pd.Timestamp("2026-03-01", tz="UTC")
        assert data._advance_one_bar(df.index[-1], "1M") == pd.Timestamp("2026-04-01", tz="UTC")

    def test_asof_none_is_most_recent_call(self):
        """as_of=None ⇒ the live path: fetch_ohlcv(limit=count) with NO `since` (byte-identical)."""
        rows = [[1000000, 1.0, 2.0, 0.5, 1.5, 100.0], [1000001, 1.5, 2.5, 1.0, 2.0, 200.0]]
        df = self._run(rows, None, "4h", 2)
        assert len(df) == 2
        assert "since" not in self.mock.fetch_ohlcv.call_args.kwargs
        assert self.mock.fetch_ohlcv.call_args.kwargs.get("limit") == 2


class TestIsTransientCcxt:
    """_is_transient_ccxt decides whether an error warrants retry.

    Geo-block signals must classify as non-transient — retrying them from a
    blocked host just burns the scan timeout.
    """

    def test_rate_limit_class_is_transient(self):
        assert data._is_transient_ccxt(ccxt.RateLimitExceeded("429"))

    def test_network_error_class_is_transient(self):
        assert data._is_transient_ccxt(ccxt.NetworkError("conn reset"))

    def test_exchange_not_available_is_transient(self):
        assert data._is_transient_ccxt(ccxt.ExchangeNotAvailable("upstream down"))

    def test_hl_cumulative_string_is_transient(self):
        assert data._is_transient_ccxt(Exception("Too many cumulative requests"))

    def test_503_504_in_message_is_transient(self):
        assert data._is_transient_ccxt(Exception("got 503 from gateway"))
        assert data._is_transient_ccxt(Exception("504 timeout"))

    def test_bybit_cloudfront_403_is_not_transient(self):
        """bybit CloudFront 403 — permanent geo-block from GCE us-central1."""
        exc = ccxt.ExchangeNotAvailable(
            "bybit GET https://api.bybit.com/v5/market/instruments-info 403 "
            "Forbidden <html>Generated by cloudfront (CloudFront)</html>"
        )
        assert not data._is_transient_ccxt(exc)

    def test_binance_451_restricted_location_is_not_transient(self):
        """binance 451 — permanent compliance block from blocked region."""
        exc = ccxt.ExchangeNotAvailable(
            "binanceusdm GET https://fapi.binance.com/fapi/v1/exchangeInfo 451 "
            '{"msg": "Service unavailable from a restricted location..."}'
        )
        assert not data._is_transient_ccxt(exc)

    def test_binance_eligibility_marker_is_not_transient(self):
        exc = Exception("binance 451 see Eligibility section of terms")
        assert not data._is_transient_ccxt(exc)

    def test_unknown_error_is_not_transient(self):
        assert not data._is_transient_ccxt(Exception("some unrelated bug"))


class TestGetExchangeFailureCache:
    """Non-transient load_markets() failures must be cached per scan so the
    subsequent 80 coin calls fail fast instead of each triggering a retry."""

    def setup_method(self):
        data._exchanges.clear()
        data._exchange_locks.clear()
        data._exchange_failures.clear()

    def test_permanent_failure_is_cached_and_reraised(self):
        """Geo-block raises once, then is re-raised from cache without retry."""
        mock_exchange = MagicMock()
        mock_exchange.load_markets.side_effect = ccxt.ExchangeNotAvailable(
            "bybit GET / 403 CloudFront"
        )
        mock_cls = MagicMock(return_value=mock_exchange)

        with patch("scanner.data.ccxt") as mock_ccxt:
            mock_ccxt.bybit = mock_cls
            mock_ccxt.ExchangeNotAvailable = ccxt.ExchangeNotAvailable
            with patch("scanner.data.get_exchange_meta", return_value={
                "ccxt_class": "bybit",
            }):
                with pytest.raises(ccxt.ExchangeNotAvailable):
                    data.get_exchange("bybit")
                # Second call — must use cache, not re-instantiate or re-load.
                with pytest.raises(ccxt.ExchangeNotAvailable):
                    data.get_exchange("bybit")
                # Third call for good measure.
                with pytest.raises(ccxt.ExchangeNotAvailable):
                    data.get_exchange("bybit")

        # Exchange class instantiated once, load_markets called once — the
        # remaining two calls short-circuited on the failure cache.
        assert mock_cls.call_count == 1
        assert mock_exchange.load_markets.call_count == 1
        assert "bybit" in data._exchange_failures

    def test_transient_exhausted_is_not_cached(self):
        """Transient retries exhausted → don't cache; next caller may retry.

        A brief HL blip shouldn't permanently kill the scan — if the blip
        clears, later coins succeed.
        """
        mock_exchange = MagicMock()
        mock_exchange.load_markets.side_effect = ccxt.NetworkError("conn reset")
        mock_cls = MagicMock(return_value=mock_exchange)

        with patch("scanner.data.ccxt") as mock_ccxt:
            mock_ccxt.hyperliquid = mock_cls
            mock_ccxt.NetworkError = ccxt.NetworkError
            with patch("scanner.data.get_exchange_meta", return_value={
                "ccxt_class": "hyperliquid",
            }):
                with patch("scanner.data.time.sleep"):
                    with pytest.raises(ccxt.NetworkError):
                        data.get_exchange("hyperliquid")

        # 4 attempts before raising — matches original retry policy.
        assert mock_exchange.load_markets.call_count == 4
        # Not cached — caller can retry on the next coin.
        assert "hyperliquid" not in data._exchange_failures


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
