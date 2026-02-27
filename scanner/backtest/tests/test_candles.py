"""Tests for candle cache layer."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from scanner.backtest.candles import cache_candles, get_cached_candles, fetch_and_cache
from .conftest import make_candles


class TestCacheRoundtrip:
    """Cache write then read returns same data."""

    def test_basic_roundtrip(self, db_conn):
        df = make_candles(n=10, start="2026-01-01", freq="1h")
        cached = cache_candles("BTC", "1h", df, db_conn)
        assert cached == 10

        result = get_cached_candles(
            "BTC", "1h",
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T10:00:00+00:00",
            db_conn,
        )
        assert result is not None
        assert len(result) == 10
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_empty_cache_returns_none(self, db_conn):
        result = get_cached_candles(
            "ETH", "1h",
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T10:00:00+00:00",
            db_conn,
        )
        assert result is None

    def test_dedup_on_insert(self, db_conn):
        df = make_candles(n=5, start="2026-01-01", freq="1h")
        cache_candles("BTC", "1h", df, db_conn)
        cache_candles("BTC", "1h", df, db_conn)  # duplicate insert

        result = get_cached_candles(
            "BTC", "1h",
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T05:00:00+00:00",
            db_conn,
        )
        assert len(result) == 5

    def test_range_filter(self, db_conn):
        df = make_candles(n=24, start="2026-01-01", freq="1h")
        cache_candles("BTC", "1h", df, db_conn)

        # Request only first 6 hours
        result = get_cached_candles(
            "BTC", "1h",
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T05:00:00+00:00",
            db_conn,
        )
        assert len(result) == 6

    def test_different_coins_isolated(self, db_conn):
        df = make_candles(n=5, start="2026-01-01", freq="1h")
        cache_candles("BTC", "1h", df, db_conn)
        cache_candles("ETH", "1h", df, db_conn)

        btc = get_cached_candles("BTC", "1h", "2026-01-01", "2026-01-02", db_conn)
        eth = get_cached_candles("ETH", "1h", "2026-01-01", "2026-01-02", db_conn)
        assert len(btc) == 5
        assert len(eth) == 5

    def test_different_intervals_isolated(self, db_conn):
        df_1h = make_candles(n=5, start="2026-01-01", freq="1h")
        df_4h = make_candles(n=5, start="2026-01-01", freq="4h")
        cache_candles("BTC", "1h", df_1h, db_conn)
        cache_candles("BTC", "4h", df_4h, db_conn)

        h1 = get_cached_candles("BTC", "1h", "2026-01-01", "2026-01-02", db_conn)
        h4 = get_cached_candles("BTC", "4h", "2026-01-01", "2026-01-02", db_conn)
        assert len(h1) == 5
        assert len(h4) == 5


def _ohlcv_row(ts_ms, price=100.0):
    """Build a single ccxt-style OHLCV row."""
    return [ts_ms, price, price + 1, price - 1, price, 1000.0]


class TestFetchAndCachePreGap:
    """fetch_and_cache backfills pre-cache gaps."""

    @pytest.fixture(autouse=True)
    def _patch_exchange(self, monkeypatch):
        self.mock_exchange = MagicMock()
        self.mock_exchange.fetch_ohlcv.return_value = []
        monkeypatch.setattr(
            "scanner.data.get_exchange",
            lambda _: self.mock_exchange,
        )

    def test_backfills_pre_cache_gap(self, db_conn):
        """Cache covers days 5-10, request days 1-10 → fetches 1-4."""
        # Seed cache with hours 4-9 (Jan 1 04:00 to 09:00)
        cached_df = make_candles(n=6, start="2026-01-01T04:00:00+00:00", freq="1h")
        cache_candles("BTC", "1h", cached_df, db_conn)

        # Mock returns pre-gap candles (hours 0-3)
        pre_gap_candles = [
            _ohlcv_row(int(pd.Timestamp(f"2026-01-01T{h:02d}:00:00+00:00").timestamp() * 1000))
            for h in range(4)
        ]
        # First call returns pre-gap, second call (post-gap) returns nothing
        self.mock_exchange.fetch_ohlcv.side_effect = [pre_gap_candles, []]

        result = fetch_and_cache(
            "BTC", "1h",
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T09:00:00+00:00",
            db_conn,
        )

        assert len(result) == 10  # 4 pre-gap + 6 cached
        # Should have fetched the pre-gap
        assert self.mock_exchange.fetch_ohlcv.call_count >= 1
        first_call_since = self.mock_exchange.fetch_ohlcv.call_args_list[0][1].get(
            "since", self.mock_exchange.fetch_ohlcv.call_args_list[0][0][2]
            if len(self.mock_exchange.fetch_ohlcv.call_args_list[0][0]) > 2 else None
        )

    def test_no_pre_gap_when_cache_covers_start(self, db_conn):
        """Cache covers days 1-10, request days 3-10 → no pre-gap fetch."""
        # Seed cache with hours 0-9 (covers full range)
        cached_df = make_candles(n=10, start="2026-01-01T00:00:00+00:00", freq="1h")
        cache_candles("BTC", "1h", cached_df, db_conn)

        result = fetch_and_cache(
            "BTC", "1h",
            "2026-01-01T02:00:00+00:00",
            "2026-01-01T09:00:00+00:00",
            db_conn,
        )

        # Cache fully covers [02:00, 09:00] → no fetch needed
        assert self.mock_exchange.fetch_ohlcv.call_count == 0
        assert len(result) == 8  # cached returns requested range (02:00–09:00)
