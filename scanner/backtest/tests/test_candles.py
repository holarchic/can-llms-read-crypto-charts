"""Tests for candle cache layer."""

import pandas as pd
import pytest

from scanner.backtest.candles import cache_candles, get_cached_candles
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
