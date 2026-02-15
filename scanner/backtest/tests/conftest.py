"""Shared fixtures for backtest tests."""

import sqlite3

import pandas as pd
import pytest

from scanner.storage import get_connection


@pytest.fixture
def db_conn(tmp_path):
    """In-memory SQLite connection with all tables created."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    yield conn
    conn.close()


def make_candles(
    n: int = 100,
    start: str = "2026-01-01",
    freq: str = "1h",
    base: float = 100.0,
    trend: float = 0.0,
    spread: float = 2.0,
) -> pd.DataFrame:
    """Build synthetic OHLCV DataFrame.

    Args:
        n: Number of candles.
        start: Start datetime.
        freq: Candle frequency.
        base: Starting price.
        trend: Per-bar price drift.
        spread: High-Low range per bar.
    """
    dates = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    data = []
    price = base
    for i in range(n):
        o = price
        h = price + spread / 2
        l = price - spread / 2
        c = price + trend
        data.append({"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1000.0})
        price = c
    df = pd.DataFrame(data, index=dates)
    return df


def make_candles_from_hlc(
    hlc_data: list[tuple[float, float, float]],
    start: str = "2026-01-01T01:00:00+00:00",
    freq: str = "1h",
) -> pd.DataFrame:
    """Build candles from (high, low, close) tuples. Open = previous close or first close."""
    dates = pd.date_range(start, periods=len(hlc_data), freq=freq, tz="UTC")
    data = []
    prev_close = hlc_data[0][2]
    for h, l, c in hlc_data:
        data.append({"Open": prev_close, "High": h, "Low": l, "Close": c, "Volume": 1000.0})
        prev_close = c
    return pd.DataFrame(data, index=dates)
