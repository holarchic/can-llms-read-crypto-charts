"""Shared fixtures for scanner tests."""

import numpy as np
import pandas as pd
import pytest


def _make_df(prices: list[tuple], start: str = "2026-01-01") -> pd.DataFrame:
    """Build a DataFrame from (open, high, low, close) tuples."""
    dates = pd.date_range(start, periods=len(prices), freq="4h", tz="UTC")
    data = {
        "Open": [p[0] for p in prices],
        "High": [p[1] for p in prices],
        "Low": [p[2] for p in prices],
        "Close": [p[3] for p in prices],
        "Volume": [1000.0] * len(prices),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def uptrend_df():
    """Clear uptrend: higher highs and higher lows with pronounced swings."""
    np.random.seed(42)
    prices = []
    base = 100.0
    for i in range(80):
        # Zigzag pattern with upward bias: swing every ~10 bars
        cycle = np.sin(i * 0.6) * 5  # ±5 oscillation
        trend = i * 0.5  # upward slope
        mid = base + trend + cycle
        o = mid + np.random.uniform(-0.5, 0.5)
        h = mid + np.random.uniform(2, 4)
        l = mid - np.random.uniform(2, 4)
        c = mid + np.random.uniform(-0.5, 0.5)
        prices.append((o, h, l, c))
    return _make_df(prices)


@pytest.fixture
def downtrend_df():
    """Clear downtrend: lower highs and lower lows with pronounced swings."""
    np.random.seed(42)
    prices = []
    base = 200.0
    for i in range(80):
        cycle = np.sin(i * 0.6) * 5
        trend = -i * 0.5
        mid = base + trend + cycle
        o = mid + np.random.uniform(-0.5, 0.5)
        h = mid + np.random.uniform(2, 4)
        l = mid - np.random.uniform(2, 4)
        c = mid + np.random.uniform(-0.5, 0.5)
        prices.append((o, h, l, c))
    return _make_df(prices)


@pytest.fixture
def ranging_df():
    """Ranging/sideways market between 95 and 105."""
    np.random.seed(42)
    prices = []
    for i in range(80):
        mid = 100 + 5 * np.sin(i * 0.3)
        o = mid + np.random.uniform(-1, 1)
        h = o + np.random.uniform(0.5, 2.0)
        l = o - np.random.uniform(0.5, 2.0)
        c = mid + np.random.uniform(-1, 1)
        prices.append((o, h, l, c))
    return _make_df(prices)


@pytest.fixture
def spike_df():
    """Data with a clear SFP/spike — wick beyond a level then close back inside."""
    prices = []
    # Build up to a range around 100
    for i in range(70):
        base = 100 + np.sin(i * 0.2) * 3
        o = base
        h = base + 1.5
        l = base - 1.5
        c = base + 0.5
        prices.append((o, h, l, c))

    # Spike candle: wick way above 105 (prior high area) but close back at 102
    prices.append((102, 110, 101, 102))  # Big upper wick
    # Follow-up candles
    for i in range(9):
        prices.append((101 + i * 0.1, 102, 100, 101))

    return _make_df(prices)
