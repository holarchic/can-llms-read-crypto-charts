"""Tests for scanner/compression_exit.py (T-019 Chunk 2).

Synthetic hand-crafted bands + closes so breakout/defense (held ≥2 bars + band
re-fanned) and the ``armed`` precondition are pinned deterministically.
"""

import pandas as pd

from scanner.compression_exit import evaluate_compression_exit
from scanner.ma_indicators import EMABand
from scanner.trend_state import TrendDirection


def _df(closes: list[float]) -> pd.DataFrame:
    c = [float(x) for x in closes]
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": [1.0] * len(c)})


def _band(fast: list[float], mid: list[float], slow: list[float]) -> EMABand:
    return EMABand(pd.Series(fast, dtype=float), pd.Series(mid, dtype=float), pd.Series(slow, dtype=float))


def test_trip_up_breakout_held_and_fanned():
    df = _df([5, 12, 15])  # close held above the band for two bars
    band = _band([5, 9, 11], [5, 8, 10], [5, 7, 9])  # latest 11>10>9 fan up; high_now=11
    ce = evaluate_compression_exit(df, band, armed=True)
    assert ce.tripped is True
    assert ce.direction is TrendDirection.UP


def test_trip_down_breakout_held_and_fanned():
    df = _df([15, 8, 5])
    band = _band([15, 11, 9], [15, 12, 10], [15, 13, 11])  # latest 9<10<11 fan down; low_now=9
    ce = evaluate_compression_exit(df, band, armed=True)
    assert ce.tripped is True
    assert ce.direction is TrendDirection.DOWN


def test_no_trip_when_not_armed():
    # Same setup as test_trip_up, but no compression context → must not trip.
    df = _df([5, 12, 15])
    band = _band([5, 9, 11], [5, 8, 10], [5, 7, 9])
    assert evaluate_compression_exit(df, band, armed=False).tripped is False


def test_no_trip_single_bar_poke():
    df = _df([5, 5, 15])  # only the final bar pokes out — not defended
    band = _band([5, 9, 11], [5, 8, 10], [5, 7, 9])
    assert evaluate_compression_exit(df, band, armed=True).tripped is False


def test_no_trip_price_inside_band():
    df = _df([5, 8, 10])  # 10 <= high_now (11), no breakout
    band = _band([5, 9, 11], [5, 8, 10], [5, 7, 9])
    assert evaluate_compression_exit(df, band, armed=True).tripped is False


def test_no_trip_breakout_without_fan():
    df = _df([5, 12, 15])  # broke out + held, but the band has NOT re-fanned up
    band = _band([5, 9, 9], [5, 8, 11], [5, 7, 10])  # latest 9/11/10 — not f>m>s
    assert evaluate_compression_exit(df, band, armed=True).tripped is False


def test_no_trip_insufficient_data():
    assert evaluate_compression_exit(_df([5]), _band([5], [5], [5]), armed=True).tripped is False


def test_no_trip_short_or_misaligned_band():
    # A component with <2 bars → cannot read a 2-bar breakout → not tripped, not a crash.
    df = _df([5, 12, 15])
    band = _band([5, 9, 11], [10.0], [5, 7, 9])  # mid has only 1 bar
    assert evaluate_compression_exit(df, band, armed=True).tripped is False


def test_no_trip_nan_band():
    df = _df([5, 12, 15])
    band = _band([5, 9, float("nan")], [5, 8, 10], [5, 7, 9])
    assert evaluate_compression_exit(df, band, armed=True).tripped is False
