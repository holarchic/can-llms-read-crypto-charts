"""Tests for scanner/trend_state.py (T-019 Chunk 2).

Direction/position use realistic arange fixtures (real bands from Chunk 1); the
compression predicate uses synthetic hand-crafted bands so contracting-vs-fanning
and slope-disagreement are pinned deterministically rather than reverse-engineered
out of EMA dynamics.
"""

import math

import pandas as pd

from scanner.ma_indicators import EMABand, LongerMASet, compute_ema_band, compute_longer_set
from scanner.trend_state import BandPosition, TrendDirection, derive_trend_state


def _df(closes: list[float]) -> pd.DataFrame:
    c = [float(x) for x in closes]
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": [1.0] * len(c)})


def _band(fast: list[float], mid: list[float], slow: list[float]) -> EMABand:
    return EMABand(pd.Series(fast, dtype=float), pd.Series(mid, dtype=float), pd.Series(slow, dtype=float))


def _nan_longer(n: int) -> LongerMASet:
    s = pd.Series([float("nan")] * n)
    return LongerMASet(s, s, s)


def test_uptrend_state():
    df = _df(list(range(1, 350)))
    c = df["Close"]
    ts = derive_trend_state(df, compute_ema_band(c), compute_longer_set(c))
    assert ts.local_trend_direction is TrendDirection.UP
    assert ts.price_vs_ribbon is BandPosition.ABOVE
    assert ts.longer_trend_direction is TrendDirection.UP
    assert ts.price_vs_longer is BandPosition.ABOVE
    assert ts.ribbon_compressed is False
    assert ts.gap_to_next_ma > 0 and not math.isnan(ts.gap_to_next_ma)


def test_downtrend_state():
    df = _df(list(range(350, 1, -1)))
    c = df["Close"]
    ts = derive_trend_state(df, compute_ema_band(c), compute_longer_set(c))
    assert ts.local_trend_direction is TrendDirection.DOWN
    assert ts.price_vs_ribbon is BandPosition.BELOW
    assert ts.longer_trend_direction is TrendDirection.DOWN
    assert ts.price_vs_longer is BandPosition.BELOW
    assert ts.ribbon_compressed is False


def test_insufficient_data_is_neutral_not_a_crash():
    df = _df([1, 2, 3])
    c = df["Close"]
    ts = derive_trend_state(df, compute_ema_band(c), compute_longer_set(c))
    assert ts.local_trend_direction is TrendDirection.FLAT
    assert ts.price_vs_ribbon is BandPosition.INSIDE
    assert ts.ribbon_compressed is False
    assert ts.longer_trend_direction is TrendDirection.FLAT
    assert ts.price_vs_longer is BandPosition.INSIDE
    assert math.isnan(ts.gap_to_next_ma)
    assert math.isnan(ts.compression_tightness)


def test_empty_input_is_neutral_not_a_crash():
    # The natural empty composition must honor the never-crash contract.
    df = _df([])
    c = df["Close"]
    ts = derive_trend_state(df, compute_ema_band(c), compute_longer_set(c))
    assert ts.local_trend_direction is TrendDirection.FLAT
    assert ts.price_vs_ribbon is BandPosition.INSIDE
    assert ts.longer_trend_direction is TrendDirection.FLAT
    assert ts.price_vs_longer is BandPosition.INSIDE
    assert ts.ribbon_compressed is False
    assert math.isnan(ts.gap_to_next_ma)
    assert math.isnan(ts.compression_tightness)


def test_misaligned_or_short_band_degrades_to_neutral():
    # Short components (early-return branch of _is_compressed).
    df = _df([1, 2, 3])
    band = EMABand(pd.Series([1.0, 2.0, 3.0]), pd.Series([], dtype=float), pd.Series([1.0]))
    longer = LongerMASet(pd.Series([], dtype=float), pd.Series([1.0]), pd.Series([], dtype=float))
    ts = derive_trend_state(df, band, longer)
    assert ts.local_trend_direction is TrendDirection.FLAT
    assert ts.price_vs_ribbon is BandPosition.INSIDE
    assert ts.ribbon_compressed is False
    assert math.isnan(ts.gap_to_next_ma)


def test_misaligned_band_longer_than_window_degrades_to_neutral():
    # fast LONGER than contraction_window (5) with an empty mid — the deep
    # _is_compressed read that round-2's len(fast)<=window guard would have crashed on.
    df = _df([1, 2, 3, 4, 5, 6, 7])
    seven = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    band = EMABand(seven, pd.Series([], dtype=float), seven)
    longer = LongerMASet(pd.Series([], dtype=float), pd.Series([], dtype=float), pd.Series([], dtype=float))
    ts = derive_trend_state(df, band, longer)
    assert ts.ribbon_compressed is False  # would IndexError on mid.iloc[-1] before the _at fix
    assert ts.local_trend_direction is TrendDirection.FLAT


def test_compression_predicate_true_when_contracting_and_slopes_disagree():
    # fast flattening/down, mid+slow rising → slope-disagreement; band narrowing.
    band = _band(
        [12, 12, 11.8, 11.6, 11.4, 11.2, 11.0],   # slope over 3 = -0.6
        [9, 9.3, 9.6, 9.9, 10.2, 10.5, 10.8],     # slope = +0.9
        [6, 6.6, 7.2, 7.8, 8.4, 9.0, 9.6],        # slope = +1.8
    )
    ts = derive_trend_state(_df([1, 2, 3, 4, 5, 6, 7]), band, _nan_longer(7))
    assert ts.ribbon_compressed is True


def test_compression_predicate_false_when_fanning():
    # Three parallel rising EMAs → all-positive slopes, constant width → not compressed.
    band = _band(
        [1, 2, 3, 4, 5, 6, 7], [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], [0, 1, 2, 3, 4, 5, 6],
    )
    ts = derive_trend_state(_df([1, 2, 3, 4, 5, 6, 7]), band, _nan_longer(7))
    assert ts.ribbon_compressed is False


def test_mixed_stack_is_flat():
    band = _band(
        [5, 6, 7, 8, 9, 10, 10], [5, 6, 7, 8, 9, 11, 12], [5, 6, 7, 8, 9, 9, 8],  # latest 10/12/8 — crossed
    )
    ts = derive_trend_state(_df([1, 2, 3, 4, 5, 6, 7]), band, _nan_longer(7))
    assert ts.local_trend_direction is TrendDirection.FLAT
