"""Tests for scanner/ma_indicators.py (T-019 Chunk 1).

Oracles are hand-computed from the documented formulas (independent of the pandas
implementation) or are convention-independent properties (all-gain RSI = 100,
value-area brackets the POC). Per "Green ≠ verified" — no oracle is produced by
the code under test. The EMA oracle pins the SMA-seed convention specifically
(it would fail under a first-observation seed), so the seed choice is tested, not
just asserted in a docstring.
"""

import numpy as np
import pandas as pd
import pytest

from scanner.ma_indicators import (
    compute_ema,
    compute_ema_band,
    compute_longer_set,
    compute_relative_volume,
    compute_rsi,
    compute_sma,
    compute_stoch_rsi,
    compute_volume_profile,
)


def test_compute_ema_sma_seeded():
    # SMA-seeded (ta-lib) EMA, period 3, alpha=0.5:
    # idx0,1 = NaN (warmup); idx2 = SMA(10,11,12) = 11; idx3 = 0.5*13+0.5*11 = 12;
    # idx4 = 0.5*14+0.5*12 = 13.  (A first-value seed would give 11.25/12.125 here.)
    s = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    ema = compute_ema(s, 3)
    assert ema.iloc[:2].isna().all()
    assert np.allclose(ema.iloc[2:].to_numpy(), [11.0, 12.0, 13.0])


def test_compute_ema_insufficient_data_all_nan():
    assert compute_ema(pd.Series([1.0, 2.0]), 3).isna().all()


def test_compute_sma_warmup_then_mean():
    s = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    sma = compute_sma(s, 3)
    assert sma.iloc[:2].isna().all()
    assert np.allclose(sma.iloc[2:].to_numpy(), [11.0, 12.0, 13.0])


def test_compute_ema_band_periods_and_math():
    s = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    band = compute_ema_band(s, fast=2, mid=3, slow=4)
    # SMA-seeded fast EMA, period 2, alpha=2/3: NaN, 10.5, 11.5, 12.5, 13.5
    assert band.fast.iloc[:1].isna().all()
    assert np.allclose(band.fast.iloc[1:].to_numpy(), [10.5, 11.5, 12.5, 13.5])
    # Default-period wiring: the slow line must be the 36 EMA (catches a transposed default).
    long_s = pd.Series(np.arange(1.0, 200.0))
    assert compute_ema_band(long_s).slow.equals(compute_ema(long_s, 36))


def test_compute_longer_set_fields_and_values():
    s = pd.Series(np.arange(1.0, 401.0))  # 1..400
    ls = compute_longer_set(s)
    assert ls._fields == ("sma_100", "ema_200", "sma_300")
    assert ls.sma_100.iloc[-1] == pytest.approx(350.5)  # mean(301..400)
    assert ls.sma_300.iloc[-1] == pytest.approx(250.5)  # mean(101..400)
    # Insufficient data → all NaN, not an error.
    assert compute_longer_set(pd.Series([1.0, 2.0, 3.0])).sma_100.isna().all()


def test_compute_rsi_extremes():
    # Strictly rising → zero losses → RSI 100; strictly falling → RSI 0.
    assert compute_rsi(pd.Series(np.arange(1.0, 51.0)), 14).iloc[-1] == pytest.approx(100.0)
    assert compute_rsi(pd.Series(np.arange(50.0, 0.0, -1.0)), 14).iloc[-1] == pytest.approx(0.0)


def test_compute_rsi_hand_computed_wilder():
    # Standard Wilder, SMA-seeded over the first `length` REAL deltas → first RSI
    # at index `length`. close=[10,11,10,11,10,11], length=2:
    # deltas (from idx1): +1,-1,+1,-1,+1 → RSI = [nan, nan, 50, 75, 37.5, 68.75]
    s = pd.Series([10.0, 11.0, 10.0, 11.0, 10.0, 11.0])
    rsi = compute_rsi(s, 2)
    assert rsi.iloc[:2].isna().all()  # first valid at index length=2
    assert rsi.iloc[2] == pytest.approx(50.0)
    assert rsi.iloc[-1] == pytest.approx(68.75)


def test_compute_rsi_propagates_interior_nan():
    # A malformed interior close surfaces as NaN downstream, not a fake number.
    s = pd.Series([10.0, 11.0, 12.0, float("nan"), 14.0, 15.0, 16.0])
    rsi = compute_rsi(s, 2)
    assert not np.isnan(rsi.iloc[2])  # clean values before the gap
    assert rsi.iloc[3:].isna().all()  # the gap propagates — no plausible-wrong value


def test_compute_stoch_rsi_bounds_and_shape():
    # Oscillating series so RSI varies and the stochastic range is well-defined.
    wave = ([float(i) for i in range(1, 21)] + [float(i) for i in range(20, 0, -1)]) * 3
    sr = compute_stoch_rsi(pd.Series(wave), length=14, k=3, d=3)
    assert len(sr.k) == len(wave) and len(sr.d) == len(wave)
    for line in (sr.k.dropna(), sr.d.dropna()):
        assert len(line) > 0
        assert ((line >= -1e-9) & (line <= 100 + 1e-9)).all()


def test_compute_volume_profile_poc_at_dominant_price():
    # 50 heavy bars at price 100, light bars at 90/110 → POC ≈ 100, VA brackets it.
    prices = [100.0] * 50 + [90.0] * 5 + [110.0] * 5
    vols = [100.0] * 50 + [10.0] * 5 + [10.0] * 5
    df = pd.DataFrame({"Open": prices, "High": prices, "Low": prices, "Close": prices, "Volume": vols})
    vp = compute_volume_profile(df)
    assert vp.poc == pytest.approx(100.0, abs=1.5)  # within one bin of the dominant price
    assert vp.val <= vp.poc <= vp.vah


def test_compute_volume_profile_drops_nan_rows():
    # A NaN-price bar with huge volume must be dropped, not misbinned into an edge.
    prices = [100.0] * 50 + [float("nan")]
    vols = [100.0] * 50 + [10_000.0]
    df = pd.DataFrame({"Open": [100.0] * 51, "High": prices, "Low": prices,
                       "Close": prices, "Volume": vols})
    vp = compute_volume_profile(df)
    assert vp.poc == pytest.approx(100.0, abs=0.01)  # NaN bar excluded → single clean price


def test_compute_volume_profile_degenerate_single_price():
    df = pd.DataFrame({"Open": [5.0] * 10, "High": [5.0] * 10, "Low": [5.0] * 10,
                       "Close": [5.0] * 10, "Volume": [1.0] * 10})
    vp = compute_volume_profile(df)
    assert (vp.vah, vp.val, vp.poc) == (5.0, 5.0, 5.0)


def test_compute_relative_volume_ratio():
    # Constant volume → ratio 1.0 once the window fills; warmup is NaN.
    rv = compute_relative_volume(pd.Series([10.0] * 5), window=5)
    assert rv.iloc[:4].isna().all()
    assert rv.iloc[-1] == pytest.approx(1.0)
    # A spike against a 10-flat window: 20 / mean(10,10,10,10,20)=12 → 1.6667
    rv2 = compute_relative_volume(pd.Series([10.0, 10.0, 10.0, 10.0, 20.0]), window=5)
    assert rv2.iloc[-1] == pytest.approx(20.0 / 12.0)
