"""Tests for scanner/setups_v2.py (T-019 Chunk 5a, trend-family detectors).

Synthetic, deterministic, fixed-date fixtures. The MA substrate
(``TrendState``/``RegimeState``/bands) is built through the REAL Chunk-1/2/4
functions; ``sr_levels`` are hand-built ``SRLevelV2`` inputs (precise control over
the AT_SR / HTF-resistance / deviation-support conditions — ``derive_sr_levels`` is
already tested in Chunk 3). Frames use ``Open == Close`` for clean swing bodies
except where a directional impulse candle is needed (continuation).

The reclaim fixtures pass ``msb_lookback=25``: a reclaim's confirming structure break
sits a handful of bars behind the reclaim point, and the threshold is a v1 default
calibrated in Chunk 8b (the module default of 10 is intentionally tight). Both
reclaim fixtures carry ≥3 swing highs, so their ``weakness_flag`` assertions exercise
the real last-two-highs comparison, not the ``len < 2`` fast path.
"""

from datetime import datetime

import pandas as pd
import pytest

from scanner.ma_indicators import compute_ema_band, compute_longer_set
from scanner.models import (
    ConfluenceFactor,
    Direction,
    SetupTypeV2,
    SoftStopMode,
    SRLevelV2,
)
from scanner.regime import detect_regime
from scanner.setups_v2 import (
    TFSubstrate,
    detect_deviation_reclaim,
    detect_trend_continuation,
    detect_trend_reclaim,
    setup_context_from_df,
)
from scanner.trend_state import derive_trend_state


def _frame(closes, opens=None, wick=0.4):
    n = len(closes)
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    close = pd.Series([float(c) for c in closes], index=idx)
    open_ = pd.Series([float(o) for o in opens], index=idx) if opens is not None else close.copy()
    hi = pd.concat([open_, close], axis=1).max(axis=1) + wick
    lo = pd.concat([open_, close], axis=1).min(axis=1) - wick
    return pd.DataFrame({"Open": open_, "High": hi, "Low": lo, "Close": close, "Volume": [1000.0] * n}, index=idx)


def _substrate(df):
    bands = compute_ema_band(df["Close"])
    longer = compute_longer_set(df["Close"])
    ts = derive_trend_state(df, bands, longer)
    return TFSubstrate(bands, longer, ts, detect_regime(df, ts, longer))


def _lvl(price):
    return SRLevelV2(
        price=float(price), tf_source="1d", pivot_type="high", confirmations=1,
        vrvp_confluence=False, formed_at=datetime(2026, 1, 1), last_test=datetime(2026, 1, 2), quality=1.0,
    )


# --- fixtures (closes) ---------------------------------------------------------

def _continuation_closes():
    """Strong 300+ bar uptrend (longer + local UP, TRENDING) with shallow pullback
    swings; the caller supplies an impulse candle via the opens override."""
    def line(i):
        return 50 + 0.6 * i
    closes = [line(i) - (4 if (i % 12 == 0 and i > 0) else 0) for i in range(280)]
    closes += [line(280 + j) for j in range(30)]
    return closes


# Reclaim: decline-then-rally with a recent bullish MSB, a structural swing low below
# entry, and a full-ribbon reclaim. SEQ_F's last two swing highs extend (weakness
# False); SEQ_T's fail to extend (weakness True).
_RECLAIM = [120.0, 117.0, 112.0, 115.0, 111.0, 108.0, 111.0, 107.0, 110.0, 113.0, 116.0, 111.0,
            106.0, 103.0, 106.0, 102.0, 99.0, 102.0, 98.0, 94.0, 90.0, 87.0, 90.0, 87.0, 84.0,
            79.0, 75.0, 71.0, 67.0, 64.0, 67.0, 70.0, 67.0, 70.0, 65.0, 62.0, 57.0, 53.0, 50.0,
            53.0, 56.0, 53.0, 49.0, 46.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 42.0, 48.0,
            52.0, 56.0, 61.0, 67.0, 72.0, 76.0, 82.0, 79.0, 83.0, 87.0, 92.0, 89.0, 86.0, 92.0,
            89.0, 95.0, 99.0, 105.0, 102.0, 99.0, 103.0, 108.0, 105.0, 102.0, 99.0, 103.0, 107.0,
            104.0, 108.0, 113.0, 119.0, 124.0, 121.0, 127.0, 132.0, 137.0, 141.0, 138.0, 135.0,
            141.0, 145.0, 151.0, 148.0, 145.0, 149.0, 146.0, 150.0, 154.0, 158.0, 155.0, 160.0,
            165.0, 170.0]

_RECLAIM_WEAK = [120.0, 116.0, 113.0, 110.0, 107.0, 110.0, 107.0, 103.0, 106.0, 102.0, 99.0, 96.0,
                 92.0, 88.0, 85.0, 88.0, 91.0, 94.0, 90.0, 85.0, 80.0, 77.0, 73.0, 70.0, 73.0, 69.0,
                 64.0, 67.0, 63.0, 60.0, 57.0, 54.0, 50.0, 45.0, 45.0, 48.0, 45.0, 45.0, 45.0, 48.0,
                 45.0, 48.0, 45.0, 45.0, 45.0, 45.0, 48.0, 45.0, 45.0, 45.0, 48.0, 52.0, 57.0, 63.0,
                 60.0, 65.0, 70.0, 74.0, 71.0, 75.0, 72.0, 69.0, 75.0, 79.0, 76.0, 73.0, 79.0, 85.0,
                 91.0, 88.0, 85.0, 82.0, 79.0, 85.0, 91.0, 88.0, 85.0, 82.0, 79.0, 84.0, 81.0, 85.0,
                 89.0, 94.0, 91.0, 96.0, 93.0, 98.0, 102.0, 99.0, 105.0, 110.0, 114.0, 120.0, 117.0,
                 114.0, 111.0, 115.0, 120.0, 124.0, 130.0, 136.0, 133.0, 138.0, 135.0, 140.0]

_FLAT = [100, 98, 96, 93, 96, 94, 96, 93, 90, 92, 95, 93, 96, 94, 92, 95, 98, 101, 98, 96, 99,
         102, 100, 103, 101, 104, 102, 99, 96, 98, 100, 98, 96, 94, 97, 99, 97, 99, 96, 98, 95,
         98, 95, 98, 101, 98, 101, 99, 97, 99, 101, 104, 106, 109, 107, 109, 111, 114, 117, 114,
         116, 114, 111, 114, 112, 114, 111, 109, 112, 115, 118, 121, 119, 116, 114, 112, 109, 111,
         108, 111, 114, 112, 109, 107, 105, 107, 110, 113, 110, 107, 104, 101, 104, 106, 104, 101,
         103, 100, 97, 99, 101, 98, 95, 98, 101, 99, 101, 104, 102, 105, 108, 105, 102, 99, 101,
         104, 107, 110, 107, 105]


def _deviation_df():
    base = [50 + 0.5 * i for i in range(120)]
    return base[:-8] + [base[-8] - 10 + 1.5 * k for k in range(8)]


def _d4_inside_df():
    base = [50 + 0.6 * i for i in range(120)]
    return base[:-2] + [base[-3] - 2.5, base[-3] - 3.5]  # local UP but price INSIDE the band


# --- continuation -------------------------------------------------------------

def test_trend_continuation_emits_in_aligned_trending_with_impulse():
    closes = _continuation_closes()
    opens = list(closes)
    opens[302] = closes[302] - 7  # a green (long-direction) impulse body
    df = _frame(closes, opens=opens)
    sig = detect_trend_continuation(setup_context_from_df(df), _substrate(df), [])
    assert len(sig) == 1
    s = sig[0]
    assert s.setup_type is SetupTypeV2.TREND_CONTINUATION
    assert s.direction is Direction.LONG
    assert {ConfluenceFactor.TREND_ALIGNED, ConfluenceFactor.VERTICAL_ACCUMULATION} <= s.confluence
    assert s.hard_stop < s.entry  # a long's stop sits below entry
    assert s.soft_stop.mode is SoftStopMode.TREND_RECLAIM_AGAINST


def test_trend_continuation_suppressed_without_impulse_candle():
    df = _frame(_continuation_closes())  # Open == Close → no impulse body → slow grind
    assert detect_trend_continuation(setup_context_from_df(df), _substrate(df), []) == []


def test_trend_continuation_suppressed_by_counter_trend_impulse():
    closes = _continuation_closes()
    opens = list(closes)
    opens[302] = closes[302] + 7  # a red (counter-trend) impulse must NOT enable a long
    df = _frame(closes, opens=opens)
    assert detect_trend_continuation(setup_context_from_df(df), _substrate(df), []) == []


# --- reclaim ------------------------------------------------------------------

def test_trend_reclaim_emits_on_full_ribbon_reclaim_with_lower_tf_and_msb():
    df = _frame(_RECLAIM)
    sig = detect_trend_reclaim(setup_context_from_df(df), _substrate(df), [], lower_tf_reclaim=True, msb_lookback=25)
    assert len(sig) == 1
    s = sig[0]
    assert s.setup_type is SetupTypeV2.TREND_RECLAIM
    assert s.direction is Direction.LONG
    assert {ConfluenceFactor.TREND_ALIGNED, ConfluenceFactor.LOWER_TF_CONFIRMED} <= s.confluence
    assert s.hard_stop < s.entry
    assert s.weakness_flag is False  # last two swing highs extended (non-vacuous: ≥2 highs)


def test_trend_reclaim_weakness_flag_set_on_failed_extension():
    df = _frame(_RECLAIM_WEAK)
    sig = detect_trend_reclaim(setup_context_from_df(df), _substrate(df), [], lower_tf_reclaim=True, msb_lookback=25)
    assert len(sig) == 1
    assert sig[0].weakness_flag is True  # most recent swing high failed to exceed the prior


def test_trend_reclaim_suppressed_without_lower_tf_reclaim():
    df = _frame(_RECLAIM)
    assert detect_trend_reclaim(setup_context_from_df(df), _substrate(df), [], lower_tf_reclaim=False, msb_lookback=25) == []


def test_trend_reclaim_suppressed_by_nearby_htf_resistance():
    df = _frame(_RECLAIM)
    sub, ctx = _substrate(df), setup_context_from_df(df)
    ema36, atr = sub.bands.slow.iloc[-1], ctx.atr
    resistance = [_lvl(ema36 + 0.5 * atr)]  # within htf_res_tol (1.0·ATR) above the reclaim
    assert detect_trend_reclaim(ctx, sub, resistance, lower_tf_reclaim=True, msb_lookback=25) == []


def test_trend_reclaim_requires_full_ribbon_not_just_above_ema36():
    df = _frame(_d4_inside_df())  # price above EMA36 but still INSIDE the band
    assert detect_trend_reclaim(setup_context_from_df(df), _substrate(df), [], lower_tf_reclaim=True, msb_lookback=25) == []


# --- deviation ----------------------------------------------------------------

def test_deviation_reclaim_emits_with_sr_support():
    df = _frame(_deviation_df())
    sub, ctx = _substrate(df), setup_context_from_df(df)
    sig = detect_deviation_reclaim(ctx, sub, [_lvl(sub.bands.slow.iloc[-1])])
    assert len(sig) == 1
    assert sig[0].setup_type is SetupTypeV2.DEVIATION_RECLAIM
    assert ConfluenceFactor.AT_SR in sig[0].confluence


def test_deviation_reclaim_requires_sr_support():
    df = _frame(_deviation_df())
    assert detect_deviation_reclaim(setup_context_from_df(df), _substrate(df), []) == []


# --- shared negatives + degradation -------------------------------------------

def test_flat_local_trend_emits_nothing():
    df = _frame(_FLAT)
    sub, ctx = _substrate(df), setup_context_from_df(df)
    assert detect_trend_continuation(ctx, sub, []) == []
    assert detect_trend_reclaim(ctx, sub, [], lower_tf_reclaim=True) == []
    assert detect_deviation_reclaim(ctx, sub, []) == []


def test_short_history_degrades_to_empty():
    df = _frame([50 + i for i in range(25)])  # below min_bars
    sub, ctx = _substrate(df), setup_context_from_df(df)
    assert detect_trend_continuation(ctx, sub, []) == []
    assert detect_trend_reclaim(ctx, sub, [], lower_tf_reclaim=True) == []
    assert detect_deviation_reclaim(ctx, sub, []) == []


def test_setup_context_rejects_empty_frame():
    with pytest.raises(ValueError):
        setup_context_from_df(_frame([]))
