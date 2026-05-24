"""Tests for scanner/regime.py (T-019 Chunk 4, regime classification).

Synthetic, deterministic, fixed-date fixtures (never ``Timestamp.now()``). The
``TrendState`` / ``LongerMASet`` each fixture feeds ``detect_regime`` are built by
the REAL Chunk-1/2 functions (``compute_ema_band`` → ``derive_trend_state``,
``compute_longer_set``), not hand-stubbed, so the substrate integration is exercised
end-to-end. Frames use ``Open == Close`` so each body is a clean point — a triangle
built from ``Open = prev Close`` reintroduces the ``find_swing_points`` turning-point
tie and suppresses every swing (the Chunk-3 lesson).

The short-history guard cannot be exercised non-vacuously at the *default*
``er_period`` (20): no 20- or 21-bar frame can produce an ``MSB`` — the swing window
is only indices 5..14, and a "higher low" needs a lower low >5 bars before it,
leaving no room for the third swing plus an out-of-window break (confirmed: 0 of
200k random 20-bar frames). So the warmup-ER guard is pinned at ``er_period=30``,
where an MSB fits — which proves the guard formula ``max(min_bars, er_period + 1)``
itself, not a hard-coded 21.
"""

import pandas as pd
import pytest

from scanner.indicators import detect_msbs, find_swing_points
from scanner.ma_indicators import compute_ema_band, compute_longer_set, compute_volume_profile
from scanner.regime import Regime, RegimeState, detect_regime
from scanner.trend_state import TrendDirection, derive_trend_state


def _frame(closes, vols=None, wick=0.3, start="2026-01-01"):
    """OHLCV frame with ``Open == Close`` (clean body points) and symmetric wicks."""
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq="D")
    close = pd.Series([float(c) for c in closes], index=idx)
    vol = pd.Series([float(v) for v in vols], index=idx) if vols is not None else pd.Series(1000.0, index=idx)
    return pd.DataFrame(
        {"Open": close, "High": close + wick, "Low": close - wick, "Close": close, "Volume": vol},
        index=idx,
    )


def _zig(n, lo, hi, half):
    """Triangle wave of ``n`` closes between ``lo`` and ``hi``, ``half``-bar half-period."""
    vals, pos, step, d = [], float(lo), (hi - lo) / half, 1
    for _ in range(n):
        vals.append(pos)
        pos += d * step
        if pos >= hi:
            pos, d = float(hi), -1
        elif pos <= lo:
            pos, d = float(lo), 1
    return vals


def _state(df):
    """Build (TrendState, LongerMASet) via the real Chunk-1/2 substrate."""
    longer = compute_longer_set(df["Close"])
    return derive_trend_state(df, compute_ema_band(df["Close"]), longer), longer


# A net-flat choppy tail riding a strong 290-bar uptrend: keeps the 13/25/36 AND
# 100/200/300 stacks fanned up (local == longer == UP) while the last 20 bars carry
# a sub-floor ER and a structure break.
_F1_ALIGN_TAIL = [252, 248, 252, 257, 262, 267, 262, 256, 250, 255, 260, 265, 259,
                  254, 258, 254, 249, 253, 257, 262, 258, 262, 257, 251, 257]

# An irregular (unequal-swing) range: RANGING with recent MSBs and low ER. With a
# value-area-inside profile it is ROTATIONAL (range wins over F1); with no profile
# it falls to F1/TRANSITIONAL.
_IRREGULAR_RANGE = [
    100, 106, 111, 106, 111, 104, 97, 90, 95, 102, 96, 91, 88, 88, 95, 88, 88, 88, 88, 95,
    89, 96, 103, 110, 103, 110, 105, 112, 105, 111, 104, 110, 112, 107, 102, 108, 112, 106,
    111, 112, 105, 112, 112, 106, 112, 107, 112, 112, 112, 107, 101, 108, 102, 109, 112, 112,
    112, 112, 107, 112, 112, 112, 107, 101, 96, 101, 94, 88, 88, 94, 99, 92, 88, 88, 94, 99,
    104, 99, 104, 97, 92, 88, 94, 100, 93, 99, 106, 99, 105, 110, 103, 110, 104, 98, 103, 96,
    103, 98, 92, 97, 90, 97, 104, 111, 106, 101, 95, 100, 94, 99, 105, 111, 106, 99, 94, 89,
    96, 103, 109, 102,
]

# A 31-bar choppy frame with a recent MSB (break bar 24) and a sub-floor real ER. At
# len 30 the ER is the warmup 0.0 (needs 31 bars) yet the MSB is present — the guard
# is what stops F1 false-firing; at len 31 the ER is real and F1 fires legitimately.
_WARMUP_MSB = [100, 96, 91, 95, 99, 94, 89, 94, 99, 103, 106, 109, 104, 101, 98, 102, 105,
               108, 103, 99, 103, 99, 102, 107, 112, 108, 104, 101, 106, 110, 105]


def test_steady_uptrend_is_trending():
    df = _frame([100 + i * 0.5 for i in range(320)])
    rs = detect_regime(df, *_state(df))
    assert rs.regime is Regime.TRENDING
    assert rs.htf_trend_direction is TrendDirection.UP
    assert not (rs.f1_trend_break or rs.f2_gapfill_stall or rs.range_valid)


def test_oscillating_range_with_value_area_inside_is_rotational():
    df = _frame(_zig(120, 95, 105, 8))  # uniform volume → central 70% value area
    vrvp = compute_volume_profile(df, bars=60)
    rs = detect_regime(df, *_state(df), vrvp=vrvp)
    assert rs.regime is Regime.ROTATIONAL
    assert rs.range_valid
    assert not (rs.f1_trend_break or rs.f2_gapfill_stall)


def test_value_area_wider_than_range_is_not_rotational():
    # Wide 60-bar profile span, but the recent swings tighten → the value area
    # breaches the structural range; the range is not volume-confirmed.
    df = _frame(_zig(75, 92, 108, 6) + _zig(45, 98, 102, 4))
    vrvp = compute_volume_profile(df, bars=60)
    rs = detect_regime(df, *_state(df), vrvp=vrvp)
    assert not rs.range_valid
    assert rs.regime is not Regime.ROTATIONAL


def test_recent_msb_with_low_er_is_f1_transitional():
    df = _frame(_IRREGULAR_RANGE)
    rs = detect_regime(df, *_state(df), vrvp=None)
    assert rs.f1_trend_break
    assert rs.regime is Regime.TRANSITIONAL


def test_f1_detected_under_ma_alignment_is_not_trending():
    # Both stacks fanned UP, yet a sub-floor ER + recent break makes it F1: alignment
    # alone does not earn TRENDING. F1 and TRENDING are disjoint by predicate
    # (F1 ⊆ {er < er_floor}), so this is the real, possible case.
    df = _frame([50 + 0.7 * i for i in range(290)] + _F1_ALIGN_TAIL)
    trend_state, longer = _state(df)
    assert trend_state.local_trend_direction is TrendDirection.UP
    assert trend_state.longer_trend_direction is TrendDirection.UP
    rs = detect_regime(df, trend_state, longer)
    assert rs.f1_trend_break
    assert rs.regime is Regime.TRANSITIONAL


def test_downtrend_gapfill_stall_is_f2_transitional():
    base = [300 - 0.6 * i for i in range(300)]
    df = _frame(base + [base[-1] + 2 * k for k in range(1, 9)])  # 8-bar bounce
    rs = detect_regime(df, *_state(df))
    assert rs.htf_trend_direction is TrendDirection.DOWN
    assert rs.f2_gapfill_stall
    assert rs.regime is Regime.TRANSITIONAL


def test_uptrend_gapfill_stall_is_f2_transitional():
    base = [50 + 0.6 * i for i in range(300)]
    df = _frame(base + [base[-1] - 2 * k for k in range(1, 9)])  # 8-bar dip
    rs = detect_regime(df, *_state(df))
    assert rs.htf_trend_direction is TrendDirection.UP
    assert rs.f2_gapfill_stall
    assert rs.regime is Regime.TRANSITIONAL


def test_f2_does_not_apply_without_dominant_htf_trend():
    # Same bounce geometry, but a 150-bar base leaves sma_300 NaN → htf FLAT →
    # F2 has no trend to fill against and does not fire.
    base = [300 - 0.6 * i for i in range(150)]
    df = _frame(base + [base[-1] + 2 * k for k in range(1, 9)])
    rs = detect_regime(df, *_state(df))
    assert rs.htf_trend_direction is TrendDirection.FLAT
    assert not rs.f2_gapfill_stall


def test_range_valid_takes_precedence_over_f1():
    # The irregular range is both volume-confirmed AND carries a recent MSB; clause
    # 5a (ROTATIONAL) precedes 5b (TRANSITIONAL), so the label is ROTATIONAL while
    # the diagnostic f1 flag is still recorded.
    df = _frame(_IRREGULAR_RANGE)
    vrvp = compute_volume_profile(df, bars=60)
    rs = detect_regime(df, *_state(df), vrvp=vrvp)
    assert rs.range_valid and rs.f1_trend_break
    assert rs.regime is Regime.ROTATIONAL


def test_vrvp_none_makes_rotational_unreachable():
    # A frame that is ROTATIONAL with a profile is never ROTATIONAL without one.
    df = _frame(_zig(120, 95, 105, 8))
    assert detect_regime(df, *_state(df), vrvp=compute_volume_profile(df, bars=60)).regime is Regime.ROTATIONAL
    rs = detect_regime(df, *_state(df), vrvp=None)
    assert not rs.range_valid
    assert rs.regime is not Regime.ROTATIONAL


@pytest.mark.parametrize("length", [29, 30, 31])
def test_warmup_er_guard_suppresses_f1_false_fire(length):
    # er_period=30 → guard threshold max(20, 31) = 31. A recent MSB is present from
    # len 30 onward, so the only thing stopping F1 at len 30 is the guard reading the
    # warmup ER (0.0 below 31 bars). At len 31 the ER is real and F1 fires.
    df = _frame(_WARMUP_MSB).iloc[:length]
    rs = detect_regime(df, *_state(df), er_period=30)
    if length < 31:
        assert rs == RegimeState(Regime.TRANSITIONAL, TrendDirection.FLAT, False, False, False)
    else:
        assert rs.f1_trend_break
        assert rs.regime is Regime.TRANSITIONAL


def test_warmup_guard_is_non_vacuous_msb_present_at_threshold():
    # At len 30 (the discriminating bar) a recent MSB genuinely exists — without the
    # er_period+1 guard, the warmup ER=0.0 would make F1 false-fire.
    df = _frame(_WARMUP_MSB).iloc[:30]
    recent_msbs = [m for m in detect_msbs(find_swing_points(df), df) if m.index >= len(df) - 10]
    assert recent_msbs  # the break is present...
    assert not detect_regime(df, *_state(df), er_period=30).f1_trend_break  # ...yet the guard suppresses F1


def test_empty_frame_degrades_to_transitional():
    df = _frame([])
    rs = detect_regime(df, *_state(df))
    assert rs == RegimeState(Regime.TRANSITIONAL, TrendDirection.FLAT, False, False, False)


def test_all_nan_substrate_degrades_without_raising():
    df = _frame([float("nan")] * 40)
    rs = detect_regime(df, *_state(df))
    assert rs.regime is Regime.TRANSITIONAL
    assert not (rs.f1_trend_break or rs.f2_gapfill_stall or rs.range_valid)
