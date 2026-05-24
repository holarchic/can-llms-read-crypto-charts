"""Tests for the Chunk-5b rotational/counter-trend detectors in scanner/setups_v2.py.

Synthetic, deterministic, fixed-date fixtures; MA/RSI substrate via the REAL
Chunk-1/2/4 functions; ``sr_levels`` hand-built ``SRLevelV2``. Deterministic shape
builders (``_zig`` / ``_grind`` / ``_broaden``) replace RNG.

Two detector properties drive the fixture shapes and are asserted as the contract:

- **gap_fill geometry ⟹ stall.** A counter-trend reclaim that is still short of the
  100MA (the geometry) is by definition an incomplete bounce, so the local stack is
  FLAT — i.e. ``regime.f2_gapfill_stall`` holds. A bounce strong enough to turn the
  local stack UP overshoots the 100MA and breaks the geometry. So the operative
  branch is the stall; an RSI divergence (when present) adds confluence rather than
  being an independent trigger.
- **RSI divergence needs no 50-reset, which forbids a deep pullback.** A 14-period
  RSI drops below 50 on any real pullback, so a valid (no-50-reset) divergence lives
  in a strong grind with shallow pullbacks — not a deep-lowed RANGING structure
  (whose deep low resets RSI; that is the 50-reset negative here). Overbought/oversold
  is therefore a confluence factor, not an emit gate.
"""

from datetime import datetime

import pandas as pd

from scanner.ma_indicators import compute_ema_band, compute_longer_set
from scanner.models import ConfluenceFactor, Direction, SetupTypeV2, SoftStopMode, SRLevelV2
from scanner.regime import detect_regime
from scanner.setups_v2 import (
    TFSubstrate,
    detect_gap_fill,
    detect_rsi_divergence_range,
    detect_sfp_range,
    setup_context_from_df,
)
from scanner.trend_state import derive_trend_state


def _frame(closes, highs=None, lows=None, wick=0.4):
    n = len(closes)
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    close = pd.Series([float(c) for c in closes], index=idx)
    hi = pd.Series([float(h) for h in highs], index=idx) if highs is not None else close + wick
    lo = pd.Series([float(x) for x in lows], index=idx) if lows is not None else close - wick
    return pd.DataFrame({"Open": close, "High": hi, "Low": lo, "Close": close, "Volume": [1000.0] * n}, index=idx)


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


def _zig(n, lo, hi, half):
    v, p, step, d = [], float(lo), (hi - lo) / half, 1
    for _ in range(n):
        v.append(p)
        p += d * step
        if p >= hi:
            p, d = float(hi), -1
        elif p <= lo:
            p, d = float(lo), 1
    return v


def _seg(a, b, n):
    return [a + (b - a) * k / n for k in range(1, n + 1)]


def _grind():
    """Strong steady rise then a sharp high, a tiny 3-pt dip (RSI stays > 50), and a
    less-sharp higher high → a bearish RSI divergence with no 50-reset."""
    s = [60.0] + _seg(60.0, 100.0, 70)
    s += _seg(s[-1], 106.0, 3) + _seg(106.0, 103.0, 3) + _seg(103.0, 108.0, 5) + _seg(108.0, 102.0, 6)
    return s


def _broaden():
    """Higher highs + lower lows (RANGING) — the deep low resets RSI through 50."""
    wm, p, d = [], 96.0, 1
    for _ in range(70):
        wm.append(p)
        p += d * 2.0
        if p >= 100:
            p, d = 100.0, -1
        elif p <= 92:
            p, d = 92.0, 1
    s = wm + _seg(wm[-1], 106, 4) + _seg(106, 93, 4) + _seg(93, 109, 4)
    s += _seg(109, 86, 4) + _seg(86, 111, 4) + _seg(111, 102, 6)
    return s


def _f2_down(base_len=300):
    base = [300 - 0.6 * i for i in range(base_len)]
    return base + [base[-1] + 2 * k for k in range(1, 9)]  # shallow bounce → stalled below the 100MA


def _continuation_uptrend():
    def line(i):
        return 50 + 0.6 * i
    return [line(i) - (4 if (i % 12 == 0 and i > 0) else 0) for i in range(280)] + [line(280 + j) for j in range(30)]


# --- gap_fill -----------------------------------------------------------------

def test_gap_fill_emits_on_stalled_counter_trend_bounce():
    df = _frame(_f2_down())  # htf DOWN, bounce above ribbon but short of the 100MA → f2 stall
    sub, ctx = _substrate(df), setup_context_from_df(df)
    assert sub.regime.f2_gapfill_stall  # the geometry-implies-stall property
    sig = detect_gap_fill(ctx, sub, [])
    assert len(sig) == 1
    s = sig[0]
    assert s.setup_type is SetupTypeV2.GAP_FILL
    assert s.direction is Direction.LONG  # counter-trend to the downtrend
    assert {ConfluenceFactor.COUNTER_TREND, ConfluenceFactor.GAPFILL_STALL} <= s.confluence
    assert s.hard_stop < s.entry
    assert s.soft_stop.mode is SoftStopMode.TREND_RECLAIM_AGAINST


def test_gap_fill_suppressed_without_dominant_htf_trend():
    df = _frame(_f2_down(base_len=150))  # 150 bars → sma_300 NaN → htf FLAT → no fill
    assert detect_gap_fill(setup_context_from_df(df), _substrate(df), []) == []


def test_gap_fill_suppressed_without_counter_trend_geometry():
    df = _frame(_continuation_uptrend())  # htf UP but price ABOVE the ribbon (with-trend) → no geometry
    assert detect_gap_fill(setup_context_from_df(df), _substrate(df), []) == []


# --- sfp_range ----------------------------------------------------------------

def _sweep_frame(last3, sweep_high):
    closes = _zig(110, 95, 105, 8) + list(last3)
    highs = [c + 0.4 for c in closes]
    highs[110] = sweep_high
    return _frame(closes, highs=highs)


def test_sfp_emits_at_sr_level_with_close_back_and_reaction():
    df = _sweep_frame([102.0, 101.0, 100.0], sweep_high=108.0)  # wick 108 pierces sr 106, closes back to 102
    sub, ctx = _substrate(df), setup_context_from_df(df)
    sig = detect_sfp_range(ctx, sub, [_lvl(106)])
    assert len(sig) == 1
    s = sig[0]
    assert s.setup_type is SetupTypeV2.SFP_RANGE
    assert s.direction is Direction.SHORT
    assert {ConfluenceFactor.SWEEP_RECLAIM, ConfluenceFactor.AT_SR} <= s.confluence
    assert s.hard_stop > s.entry  # stop beyond the sweep wick, above the swept level
    assert s.soft_stop.mode is SoftStopMode.LEVEL_RECLAIM_AGAINST


def test_sfp_emits_on_external_liquidity_beyond_range_without_at_sr():
    df = _sweep_frame([102.0, 101.0, 100.0], sweep_high=108.0)  # beyond range high 105, no sr supplied
    sig = detect_sfp_range(setup_context_from_df(df), _substrate(df), [])
    assert len(sig) == 1
    assert ConfluenceFactor.SWEEP_RECLAIM in sig[0].confluence
    assert ConfluenceFactor.AT_SR not in sig[0].confluence  # external-liquidity branch, not an sr level


def test_sfp_suppressed_without_close_back():
    df = _sweep_frame([106.5, 101.0, 100.0], sweep_high=108.0)  # closes 106.5 still above range high (wick only)
    assert detect_sfp_range(setup_context_from_df(df), _substrate(df), []) == []


def test_sfp_suppressed_without_reaction():
    df = _sweep_frame([102.0, 107.0, 100.0], sweep_high=108.0)  # next bar reclaims above the level → no reaction
    assert detect_sfp_range(setup_context_from_df(df), _substrate(df), []) == []


def test_sfp_suppressed_on_mid_range_sweep():
    df = _sweep_frame([103.0, 102.0, 101.0], sweep_high=104.5)  # wick stays inside the range, no sr
    assert detect_sfp_range(setup_context_from_df(df), _substrate(df), []) == []


# --- rsi_divergence_range -----------------------------------------------------

def test_rsi_divergence_emits_on_higher_high_lower_rsi():
    df = _frame(_grind())
    sub, ctx = _substrate(df), setup_context_from_df(df)
    sig = detect_rsi_divergence_range(ctx, sub, [])
    assert len(sig) == 1
    s = sig[0]
    assert s.setup_type is SetupTypeV2.RSI_DIVERGENCE_RANGE
    assert s.direction is Direction.SHORT
    assert ConfluenceFactor.DIVERGENCE in s.confluence
    assert ConfluenceFactor.OVERSOLD_OVERBOUGHT in s.confluence  # the reference high was overbought
    assert s.hard_stop > s.entry  # beyond the reacting swing's wick
    assert s.soft_stop.mode is SoftStopMode.LEVEL_RECLAIM_AGAINST


def test_rsi_divergence_suppressed_by_50_reset():
    df = _frame(_broaden())  # the deep low between the highs resets RSI through 50
    assert detect_rsi_divergence_range(setup_context_from_df(df), _substrate(df), []) == []


def test_rsi_divergence_suppressed_without_divergence():
    df = _frame(_zig(120, 95, 105, 8))  # equal highs, RSI confirms → no divergence
    assert detect_rsi_divergence_range(setup_context_from_df(df), _substrate(df), []) == []


def test_rsi_divergence_suppressed_in_trending_regime():
    df = _frame(_continuation_uptrend())
    sub = _substrate(df)
    assert sub.regime.regime.value == "trending"
    assert detect_rsi_divergence_range(setup_context_from_df(df), sub, []) == []


# --- degradation --------------------------------------------------------------

def test_rotational_detectors_degrade_on_short_history():
    df = _frame([100 + (i % 5) for i in range(25)])  # below min_bars
    sub, ctx = _substrate(df), setup_context_from_df(df)
    assert detect_gap_fill(ctx, sub, []) == []
    assert detect_sfp_range(ctx, sub, []) == []
    assert detect_rsi_divergence_range(ctx, sub, []) == []
