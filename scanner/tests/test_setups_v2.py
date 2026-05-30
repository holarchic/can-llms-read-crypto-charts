"""Tests for scanner/setups_v2.py (T-019 Chunk 5a, trend-family detectors).

Synthetic, deterministic, fixed-date fixtures. The MA substrate
(``TrendState``/``RegimeState``/bands) is built through the REAL Chunk-1/2/4
functions; ``sr_levels`` are hand-built ``SRLevelV2`` inputs (precise control over
the AT_SR / HTF-resistance / deviation-support conditions — ``derive_sr_levels`` is
already tested in Chunk 3). Frames use ``Open == Close`` for clean swing bodies
except where a directional impulse candle is needed (continuation).

The reclaim fixtures use the detector's default ``msb_lookback`` (25): the local
stack fans back up ~15-25 bars after the confirming MSB break (EMA-fanning latency),
so the break is already that old by the time the reclaim is visible — a tighter
window would make trend_reclaim near-dead. Both reclaim fixtures carry ≥3 swing
highs, so their ``weakness_flag`` assertions exercise the real last-two-highs
comparison, not the ``len < 2`` fast path.
"""

from datetime import datetime

import pandas as pd
import pytest

from scanner.ma_indicators import LongerMASet, compute_ema_band, compute_longer_set
from scanner.models import (
    BoundarySignal,
    ConfluenceFactor,
    Direction,
    SetupTypeV2,
    SoftStopMode,
    SpineContext,
    SRLevelV2,
    Structure,
    SwingPoint,
)
from scanner.regime import detect_regime
from scanner.setups_v2 import (
    _MAX_CONTINUATIONS,
    TFSubstrate,
    _continuation_count,
    detect_deviation_reclaim,
    detect_trend_continuation,
    detect_trend_reclaim,
    setup_context_from_df,
    setup_textbook,
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


def _lvl(price, zone_pct=0.005):
    """Test fixture: AT_SR levels for trend-family detectors. ``zone_pct`` is the
    one-sided zone half-width relative to ``price`` (default 0.5% — wide enough that
    a price equal to ``price`` sits inside the zone for new zone-membership semantics).
    """
    p = float(price)
    pad = abs(p) * zone_pct
    return SRLevelV2(
        price=p, tf_source="1d", pivot_type="high", confirmations=1,
        vrvp_confluence=False, formed_at=datetime(2026, 1, 1), last_test=datetime(2026, 1, 2),
        zone_low=p - pad, zone_high=p + pad, quality=1.0,
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
    sig = detect_trend_reclaim(setup_context_from_df(df), _substrate(df), [], lower_tf_reclaim=True)
    assert len(sig) == 1
    s = sig[0]
    assert s.setup_type is SetupTypeV2.TREND_RECLAIM
    assert s.direction is Direction.LONG
    assert {ConfluenceFactor.TREND_ALIGNED, ConfluenceFactor.LOWER_TF_CONFIRMED} <= s.confluence
    assert s.hard_stop < s.entry
    assert s.weakness_flag is False  # last two swing highs extended (non-vacuous: ≥2 highs)


def test_trend_reclaim_weakness_flag_set_on_failed_extension():
    df = _frame(_RECLAIM_WEAK)
    sig = detect_trend_reclaim(setup_context_from_df(df), _substrate(df), [], lower_tf_reclaim=True)
    assert len(sig) == 1
    assert sig[0].weakness_flag is True  # most recent swing high failed to exceed the prior


def test_trend_reclaim_suppressed_without_lower_tf_reclaim():
    df = _frame(_RECLAIM)
    assert detect_trend_reclaim(setup_context_from_df(df), _substrate(df), [], lower_tf_reclaim=False) == []


def test_trend_reclaim_suppressed_by_nearby_htf_resistance():
    df = _frame(_RECLAIM)
    sub, ctx = _substrate(df), setup_context_from_df(df)
    ema36, atr = sub.bands.slow.iloc[-1], ctx.atr
    resistance = [_lvl(ema36 + 0.5 * atr)]  # within htf_res_tol (1.0·ATR) above the reclaim
    assert detect_trend_reclaim(ctx, sub, resistance, lower_tf_reclaim=True) == []


def test_trend_reclaim_requires_full_ribbon_not_just_above_ema36():
    df = _frame(_d4_inside_df())  # price above EMA36 but still INSIDE the band
    assert detect_trend_reclaim(setup_context_from_df(df), _substrate(df), [], lower_tf_reclaim=True) == []


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


# --- structural textbook predicate (T-020.4 / SPEC #44) -----------------------

_TC = SetupTypeV2.TREND_CONTINUATION


def _bsig(setup_type, *, confluence=frozenset(), weakness_flag=False, continuation_count=None):
    """Minimal BoundarySignal carrying only the fields ``setup_textbook`` reads."""
    ctx = SpineContext(
        structure=Structure.BULLISH, sr_levels=[], range_bounds=None,
        atr=1.0, current_price=1.0, fib_levels=None, regime="trending",
    )
    return BoundarySignal(
        coin="TEST", setup_type=setup_type, direction=Direction.LONG, confidence=0.5,
        key_level=1.0, context=ctx, description="t", setup_tf="4h",
        confluence=confluence, weakness_flag=weakness_flag, continuation_count=continuation_count,
    )


def test_setup_textbook_continuation_textbook_passes():
    assert setup_textbook(_bsig(_TC, confluence=frozenset({ConfluenceFactor.RETEST}))) is True


def test_setup_textbook_continuation_without_retest_rejected():
    # .6 — the mid-air momentum chase. A real, non-vacuous reject.
    assert setup_textbook(_bsig(_TC, confluence=frozenset({ConfluenceFactor.TREND_ALIGNED}))) is False


def test_setup_textbook_continuation_with_weakness_rejected():
    # .5 — first-weakness disqualifies even WITH a retest.
    bs = _bsig(_TC, confluence=frozenset({ConfluenceFactor.RETEST}), weakness_flag=True)
    assert setup_textbook(bs) is False


def test_setup_textbook_reclaim_with_weakness_passes():
    # .5 is trend_continuation-only: reclaim weakness is the deterioration the setup repairs.
    assert setup_textbook(_bsig(SetupTypeV2.TREND_RECLAIM, weakness_flag=True)) is True
    assert setup_textbook(_bsig(SetupTypeV2.DEVIATION_RECLAIM, weakness_flag=True)) is True


def test_setup_textbook_non_continuation_without_retest_passes():
    # .6 is trend_continuation-only: range / counter-trend setups need no RETEST.
    assert setup_textbook(_bsig(SetupTypeV2.SFP_RANGE)) is True
    assert setup_textbook(_bsig(SetupTypeV2.GAP_FILL)) is True
    assert setup_textbook(_bsig(SetupTypeV2.RSI_DIVERGENCE_RANGE)) is True


def test_setup_textbook_rejects_real_continuation_without_retest():
    """Anti-vacuity against REAL producer output: a real ``detect_trend_continuation``
    emission the detector did not tag with RETEST is a breakout chase, and the predicate
    rejects it — proving the gate reads facts the detector actually emits, not just
    synthetic signals."""
    closes = _continuation_closes()
    opens = list(closes)
    opens[302] = closes[302] - 7  # green impulse → emits
    df = _frame(closes, opens=opens)
    emission = detect_trend_continuation(setup_context_from_df(df), _substrate(df), [])[0]
    assert ConfluenceFactor.RETEST not in emission.confluence  # breakout, not a retest
    assert emission.weakness_flag is False
    bs = _bsig(emission.setup_type, confluence=emission.confluence, weakness_flag=emission.weakness_flag)
    assert setup_textbook(bs) is False
    # ...and the same real emission WITH a retest tag passes (the textbook branch is reachable):
    bs_retest = _bsig(_TC, confluence=emission.confluence | {ConfluenceFactor.RETEST})
    assert setup_textbook(bs_retest) is True


# --- continuation cap (T-020.9b / SPEC #48) -----------------------------------

def test_setup_textbook_continuation_cap_rejects_fourth():
    """.9b — the 4th+ continuation of an established trend is not textbook (count > 3), even with a
    retest and no weakness. Anti-vacuity: the SAME signal at count 3 passes."""
    tb = frozenset({ConfluenceFactor.RETEST})
    assert setup_textbook(_bsig(_TC, confluence=tb, continuation_count=_MAX_CONTINUATIONS)) is True       # 3rd — textbook
    assert setup_textbook(_bsig(_TC, confluence=tb, continuation_count=_MAX_CONTINUATIONS + 1)) is False  # 4th — capped


def test_setup_textbook_continuation_cap_none_passes():
    # None (the default; the other 5 setups + v1) is never capped.
    assert setup_textbook(_bsig(_TC, confluence=frozenset({ConfluenceFactor.RETEST}),
                                continuation_count=None)) is True


def test_setup_textbook_cap_is_continuation_only():
    # the cap is trend_continuation-scoped: a high count on another setup never rejects.
    assert setup_textbook(_bsig(SetupTypeV2.SFP_RANGE, continuation_count=9)) is True


def _longer(stack):
    """A LongerMASet whose per-bar ``stack_direction`` follows ``stack`` (one char/bar):
    'u' → 3>2>1 (UP), 'd' → 1<2<3 (DOWN), 'f' → 2==2==2 (FLAT), 'n' → NaN (pre-warmup)."""
    nan = float("nan")
    trip = {"u": (3.0, 2.0, 1.0), "d": (1.0, 2.0, 3.0), "f": (2.0, 2.0, 2.0), "n": (nan, nan, nan)}
    idx = pd.RangeIndex(len(stack))
    cols = list(zip(*[trip[c] for c in stack]))
    return LongerMASet(pd.Series(cols[0], index=idx), pd.Series(cols[1], index=idx), pd.Series(cols[2], index=idx))


def _sw(index, price, kind="high"):
    return SwingPoint(index=index, timestamp=datetime(2026, 1, 1), price=price, type=kind, body_price=price)


def test_continuation_count_episode_start_and_monotonic_pushes():
    # 5 FLAT then 10 UP → episode starts at index 5. Highs at 3 (pre-episode, excluded),
    # 6/9/12 monotonic-up = 3 pushes; the non-new-high at 11 does not count.
    longer = _longer("f" * 5 + "u" * 10)
    swings = [_sw(3, 100), _sw(6, 101), _sw(9, 103), _sw(11, 102), _sw(12, 105)]
    assert _continuation_count(longer, swings, Direction.LONG) == 3


def test_continuation_count_flat_bar_resets_episode():
    # a single FLAT bar at index 8 resets the run → episode = bars 9..14; only highs at
    # 10/13 count (the earlier high at 6 is pre-reset).
    longer = _longer("u" * 8 + "f" + "u" * 6)
    swings = [_sw(6, 101), _sw(10, 102), _sw(13, 104)]
    assert _continuation_count(longer, swings, Direction.LONG) == 2


def test_continuation_count_no_flip_counts_whole_defined_window():
    # whole defined window UP (no flip) → episode = bar 0; all 4 monotonic highs count.
    longer = _longer("u" * 12)
    swings = [_sw(1, 100), _sw(4, 101), _sw(7, 102), _sw(10, 103)]
    assert _continuation_count(longer, swings, Direction.LONG) == 4


def test_continuation_count_right_edge_lag_confirmed_only():
    # confirmed-swings-only: a 4th push not yet in `swings` reads as 3; once find_swing_points
    # confirms it (it appears in the list), it reads 4 — the deliberate right-edge lag.
    longer = _longer("u" * 20)
    confirmed3 = [_sw(2, 100), _sw(6, 101), _sw(10, 102)]
    assert _continuation_count(longer, confirmed3, Direction.LONG) == 3
    assert _continuation_count(longer, confirmed3 + [_sw(14, 103)], Direction.LONG) == 4


def test_continuation_count_short_uses_lows():
    longer = _longer("d" * 12)
    swings = [_sw(2, 100, "low"), _sw(6, 98, "low"), _sw(10, 95, "low"), _sw(5, 200, "high")]
    assert _continuation_count(longer, swings, Direction.SHORT) == 3  # the high is ignored


def test_continuation_count_excludes_pre_warmup_pushes():
    # the recent-window limitation: NaN warmup bars (pre-SMA300) resolve to FLAT via
    # stack_direction, so the episode cannot start before the stack is defined — a push in the
    # warmup region (here a higher high at index 4) is excluded from the count.
    longer = _longer("n" * 6 + "u" * 8)   # bars 0-5 undefined, 6-13 UP → episode starts at 6
    swings = [_sw(2, 100), _sw(4, 150),   # pre-warmup highs (index < 6) — excluded
              _sw(7, 101), _sw(10, 102), _sw(12, 104)]   # in-episode monotonic → 3
    assert _continuation_count(longer, swings, Direction.LONG) == 3


def test_continuation_count_zero_when_not_in_direction_at_edge():
    # last bar FLAT → not an established `direction` run at the edge → 0 (defensive; the
    # detector's alignment+TRENDING gates ensure the real edge is in `direction`).
    longer = _longer("u" * 10 + "f")
    assert _continuation_count(longer, [_sw(2, 100), _sw(5, 101)], Direction.LONG) == 0


def test_detect_trend_continuation_attaches_count():
    # the real detector populates continuation_count (an int, not None) on its emission.
    closes = _continuation_closes()
    opens = list(closes)
    opens[302] = closes[302] - 7  # green impulse → emits
    df = _frame(closes, opens=opens)
    sig = detect_trend_continuation(setup_context_from_df(df), _substrate(df), [])[0]
    assert isinstance(sig.continuation_count, int) and sig.continuation_count >= 0
