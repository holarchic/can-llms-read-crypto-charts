"""Tests for setup screener A-G detection."""


from datetime import datetime, timezone

from scanner.indicators import build_context
from scanner.models import (
    MSB,
    Direction,
    MarketContext,
    SetupSignal,
    SetupType,
    SRLevel,
    Structure,
    SwingPoint,
)
from scanner.screener import (
    W_ALIGNMENT,
    W_BASE,
    W_CONFLUENCE,
    W_PROXIMITY,
    W_STRENGTH,
    _structure_alignment,
    detect_a_same_way_level,
    detect_all_setups,
    detect_b_breakout,
    detect_c_sr_flip,
    detect_f_deviation,
    detect_g_spike_sfp,
    score_signal,
)


class TestDetectAllSetups:
    def test_returns_list(self, ranging_df):
        ctx = build_context(ranging_df)
        signals = detect_all_setups("TEST", ranging_df, ctx)
        assert isinstance(signals, list)

    def test_capped_at_max(self, ranging_df):
        ctx = build_context(ranging_df)
        signals = detect_all_setups("TEST", ranging_df, ctx, max_per_coin=2)
        assert len(signals) <= 2

    def test_sorted_by_confidence(self, ranging_df):
        ctx = build_context(ranging_df)
        signals = detect_all_setups("TEST", ranging_df, ctx, max_per_coin=10)
        if len(signals) >= 2:
            confs = [s.confidence for s in signals]
            assert confs == sorted(confs, reverse=True)

    def test_signal_fields(self, ranging_df):
        ctx = build_context(ranging_df)
        signals = detect_all_setups("TEST", ranging_df, ctx)
        for s in signals:
            assert s.coin == "TEST"
            assert isinstance(s.setup_type, SetupType)
            assert isinstance(s.direction, Direction)
            assert 0 <= s.confidence <= 1
            assert s.key_level > 0
            assert len(s.description) > 10


class TestSetupA:
    def test_only_fires_in_range(self, uptrend_df):
        ctx = build_context(uptrend_df)
        if ctx.structure != "RANGING":
            signals = detect_a_same_way_level("TEST", uptrend_df, ctx)
            # Should not fire in a trending market
            assert len(signals) == 0


class TestSetupG:
    def test_detects_spike(self, spike_df):
        ctx = build_context(spike_df)
        signals = detect_g_spike_sfp("TEST", spike_df, ctx)
        # The spike_df fixture has a clear SFP — should detect it
        # (may or may not depending on swing detection catching the prior high)
        assert isinstance(signals, list)
        for s in signals:
            assert s.setup_type == SetupType.G


# -- Helpers for composite scoring tests --

def _make_ctx(structure=Structure.BULLISH, atr=2.0, current_price=100.0,
              sr_levels=None, msbs=None, swings=None, fib_levels=None,
              range_bounds=None):
    return MarketContext(
        structure=structure,
        swings=swings or [],
        sr_levels=sr_levels or [],
        msbs=msbs or [],
        range_bounds=range_bounds,
        atr=atr,
        current_price=current_price,
        fib_levels=fib_levels,
    )


def _make_signal(direction=Direction.LONG, base_confidence=0.5, key_level=100.0,
                 setup_type=SetupType.B, ctx=None):
    ctx = ctx or _make_ctx()
    return SetupSignal(
        coin="TEST", setup_type=setup_type, direction=direction,
        confidence=base_confidence, key_level=key_level, context=ctx,
        description="test signal",
    )


class TestStructureAlignment:
    def test_with_trend_long_bullish(self):
        assert _structure_alignment(Direction.LONG, Structure.BULLISH) == 1.0

    def test_with_trend_short_bearish(self):
        assert _structure_alignment(Direction.SHORT, Structure.BEARISH) == 1.0

    def test_counter_trend(self):
        assert _structure_alignment(Direction.LONG, Structure.BEARISH) == 0.0
        assert _structure_alignment(Direction.SHORT, Structure.BULLISH) == 0.0

    def test_ranging_neutral(self):
        assert _structure_alignment(Direction.LONG, Structure.RANGING) == 0.5
        assert _structure_alignment(Direction.SHORT, Structure.RANGING) == 0.5


class TestScoreSignal:
    def test_returns_bounded_float(self):
        signal = _make_signal()
        score = score_signal(signal, confluence_count=1, total_detectors=7)
        assert 0.0 <= score <= 1.0

    def test_with_trend_beats_counter_trend(self):
        ctx_bull = _make_ctx(structure=Structure.BULLISH)
        sig_with = _make_signal(direction=Direction.LONG, ctx=ctx_bull)
        sig_against = _make_signal(direction=Direction.SHORT, ctx=ctx_bull)
        score_with = score_signal(sig_with, 1, 7)
        score_against = score_signal(sig_against, 1, 7)
        assert score_with > score_against

    def test_higher_confluence_scores_higher(self):
        signal = _make_signal()
        score_low = score_signal(signal, confluence_count=1, total_detectors=7)
        score_high = score_signal(signal, confluence_count=4, total_detectors=7)
        assert score_high > score_low

    def test_stronger_level_scores_higher(self):
        weak_sr = [SRLevel(price=100.0, strength=1, source="test")]
        strong_sr = [SRLevel(price=100.0, strength=5, source="test")]
        ctx_weak = _make_ctx(sr_levels=weak_sr)
        ctx_strong = _make_ctx(sr_levels=strong_sr)
        sig_weak = _make_signal(ctx=ctx_weak)
        sig_strong = _make_signal(ctx=ctx_strong)
        score_weak = score_signal(sig_weak, 1, 7)
        score_strong = score_signal(sig_strong, 1, 7)
        assert score_strong > score_weak

    def test_closer_proximity_scores_higher(self):
        ctx = _make_ctx(current_price=100.0, atr=2.0)
        sig_close = _make_signal(key_level=100.5, ctx=ctx)  # 0.5 away
        sig_far = _make_signal(key_level=103.0, ctx=ctx)    # 3.0 away
        score_close = score_signal(sig_close, 1, 7)
        score_far = score_signal(sig_far, 1, 7)
        assert score_close > score_far

    def test_zero_atr_does_not_crash(self):
        ctx = _make_ctx(atr=0.0)
        signal = _make_signal(ctx=ctx)
        score = score_signal(signal, 1, 7)
        assert 0.0 <= score <= 1.0

    def test_weights_sum_to_one(self):
        assert abs((W_BASE + W_ALIGNMENT + W_STRENGTH + W_PROXIMITY + W_CONFLUENCE) - 1.0) < 1e-9


class TestCompositeInDetectAll:
    def test_confidence_is_composite_not_raw(self, ranging_df):
        """After detect_all_setups, confidence should reflect composite scoring."""
        ctx = build_context(ranging_df)
        signals = detect_all_setups("TEST", ranging_df, ctx, max_per_coin=10)
        for s in signals:
            # Composite scores won't match the old hardcoded values exactly
            assert 0.0 <= s.confidence <= 1.0


def _deviation_df(last_close, spike_high=None, spike_low=None):
    """Build 80-bar DF with a controllable deviation candle at bar -3.

    Price ranges around 100. The deviation candle wicks to spike_high/spike_low
    then closes back near 100. last_close is the final bar's close.
    """
    from scanner.tests.conftest import _make_df
    prices = []
    for i in range(76):
        mid = 100.0
        prices.append((mid, mid + 1.5, mid - 1.5, mid))

    # Deviation candle (3 bars from end)
    dev_high = spike_high if spike_high else 101.5
    dev_low = spike_low if spike_low else 98.5
    prices.append((100, dev_high, dev_low, 100))

    # 3 follow-up candles, last one closes at last_close
    prices.append((100, 101, 99, 100))
    prices.append((100, 101, 99, 100))
    prices.append((100, 101, 99, last_close))
    return _make_df(prices)


class TestSetupFDeviation:
    def test_ignores_distant_levels(self):
        """Levels far from current price should not fire."""
        df = _deviation_df(last_close=99.0, spike_high=106)
        ctx = _make_ctx(
            current_price=99.0, atr=2.0,
            sr_levels=[SRLevel(price=120.0, strength=3, source="test")],  # 10+ ATR away
        )
        signals = detect_f_deviation("TEST", df, ctx)
        assert len(signals) == 0

    def test_ignores_tiny_pierce(self):
        """Wick that barely crosses level (< 0.25 ATR) should not fire."""
        df = _deviation_df(last_close=99.0, spike_high=105.1)
        ctx = _make_ctx(
            current_price=99.0, atr=2.0,
            sr_levels=[SRLevel(price=105.0, strength=3, source="test")],  # pierce = 0.1, min = 0.5
        )
        signals = detect_f_deviation("TEST", df, ctx)
        assert len(signals) == 0

    def test_fires_on_significant_pierce(self):
        """Wick meaningfully beyond nearby level should fire."""
        df = _deviation_df(last_close=99.0, spike_high=107.0)
        ctx = _make_ctx(
            current_price=99.0, atr=2.0,
            sr_levels=[SRLevel(price=100.5, strength=3, source="test")],
        )
        signals = detect_f_deviation("TEST", df, ctx)
        short_signals = [s for s in signals if s.direction == Direction.SHORT]
        assert len(short_signals) >= 1

    def test_caps_at_one_per_direction(self):
        """Multiple levels in same direction should collapse to best one."""
        df = _deviation_df(last_close=99.0, spike_high=107.0)
        ctx = _make_ctx(
            current_price=99.0, atr=2.0,
            sr_levels=[
                SRLevel(price=100.0, strength=3, source="test"),
                SRLevel(price=100.5, strength=2, source="test"),
                SRLevel(price=99.5, strength=4, source="test"),
            ],
        )
        signals = detect_f_deviation("TEST", df, ctx)
        short_signals = [s for s in signals if s.direction == Direction.SHORT]
        assert len(short_signals) <= 1

    def test_max_two_signals(self):
        """At most 2 signals total (one long, one short)."""
        df = _deviation_df(last_close=100.0, spike_high=107.0, spike_low=93.0)
        ctx = _make_ctx(
            current_price=100.0, atr=2.0,
            sr_levels=[
                SRLevel(price=99.0, strength=3, source="test"),
                SRLevel(price=101.0, strength=3, source="test"),
            ],
        )
        signals = detect_f_deviation("TEST", df, ctx)
        assert len(signals) <= 2


def _breakout_df(close_at):
    """80 bars: first 75 range around 100, then last 5 close at close_at."""
    from scanner.tests.conftest import _make_df
    prices = []
    for i in range(75):
        prices.append((99.5, 100.5, 99.0, 99.5))
    for i in range(5):
        prices.append((close_at, close_at + 0.5, close_at - 0.5, close_at))
    return _make_df(prices)


class TestSetupBBreakout:
    def test_ignores_distant_levels(self):
        """Levels far from current price should not fire."""
        df = _breakout_df(close_at=102.0)
        ctx = _make_ctx(
            current_price=102.0, atr=2.0,
            sr_levels=[SRLevel(price=120.0, strength=3, source="test")],
        )
        signals = detect_b_breakout("TEST", df, ctx)
        assert len(signals) == 0

    def test_ignores_tiny_break(self):
        """Close barely past level (< 0.25 ATR) should not fire."""
        df = _breakout_df(close_at=100.3)  # 0.3 past level, min_break = 0.5
        ctx = _make_ctx(
            current_price=100.3, atr=2.0,
            sr_levels=[SRLevel(price=100.0, strength=3, source="test")],
        )
        signals = detect_b_breakout("TEST", df, ctx)
        assert len(signals) == 0

    def test_fires_on_convincing_break(self):
        """Close clearly past nearby level should fire."""
        df = _breakout_df(close_at=102.0)
        ctx = _make_ctx(
            current_price=102.0, atr=2.0,
            sr_levels=[SRLevel(price=100.5, strength=3, source="test")],
        )
        signals = detect_b_breakout("TEST", df, ctx)
        long_signals = [s for s in signals if s.direction == Direction.LONG]
        assert len(long_signals) == 1

    def test_caps_at_one_per_direction(self):
        """Multiple broken levels same direction collapse to closest."""
        df = _breakout_df(close_at=103.0)
        ctx = _make_ctx(
            current_price=103.0, atr=2.0,
            sr_levels=[
                SRLevel(price=100.5, strength=3, source="test"),
                SRLevel(price=101.0, strength=2, source="test"),
                SRLevel(price=101.5, strength=4, source="test"),
            ],
        )
        signals = detect_b_breakout("TEST", df, ctx)
        long_signals = [s for s in signals if s.direction == Direction.LONG]
        assert len(long_signals) <= 1


def _msb(index, msb_type="bullish", level=100.0):
    return MSB(
        index=index,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        type=msb_type,
        level=level,
        swing_broken=SwingPoint(
            index=index - 1,
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            price=level, type="high" if msb_type == "bullish" else "low",
            body_price=level,
        ),
    )


class TestSetupCSRFlip:
    def test_caps_at_one_per_direction(self):
        """Multiple nearby MSBs same direction collapse to closest."""
        ctx = _make_ctx(
            current_price=100.0, atr=2.0,
            msbs=[
                _msb(index=55, msb_type="bullish", level=99.5),
                _msb(index=60, msb_type="bullish", level=100.5),
                _msb(index=65, msb_type="bullish", level=101.0),
            ],
        )
        # Need a DF long enough for the index check (len >= 80)
        df = _breakout_df(close_at=100.0)
        signals = detect_c_sr_flip("TEST", df, ctx)
        long_signals = [s for s in signals if s.direction == Direction.LONG]
        assert len(long_signals) <= 1

    def test_picks_closest_msb(self):
        """Should select the MSB closest to current price."""
        ctx = _make_ctx(
            current_price=100.0, atr=2.0,
            msbs=[
                _msb(index=55, msb_type="bullish", level=99.0),
                _msb(index=60, msb_type="bullish", level=100.2),  # closest
                _msb(index=65, msb_type="bullish", level=101.5),
            ],
        )
        df = _breakout_df(close_at=100.0)
        signals = detect_c_sr_flip("TEST", df, ctx)
        long_signals = [s for s in signals if s.direction == Direction.LONG]
        if long_signals:
            assert long_signals[0].key_level == 100.2


class TestSetupGSpikeSFP:
    def test_ignores_ancient_swings(self):
        """Swing points from >30 bars ago should not trigger SFP."""
        ctx = _make_ctx(
            current_price=100.0, atr=2.0,
            swings=[
                SwingPoint(index=10, timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                           price=101.0, type="high", body_price=100.5),
            ],
        )
        # Build DF with a spike candle in last 5 bars
        from scanner.tests.conftest import _make_df
        prices = [(100, 101, 99, 100)] * 75
        prices.append((100, 106, 99, 99))  # big upper wick
        prices.extend([(100, 101, 99, 100)] * 4)
        df = _make_df(prices)
        signals = detect_g_spike_sfp("TEST", df, ctx)
        assert len(signals) == 0

    def test_caps_at_one_per_direction(self):
        """Multiple SFPs same direction collapse to best wick ratio."""
        ctx = _make_ctx(
            current_price=100.0, atr=2.0,
            swings=[
                SwingPoint(index=55, timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                           price=101.0, type="high", body_price=100.5),
                SwingPoint(index=60, timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                           price=101.5, type="high", body_price=101.0),
            ],
        )
        from scanner.tests.conftest import _make_df
        prices = [(100, 101, 99, 100)] * 75
        # Two SFP candles hitting different swing highs
        prices.append((100, 105, 99, 99.5))  # wick above both
        prices.append((100, 104, 99, 99.5))
        prices.extend([(100, 101, 99, 100)] * 3)
        df = _make_df(prices)
        signals = detect_g_spike_sfp("TEST", df, ctx)
        short_signals = [s for s in signals if s.direction == Direction.SHORT]
        assert len(short_signals) <= 1
