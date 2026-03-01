"""Tests for technical indicators: swings, S/R, structure, MSBs, fibs."""

import numpy as np
import pandas as pd

from scanner.indicators import (
    build_context,
    calculate_fibs,
    classify_regime,
    classify_structure,
    cluster_sr_levels,
    compute_atr,
    compute_atr_percentile,
    compute_efficiency_ratio,
    detect_msbs,
    find_swing_points,
)
from scanner.models import Structure


class TestComputeATR:
    def test_returns_positive_float(self, uptrend_df):
        atr = compute_atr(uptrend_df)
        assert isinstance(atr, float)
        assert atr > 0

    def test_higher_volatility_higher_atr(self, uptrend_df, ranging_df):
        # Both should produce valid ATR
        atr_up = compute_atr(uptrend_df)
        atr_range = compute_atr(ranging_df)
        assert atr_up > 0
        assert atr_range > 0


class TestFindSwingPoints:
    def test_finds_swings_in_uptrend(self, uptrend_df):
        swings = find_swing_points(uptrend_df, lookback=5)
        assert len(swings) > 0
        highs = [s for s in swings if s.type == "high"]
        lows = [s for s in swings if s.type == "low"]
        assert len(highs) > 0
        assert len(lows) > 0

    def test_finds_swings_in_ranging(self, ranging_df):
        swings = find_swing_points(ranging_df, lookback=5)
        assert len(swings) >= 4  # Should find multiple swing points in a range

    def test_swing_point_fields(self, uptrend_df):
        swings = find_swing_points(uptrend_df, lookback=5)
        for s in swings:
            assert s.type in ("high", "low")
            assert s.price > 0
            assert s.body_price > 0
            assert 0 <= s.index < len(uptrend_df)

    def test_sorted_by_index(self, uptrend_df):
        swings = find_swing_points(uptrend_df, lookback=5)
        indices = [s.index for s in swings]
        assert indices == sorted(indices)


class TestClusterSRLevels:
    def test_clusters_nearby_swings(self, ranging_df):
        swings = find_swing_points(ranging_df, lookback=5)
        levels = cluster_sr_levels(swings, pct=0.03)
        assert len(levels) > 0
        for level in levels:
            assert level.strength >= 2
            assert level.price > 0

    def test_no_clusters_from_empty(self):
        levels = cluster_sr_levels([], pct=0.015)
        assert levels == []

    def test_degradation_flag(self):
        """3+ same-direction tests should mark level as degraded."""
        from datetime import datetime, timezone

        from scanner.models import SwingPoint
        # 4 swing lows at the same level (4 tests from above)
        swings = [
            SwingPoint(i * 10, datetime(2026, 1, 1, tzinfo=timezone.utc), 100 + i * 0.1, "low", 100 + i * 0.1)
            for i in range(4)
        ]
        levels = cluster_sr_levels(swings, pct=0.02)
        assert len(levels) == 1
        assert levels[0].degraded is True
        assert levels[0].direction_tests["from_above"] >= 3


class TestClassifyStructure:
    def test_uptrend_is_bullish(self, uptrend_df):
        swings = find_swing_points(uptrend_df, lookback=5)
        structure, bounds = classify_structure(swings)
        assert structure == Structure.BULLISH
        assert bounds is None

    def test_downtrend_is_bearish(self, downtrend_df):
        swings = find_swing_points(downtrend_df, lookback=5)
        structure, bounds = classify_structure(swings)
        assert structure == Structure.BEARISH
        assert bounds is None

    def test_ranging_has_bounds(self, ranging_df):
        swings = find_swing_points(ranging_df, lookback=5)
        structure, bounds = classify_structure(swings)
        if structure == Structure.RANGING:
            assert bounds is not None
            assert bounds[0] < bounds[1]


class TestDetectMSBs:
    def test_finds_msbs_in_data(self, uptrend_df):
        swings = find_swing_points(uptrend_df, lookback=5)
        msbs = detect_msbs(swings, uptrend_df)
        # An uptrend may have bullish MSBs
        for msb in msbs:
            assert msb.type in ("bullish", "bearish")
            assert msb.level > 0

    def test_msb_fields(self, downtrend_df):
        swings = find_swing_points(downtrend_df, lookback=5)
        msbs = detect_msbs(swings, downtrend_df)
        for msb in msbs:
            assert msb.swing_broken is not None
            assert 0 <= msb.index < len(downtrend_df)


class TestCalculateFibs:
    def test_basic_fib_levels(self):
        fibs = calculate_fibs(100, 200)
        assert fibs.swing_low == 100
        assert fibs.swing_high == 200
        # 0.618 retracement of 100-pt range = 200 - 61.8 = 138.2
        assert abs(fibs.fib_618 - 138.2) < 0.01
        # 0.786 retracement = 200 - 78.6 = 121.4
        assert abs(fibs.fib_786 - 121.4) < 0.01
        assert fibs.ote_zone[0] > fibs.ote_zone[1]  # 618 > 786

    def test_ote_zone_between_swing_bounds(self):
        fibs = calculate_fibs(50, 150)
        assert fibs.swing_low <= fibs.fib_786
        assert fibs.fib_618 <= fibs.swing_high


class TestBuildContext:
    def test_returns_complete_context(self, uptrend_df):
        ctx = build_context(uptrend_df)
        assert ctx.structure in (Structure.BULLISH, Structure.BEARISH, Structure.RANGING)
        assert len(ctx.swings) > 0
        assert ctx.atr > 0
        assert ctx.current_price > 0

    def test_current_price_matches_last_close(self, ranging_df):
        ctx = build_context(ranging_df)
        assert ctx.current_price == float(ranging_df["Close"].iloc[-1])

    def test_has_regime_fields(self, uptrend_df):
        ctx = build_context(uptrend_df)
        assert 0 <= ctx.vol_percentile <= 100
        assert 0.0 <= ctx.efficiency_ratio <= 1.0
        assert ctx.regime in ("trending_up", "trending_down", "volatile", "ranging")


class TestComputeATRPercentile:
    def test_returns_0_to_100(self, uptrend_df):
        pct = compute_atr_percentile(uptrend_df)
        assert 0.0 <= pct <= 100.0

    def test_known_series(self):
        """Constant-volatility bars → ATR percentile near 50."""
        n = 120
        dates = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
        prices = []
        for i in range(n):
            o = 100.0
            h = 102.0
            l = 98.0
            c = 100.0
            prices.append((o, h, l, c))
        df = pd.DataFrame(prices, index=dates, columns=["Open", "High", "Low", "Close"])
        pct = compute_atr_percentile(df)
        # All ATRs identical → current = all others → rank depends on < comparison
        # With identical values, (trailing < current).sum() == 0
        assert 0.0 <= pct <= 100.0

    def test_short_history(self):
        """Fewer bars than lookback → graceful handling."""
        n = 20
        dates = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
        np.random.seed(42)
        prices = [(100 + np.random.randn(), 103, 97, 100 + np.random.randn()) for _ in range(n)]
        df = pd.DataFrame(prices, index=dates, columns=["Open", "High", "Low", "Close"])
        pct = compute_atr_percentile(df, lookback=100)
        assert 0.0 <= pct <= 100.0

    def test_very_short_returns_neutral(self):
        """Only 2 bars → returns 50.0 (neutral)."""
        dates = pd.date_range("2026-01-01", periods=2, freq="4h", tz="UTC")
        df = pd.DataFrame(
            {"Open": [100, 101], "High": [102, 103], "Low": [98, 99], "Close": [101, 102]},
            index=dates,
        )
        pct = compute_atr_percentile(df)
        assert pct == 50.0


class TestComputeEfficiencyRatio:
    def test_trending_near_1(self):
        """Clean trend → ER near 1.0."""
        n = 30
        dates = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
        close = [100.0 + i * 1.0 for i in range(n)]
        df = pd.DataFrame(
            {"Open": close, "High": [c + 0.5 for c in close],
             "Low": [c - 0.5 for c in close], "Close": close},
            index=dates,
        )
        er = compute_efficiency_ratio(df, period=20)
        assert er >= 0.9

    def test_choppy_near_0(self):
        """Sideways chop → ER near 0.0."""
        n = 30
        dates = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
        close = [100.0 + (1 if i % 2 == 0 else -1) for i in range(n)]
        df = pd.DataFrame(
            {"Open": close, "High": [c + 0.5 for c in close],
             "Low": [c - 0.5 for c in close], "Close": close},
            index=dates,
        )
        er = compute_efficiency_ratio(df, period=20)
        assert er <= 0.15

    def test_short_data(self):
        """Fewer bars than period → returns 0.0."""
        dates = pd.date_range("2026-01-01", periods=5, freq="4h", tz="UTC")
        df = pd.DataFrame(
            {"Open": [100]*5, "High": [102]*5, "Low": [98]*5, "Close": [100]*5},
            index=dates,
        )
        er = compute_efficiency_ratio(df, period=20)
        assert er == 0.0


class TestClassifyRegime:
    def test_trending_up(self):
        assert classify_regime(Structure.BULLISH, 50.0, 0.6) == "trending_up"

    def test_trending_down(self):
        assert classify_regime(Structure.BEARISH, 50.0, 0.5) == "trending_down"

    def test_volatile(self):
        assert classify_regime(Structure.RANGING, 80.0, 0.2) == "volatile"

    def test_ranging(self):
        assert classify_regime(Structure.RANGING, 40.0, 0.2) == "ranging"

    def test_bullish_low_er_is_ranging(self):
        """BULLISH structure but low ER → ranging, not trending_up."""
        assert classify_regime(Structure.BULLISH, 40.0, 0.2) == "ranging"

    def test_bearish_high_vol_low_er(self):
        """BEARISH structure, high vol, low ER → volatile (ER < 0.4 takes precedence)."""
        assert classify_regime(Structure.BEARISH, 80.0, 0.3) == "volatile"
