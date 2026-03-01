"""Tests for grid search sweep."""

import pandas as pd
import pytest

from scanner.backtest.candles import cache_candles
from scanner.strategy.sweep import _parse_range, walk_forward_sweep


def _make_signal_row(coin, timestamp, direction="LONG", key_level=100.0, atr=5.0,
                     setup_type="C", confidence=0.7, structure="BULLISH"):
    """Build a minimal signal row dict for sweep testing."""
    return {
        "coin": coin, "timestamp": timestamp, "direction": direction,
        "key_level": key_level, "atr": atr, "setup_type": setup_type,
        "confidence": confidence, "structure": structure,
        "sent_to_llm": 0, "regime": "",
    }


def _cache_candles_for_signal(conn, coin, signal_ts, n_bars=200, base=100.0):
    """Cache 1H candles starting before signal_ts so resolver can walk forward."""
    start = pd.Timestamp(signal_ts, tz="UTC") - pd.Timedelta(hours=1)
    dates = pd.date_range(start, periods=n_bars, freq="1h", tz="UTC")
    # Candles that go up slowly — ensures most setups hit target
    data = []
    price = base
    for i in range(n_bars):
        o = price
        h = price + 3
        l = price - 1
        c = price + 0.5
        data.append({"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1000.0})
        price = c
    df = pd.DataFrame(data, index=dates)
    cache_candles(coin, "1h", df, conn)


class TestParseRange:
    def test_basic_range(self):
        values = _parse_range("1.0:0.5:3.0")
        assert values == [1.0, 1.5, 2.0, 2.5, 3.0]

    def test_single_value(self):
        assert _parse_range("2.5") == [2.5]

    def test_integer_range(self):
        values = _parse_range("1:1:3")
        assert values == [1.0, 2.0, 3.0]

    def test_fine_step(self):
        values = _parse_range("0.01:0.005:0.025")
        assert len(values) == 4
        assert values[0] == 0.01
        assert values[-1] == 0.025

    def test_single_step(self):
        values = _parse_range("1.0:1.0:1.0")
        assert values == [1.0]

    def test_large_step_gives_single(self):
        values = _parse_range("1.0:10.0:3.0")
        assert values == [1.0]


class TestWalkForwardSweep:
    """Walk-forward sweep tests."""

    def test_returns_none_on_empty_signals(self, db_conn):
        result = walk_forward_sweep([], [1.5], [3.0], db_conn)
        assert result is None

    def test_fold_boundaries_non_overlapping(self, db_conn):
        # Create signals spread across 4 weeks
        rows = []
        for day in range(28):
            ts = f"2026-01-{1+day:02d}T12:00:00+00:00"
            rows.append(_make_signal_row("BTC", ts))
            _cache_candles_for_signal(db_conn, "BTC", ts)

        result = walk_forward_sweep(
            rows, [1.5, 2.0], [3.0, 4.0], db_conn,
            n_folds=3,
        )

        assert result is not None
        assert len(result.folds) == 3

        # Verify IS/OOS don't overlap within each fold
        for fold in result.folds:
            assert fold.is_end == fold.oos_start  # contiguous
            assert fold.is_start < fold.is_end     # non-empty IS
            assert fold.oos_start < fold.oos_end   # non-empty OOS

    def test_expanding_is_window(self, db_conn):
        # IS window should expand with each fold
        rows = []
        for day in range(28):
            ts = f"2026-01-{1+day:02d}T12:00:00+00:00"
            rows.append(_make_signal_row("BTC", ts))
            _cache_candles_for_signal(db_conn, "BTC", ts)

        result = walk_forward_sweep(
            rows, [1.5], [3.0], db_conn, n_folds=3,
        )

        assert result is not None
        # All folds start at the same point (expanding)
        starts = [f.is_start for f in result.folds]
        assert len(set(starts)) == 1  # all same start

        # IS end should increase with each fold
        ends = [f.is_end for f in result.folds]
        for i in range(len(ends) - 1):
            assert ends[i] <= ends[i + 1]

    def test_wfe_calculation(self, db_conn):
        rows = []
        for day in range(28):
            ts = f"2026-01-{1+day:02d}T12:00:00+00:00"
            rows.append(_make_signal_row("BTC", ts))
            _cache_candles_for_signal(db_conn, "BTC", ts)

        result = walk_forward_sweep(
            rows, [1.5], [3.0], db_conn, n_folds=3,
        )

        assert result is not None
        for fold in result.folds:
            if fold.wfe is not None:
                # WFE = OOS_exp / IS_exp
                expected_wfe = fold.oos_metrics.expectancy / fold.is_best_metrics.expectancy
                assert fold.wfe == pytest.approx(expected_wfe, abs=0.001)

    def test_combined_oos_metrics(self, db_conn):
        rows = []
        for day in range(28):
            ts = f"2026-01-{1+day:02d}T12:00:00+00:00"
            rows.append(_make_signal_row("BTC", ts))
            _cache_candles_for_signal(db_conn, "BTC", ts)

        result = walk_forward_sweep(
            rows, [1.5, 2.0], [3.0, 4.0], db_conn, n_folds=3,
        )

        assert result is not None
        # Combined OOS should have signals from all folds
        total_oos = sum(f.oos_metrics.sample_size for f in result.folds)
        assert result.overall_oos_metrics.sample_size == total_oos

    def test_insufficient_data_returns_none(self, db_conn):
        # Single signal — can't make folds
        ts = "2026-01-01T12:00:00+00:00"
        rows = [_make_signal_row("BTC", ts)]
        _cache_candles_for_signal(db_conn, "BTC", ts)

        result = walk_forward_sweep(rows, [1.5], [3.0], db_conn, n_folds=3)
        # With only 1 timestamp, total_duration is 0 → returns None
        assert result is None

    def test_wfe_none_when_is_expectancy_negative(self, db_conn):
        # Create signals where price goes down (setups always lose)
        rows = []
        for day in range(28):
            ts = f"2026-01-{1+day:02d}T12:00:00+00:00"
            rows.append(_make_signal_row("BTC", ts, direction="LONG", key_level=200.0, atr=5.0))
            # Cache candles that go down — ensures LONG setups lose
            start = pd.Timestamp(ts, tz="UTC") - pd.Timedelta(hours=1)
            dates = pd.date_range(start, periods=200, freq="1h", tz="UTC")
            data = []
            price = 200.0
            for j in range(200):
                data.append({"Open": price, "High": price + 1, "Low": price - 3, "Close": price - 0.5, "Volume": 1000.0})
                price -= 0.5
            df = pd.DataFrame(data, index=dates)
            cache_candles("BTC", "1h", df, db_conn)

        result = walk_forward_sweep(
            rows, [1.5], [3.0], db_conn, n_folds=3,
        )

        if result:
            for fold in result.folds:
                if fold.is_best_metrics.expectancy <= 0:
                    assert fold.wfe is None

    def test_n_signals_total_correct(self, db_conn):
        rows = []
        for day in range(20):
            ts = f"2026-01-{1+day:02d}T12:00:00+00:00"
            rows.append(_make_signal_row("BTC", ts))
            _cache_candles_for_signal(db_conn, "BTC", ts)

        result = walk_forward_sweep(rows, [1.5], [3.0], db_conn, n_folds=2)
        if result:
            assert result.n_signals_total == 20
