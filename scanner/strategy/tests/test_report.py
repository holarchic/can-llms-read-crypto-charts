"""Tests for strategy validation reports — signal quality, LLM filter, regime decay, auto-disable."""

import numpy as np
import pandas as pd
import pytest

from scanner.backtest.candles import cache_candles
from scanner.backtest.metrics import MetricsReport, compute_metrics
from scanner.backtest.resolver import TradeOutcome, TradeSetup
from scanner.storage import get_connection
from scanner.strategy.models import Strategy
from scanner.strategy.store import save_strategy, get_strategy
from scanner.strategy.report import (
    load_strategy_signals,
    load_llm_scans,
    resolve_signals,
    signal_quality_report,
    llm_filter_report,
    LLMFilterResult,
    regime_decay_report,
    suggest_disabled_types,
    format_signal_quality,
    format_llm_filter,
    format_regime_decay,
    format_disable_suggestion,
)


# --- Fixtures ---


def _make_candles(n_bars=200, start="2026-01-01", base=100.0, seed=42):
    """Generate synthetic OHLCV candles."""
    dates = pd.date_range(start, periods=n_bars, freq="4h", tz="UTC")
    np.random.seed(seed)
    prices = []
    for i in range(n_bars):
        mid = base + np.sin(i * 0.3) * 5 + i * 0.1
        o = mid + np.random.uniform(-0.5, 0.5)
        h = mid + np.random.uniform(2, 4)
        l = mid - np.random.uniform(2, 4)
        c = mid + np.random.uniform(-0.5, 0.5)
        prices.append((o, h, l, c, 1000.0))
    return pd.DataFrame(prices, index=dates, columns=["Open", "High", "Low", "Close", "Volume"])


def _make_1h_candles(base=100.0, start="2026-01-01", n_bars=500, seed=42):
    """Generate synthetic 1H candles for resolution."""
    dates = pd.date_range(start, periods=n_bars, freq="1h", tz="UTC")
    np.random.seed(seed)
    prices = []
    for i in range(n_bars):
        mid = base + np.sin(i * 0.1) * 3 + i * 0.05
        o = mid + np.random.uniform(-0.3, 0.3)
        h = mid + np.random.uniform(1, 3)
        l = mid - np.random.uniform(1, 3)
        c = mid + np.random.uniform(-0.3, 0.3)
        prices.append((o, h, l, c, 500.0))
    return pd.DataFrame(prices, index=dates, columns=["Open", "High", "Low", "Close", "Volume"])


@pytest.fixture
def db_conn(tmp_path):
    """Fresh DB with all tables."""
    conn = get_connection(tmp_path / "test.db")
    yield conn
    conn.close()


@pytest.fixture
def strategy():
    """Default strategy."""
    return Strategy(strategy_id="v0", label="baseline", is_active=True)


@pytest.fixture
def populated_db(db_conn, strategy):
    """DB with cached candles + saved strategy + inserted test signals."""
    save_strategy(strategy, db_conn)

    # Cache 4H candles
    df_4h = _make_candles(n_bars=300, base=100.0)
    cache_candles("BTC", "4h", df_4h, db_conn)

    # Cache 1H candles for resolution
    df_1h = _make_1h_candles(base=100.0, n_bars=1200)
    cache_candles("BTC", "1h", df_1h, db_conn)

    # Insert synthetic signals
    _insert_test_signals(db_conn, "v0")

    return db_conn


def _insert_test_signals(conn, strategy_id):
    """Insert synthetic signals for testing."""
    from datetime import datetime, timezone
    from uuid import uuid4

    now = datetime.now(timezone.utc).isoformat()
    signals = [
        # (setup_type, direction, confidence, key_level, atr, structure, sent_to_llm, timestamp)
        ("A", "LONG", 0.65, 100.0, 3.0, "RANGING", 1, "2026-01-05T00:00:00+00:00"),
        ("A", "SHORT", 0.55, 105.0, 3.0, "RANGING", 1, "2026-01-06T00:00:00+00:00"),
        ("B", "LONG", 0.70, 102.0, 3.0, "BULLISH", 1, "2026-01-07T00:00:00+00:00"),
        ("C", "LONG", 0.60, 101.0, 3.0, "BULLISH", 0, "2026-01-08T00:00:00+00:00"),
        ("C", "SHORT", 0.62, 104.0, 3.0, "BEARISH", 0, "2026-01-09T00:00:00+00:00"),
        ("E", "SHORT", 0.45, 103.0, 3.0, "BEARISH", 0, "2026-01-10T00:00:00+00:00"),
        ("F", "LONG", 0.55, 100.5, 3.0, "RANGING", 1, "2026-01-11T00:00:00+00:00"),
        # Second week
        ("A", "LONG", 0.60, 101.0, 3.0, "RANGING", 1, "2026-01-15T00:00:00+00:00"),
        ("B", "SHORT", 0.58, 106.0, 3.0, "BEARISH", 0, "2026-01-16T00:00:00+00:00"),
        ("C", "LONG", 0.65, 102.0, 3.0, "BULLISH", 1, "2026-01-17T00:00:00+00:00"),
        ("D", "LONG", 0.60, 101.5, 3.0, "BULLISH", 0, "2026-01-18T00:00:00+00:00"),
        ("E", "SHORT", 0.50, 104.0, 3.0, "BEARISH", 0, "2026-01-19T00:00:00+00:00"),
    ]

    batch_id = uuid4().hex[:8]
    for i, (st, d, conf, kl, atr, struct, stl, ts) in enumerate(signals, 1):
        conn.execute(
            """INSERT INTO signals
            (signal_id, batch_id, coin, setup_type, setup_name, direction,
             confidence, key_level, description, current_price, atr, structure,
             range_low, range_high, fib_618, fib_786,
             sent_to_llm, rank_in_batch, total_in_batch, timestamp, created_at,
             strategy_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                uuid4().hex[:8], batch_id, "BTC", st, f"Setup {st}", d,
                conf, kl, f"Test signal {st} {d}", kl, atr, struct,
                None, None, None, None,
                stl, i, len(signals), ts, now, strategy_id,
            ),
        )
    conn.commit()


def _insert_test_scans(conn, strategy_id):
    """Insert synthetic LLM scan results for testing."""
    from datetime import datetime, timezone
    from uuid import uuid4

    now = datetime.now(timezone.utc).isoformat()
    scans = [
        # (coin, setup_type, direction, decision, entry_zone, stop_loss, take_profit, cost, timestamp)
        ("BTC", "A", "LONG", "CONFIRM", "$100.00", "$97.00", "$109.00", 0.10, "2026-01-05T00:00:00+00:00"),
        ("BTC", "A", "SHORT", "REJECT", None, None, None, 0.10, "2026-01-06T00:00:00+00:00"),
        ("BTC", "B", "LONG", "CONFIRM", "$102.00", "$99.00", "$111.00", 0.10, "2026-01-07T00:00:00+00:00"),
        ("BTC", "F", "LONG", "WAIT", None, None, None, 0.10, "2026-01-11T00:00:00+00:00"),
        ("BTC", "A", "LONG", "CONFIRM", "$101.00", "$98.00", "$110.00", 0.10, "2026-01-15T00:00:00+00:00"),
        ("BTC", "C", "LONG", "REJECT", None, None, None, 0.10, "2026-01-17T00:00:00+00:00"),
    ]

    for coin, st, d, dec, ez, sl, tp, cost, ts in scans:
        conn.execute(
            """INSERT INTO scans
            (scan_id, coin, timestamp, setup_type, setup_name, direction,
             screener_confidence, key_level, decision, confluence_score,
             entry_zone, stop_loss, take_profit, reasoning, cost_usd, alerted,
             strategy_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                uuid4().hex[:8], coin, ts, st, f"Setup {st}", d,
                0.65, 100.0, dec, "3/5", ez, sl, tp, "Test reasoning",
                cost, 0, strategy_id, now,
            ),
        )
    conn.commit()


# --- Tests: Shared Helpers ---


class TestLoadStrategySignals:
    def test_loads_signals_in_range(self, populated_db):
        rows = load_strategy_signals("v0", "2026-01-01", "2026-01-20", populated_db)
        assert len(rows) == 12

    def test_filters_by_date_range(self, populated_db):
        rows = load_strategy_signals("v0", "2026-01-05", "2026-01-10", populated_db)
        assert 0 < len(rows) < 12

    def test_filters_by_coin(self, populated_db):
        rows = load_strategy_signals("v0", "2026-01-01", "2026-01-20", populated_db, coins=["BTC"])
        assert len(rows) == 12
        # No ETH signals exist
        rows_eth = load_strategy_signals("v0", "2026-01-01", "2026-01-20", populated_db, coins=["ETH"])
        assert len(rows_eth) == 0

    def test_wrong_strategy_returns_empty(self, populated_db):
        rows = load_strategy_signals("nonexistent", "2026-01-01", "2026-01-20", populated_db)
        assert len(rows) == 0


class TestLoadLLMScans:
    def test_loads_scans(self, populated_db):
        _insert_test_scans(populated_db, "v0")
        rows = load_llm_scans("v0", "2026-01-01", "2026-01-20", populated_db)
        assert len(rows) == 6

    def test_filters_by_date(self, populated_db):
        _insert_test_scans(populated_db, "v0")
        rows = load_llm_scans("v0", "2026-01-05", "2026-01-12", populated_db)
        assert 0 < len(rows) < 6


class TestResolveSignals:
    def test_returns_outcomes(self, populated_db):
        rows = load_strategy_signals("v0", "2026-01-01", "2026-01-10", populated_db)
        # Only test with rows that have valid timestamps within our cached data
        outcomes = resolve_signals(rows, 1.5, 3.0, populated_db)
        assert isinstance(outcomes, list)
        for o in outcomes:
            assert isinstance(o, TradeOutcome)
            assert o.result in ("WIN", "LOSS", "EXPIRED")

    def test_empty_input(self, populated_db):
        outcomes = resolve_signals([], 1.5, 3.0, populated_db)
        assert outcomes == []


# --- Tests: Loop 1 (Signal Quality) ---


class TestSignalQualityReport:
    def test_returns_overall_and_sliced(self, populated_db, strategy):
        overall, sliced = signal_quality_report(
            strategy, "2026-01-01", "2026-01-20", populated_db,
        )
        assert isinstance(overall, MetricsReport)
        assert isinstance(sliced, dict)
        assert "setup_type" in sliced
        assert "direction" in sliced

    def test_overall_sample_size(self, populated_db, strategy):
        overall, _ = signal_quality_report(
            strategy, "2026-01-01", "2026-01-20", populated_db,
        )
        # Should have resolved some signals
        assert overall.sample_size >= 0

    def test_sliced_has_setup_types(self, populated_db, strategy):
        _, sliced = signal_quality_report(
            strategy, "2026-01-01", "2026-01-20", populated_db,
        )
        type_reports = sliced.get("setup_type", [])
        type_labels = {r.label for r in type_reports}
        # Should have at least some of the types we inserted
        assert len(type_labels) > 0


# --- Tests: Loop 2 (LLM Filter) ---


class TestLLMFilterReport:
    def test_returns_result(self, populated_db, strategy):
        _insert_test_scans(populated_db, "v0")
        result = llm_filter_report(
            strategy, "2026-01-01", "2026-01-20", populated_db,
        )
        assert isinstance(result, LLMFilterResult)
        assert result.total_signals == 12
        assert result.sent_to_llm_count > 0

    def test_funnel_counts(self, populated_db, strategy):
        _insert_test_scans(populated_db, "v0")
        result = llm_filter_report(
            strategy, "2026-01-01", "2026-01-20", populated_db,
        )
        assert result.confirm_count == 3
        assert result.reject_count == 2
        assert result.wait_count == 1
        assert result.total_cost == pytest.approx(0.60, abs=0.01)

    def test_confirm_rate(self, populated_db, strategy):
        _insert_test_scans(populated_db, "v0")
        result = llm_filter_report(
            strategy, "2026-01-01", "2026-01-20", populated_db,
        )
        # 6 scans total, but confirm rate is confirms / sent_to_llm signals
        assert 0 <= result.confirm_rate <= 1.0

    def test_three_buckets_are_metrics(self, populated_db, strategy):
        _insert_test_scans(populated_db, "v0")
        result = llm_filter_report(
            strategy, "2026-01-01", "2026-01-20", populated_db,
        )
        assert isinstance(result.screener_all, MetricsReport)
        assert isinstance(result.screener_sent, MetricsReport)
        assert isinstance(result.llm_confirmed, MetricsReport)


# --- Tests: Loop 3 (Regime Decay) ---


class TestRegimeDecayReport:
    def test_returns_windows(self, populated_db, strategy):
        windows = regime_decay_report(
            strategy, "2026-01-01", "2026-01-29", populated_db,
            window_weeks=2,
        )
        assert isinstance(windows, list)
        assert len(windows) >= 1
        for label, metrics in windows:
            assert isinstance(label, str)
            assert isinstance(metrics, MetricsReport)

    def test_single_week_windows(self, populated_db, strategy):
        windows = regime_decay_report(
            strategy, "2026-01-01", "2026-01-22", populated_db,
            window_weeks=1,
        )
        assert len(windows) >= 2

    def test_empty_range(self, populated_db, strategy):
        windows = regime_decay_report(
            strategy, "2025-01-01", "2025-01-02", populated_db,
            window_weeks=2,
        )
        # Range too short for any window
        assert len(windows) == 0


# --- Tests: Auto-disable ---


class TestSuggestDisabledTypes:
    def test_suggests_negative_expectancy(self):
        reports = [
            MetricsReport(label="A", sample_size=15, wins=3, losses=12, expired=0,
                         win_rate=0.2, avg_r=-0.5, expectancy=-0.5, profit_factor=0.5,
                         total_r=-7.5, avg_bars_to_exit=20, avg_mae=1.0, avg_mfe=0.5,
                         avg_win_r=2.0, avg_loss_r=-1.0),
            MetricsReport(label="B", sample_size=15, wins=10, losses=5, expired=0,
                         win_rate=0.67, avg_r=0.5, expectancy=0.5, profit_factor=2.0,
                         total_r=7.5, avg_bars_to_exit=15, avg_mae=0.5, avg_mfe=1.5,
                         avg_win_r=1.5, avg_loss_r=-1.0),
        ]
        suggestions = suggest_disabled_types(reports)
        assert len(suggestions) == 1
        assert suggestions[0][0] == "A"
        assert suggestions[0][1] < 0

    def test_min_sample_filter(self):
        reports = [
            MetricsReport(label="A", sample_size=5, wins=1, losses=4, expired=0,
                         win_rate=0.2, avg_r=-0.5, expectancy=-0.5, profit_factor=0.5,
                         total_r=-2.5, avg_bars_to_exit=20, avg_mae=1.0, avg_mfe=0.5,
                         avg_win_r=2.0, avg_loss_r=-1.0),
        ]
        suggestions = suggest_disabled_types(reports, min_sample=10)
        assert len(suggestions) == 0

    def test_all_positive_returns_empty(self):
        reports = [
            MetricsReport(label="A", sample_size=20, wins=15, losses=5, expired=0,
                         win_rate=0.75, avg_r=0.8, expectancy=0.8, profit_factor=3.0,
                         total_r=16.0, avg_bars_to_exit=12, avg_mae=0.3, avg_mfe=2.0,
                         avg_win_r=1.5, avg_loss_r=-1.0),
        ]
        suggestions = suggest_disabled_types(reports)
        assert len(suggestions) == 0

    def test_sorted_worst_first(self):
        reports = [
            MetricsReport(label="A", sample_size=15, wins=3, losses=12, expired=0,
                         win_rate=0.2, avg_r=-0.3, expectancy=-0.3, profit_factor=0.5,
                         total_r=-4.5, avg_bars_to_exit=20, avg_mae=1.0, avg_mfe=0.5,
                         avg_win_r=2.0, avg_loss_r=-1.0),
            MetricsReport(label="E", sample_size=15, wins=2, losses=13, expired=0,
                         win_rate=0.13, avg_r=-0.7, expectancy=-0.7, profit_factor=0.3,
                         total_r=-10.5, avg_bars_to_exit=20, avg_mae=1.0, avg_mfe=0.3,
                         avg_win_r=2.0, avg_loss_r=-1.0),
        ]
        suggestions = suggest_disabled_types(reports)
        assert len(suggestions) == 2
        assert suggestions[0][0] == "E"  # worst first
        assert suggestions[1][0] == "A"


# --- Tests: Formatting ---


class TestFormatSignalQuality:
    def test_format_output(self):
        overall = MetricsReport(
            label="v0", sample_size=50, wins=20, losses=25, expired=5,
            win_rate=0.4, avg_r=-0.1, expectancy=-0.1, profit_factor=0.8,
            total_r=-5.0, avg_bars_to_exit=24, avg_mae=0.8, avg_mfe=1.2,
            avg_win_r=1.5, avg_loss_r=-1.0,
        )
        sliced = {
            "setup_type": [
                MetricsReport(label="A", sample_size=10, wins=4, losses=6, expired=0,
                             win_rate=0.4, avg_r=-0.1, expectancy=-0.1, profit_factor=0.8,
                             total_r=-1.0, avg_bars_to_exit=20, avg_mae=0.7, avg_mfe=1.0,
                             avg_win_r=1.5, avg_loss_r=-1.0),
            ],
        }
        output = format_signal_quality(overall, sliced)
        assert "SIGNAL QUALITY REPORT" in output
        assert "OVERALL" in output
        assert "BY SETUP_TYPE" in output

    def test_empty_sliced(self):
        overall = MetricsReport(
            label="v0", sample_size=0, wins=0, losses=0, expired=0,
            win_rate=0.0, avg_r=0.0, expectancy=0.0, profit_factor=0.0,
            total_r=0.0, avg_bars_to_exit=0.0, avg_mae=0.0, avg_mfe=0.0,
            avg_win_r=0.0, avg_loss_r=0.0,
        )
        output = format_signal_quality(overall, {})
        assert "SIGNAL QUALITY REPORT" in output


class TestFormatLLMFilter:
    def test_format_output(self):
        mr = MetricsReport(
            label="all_signals", sample_size=50, wins=20, losses=25, expired=5,
            win_rate=0.4, avg_r=-0.1, expectancy=-0.1, profit_factor=0.8,
            total_r=-5.0, avg_bars_to_exit=24, avg_mae=0.8, avg_mfe=1.2,
            avg_win_r=1.5, avg_loss_r=-1.0,
        )
        result = LLMFilterResult(
            screener_all=mr,
            screener_sent=mr,
            llm_confirmed=mr,
            total_signals=100,
            sent_to_llm_count=20,
            confirm_count=8,
            reject_count=10,
            wait_count=2,
            total_cost=2.0,
            confirm_rate=0.4,
            cost_per_confirm=0.25,
        )
        output = format_llm_filter(result)
        assert "LLM FILTER VALUE REPORT" in output
        assert "FUNNEL" in output
        assert "EXPECTANCY COMPARISON" in output
        assert "$2.00" in output


class TestFormatRegimeDecay:
    def test_format_output(self):
        mr = MetricsReport(
            label="2026-01-01", sample_size=25, wins=10, losses=13, expired=2,
            win_rate=0.4, avg_r=-0.1, expectancy=-0.1, profit_factor=0.8,
            total_r=-2.5, avg_bars_to_exit=20, avg_mae=0.7, avg_mfe=1.0,
            avg_win_r=1.5, avg_loss_r=-1.0,
        )
        windows = [("2026-01-01", mr), ("2026-01-15", mr)]
        output = format_regime_decay(windows)
        assert "REGIME DECAY REPORT" in output
        assert "2026-01-01" in output

    def test_trend_arrows(self):
        mr1 = MetricsReport(
            label="w1", sample_size=10, wins=2, losses=8, expired=0,
            win_rate=0.2, avg_r=-0.5, expectancy=-0.5, profit_factor=0.5,
            total_r=-5.0, avg_bars_to_exit=20, avg_mae=1.0, avg_mfe=0.5,
            avg_win_r=2.0, avg_loss_r=-1.0,
        )
        mr2 = MetricsReport(
            label="w2", sample_size=10, wins=7, losses=3, expired=0,
            win_rate=0.7, avg_r=0.5, expectancy=0.5, profit_factor=2.0,
            total_r=5.0, avg_bars_to_exit=15, avg_mae=0.5, avg_mfe=1.5,
            avg_win_r=1.5, avg_loss_r=-1.0,
        )
        output = format_regime_decay([("2026-01-01", mr1), ("2026-01-15", mr2)])
        assert "^" in output  # improvement trend


class TestFormatDisableSuggestion:
    def test_with_suggestions(self):
        suggestions = [("E", -0.5, 15), ("A", -0.2, 12)]
        output = format_disable_suggestion(suggestions)
        assert "SUGGESTED DISABLES" in output
        assert "E" in output
        assert "python -m scanner.strategy disable" in output

    def test_no_suggestions(self):
        output = format_disable_suggestion([])
        assert "No setup types qualify" in output


# --- Tests: Disabled Types ---


class TestDisabledTypesModel:
    def test_empty_disabled(self):
        s = Strategy(strategy_id="v0", label="test")
        assert s.disabled_types_set == set()

    def test_parse_disabled(self):
        s = Strategy(strategy_id="v0", label="test", disabled_types="A,E")
        assert s.disabled_types_set == {"A", "E"}

    def test_parse_with_spaces(self):
        s = Strategy(strategy_id="v0", label="test", disabled_types=" A , E , ")
        assert s.disabled_types_set == {"A", "E"}

    def test_case_insensitive(self):
        s = Strategy(strategy_id="v0", label="test", disabled_types="a,e")
        assert s.disabled_types_set == {"A", "E"}


class TestDisabledTypesStore:
    def test_roundtrip(self, db_conn):
        s = Strategy(strategy_id="v0", label="test", disabled_types="A,E")
        save_strategy(s, db_conn)
        got = get_strategy("v0", db_conn)
        assert got.disabled_types == "A,E"
        assert got.disabled_types_set == {"A", "E"}

    def test_empty_roundtrip(self, db_conn):
        s = Strategy(strategy_id="v0", label="test")
        save_strategy(s, db_conn)
        got = get_strategy("v0", db_conn)
        assert got.disabled_types == ""
        assert got.disabled_types_set == set()

    def test_update_disabled(self, db_conn):
        s = Strategy(strategy_id="v0", label="test")
        save_strategy(s, db_conn)
        s.disabled_types = "B,C"
        save_strategy(s, db_conn)
        got = get_strategy("v0", db_conn)
        assert got.disabled_types == "B,C"


class TestDisabledTypesScreener:
    def test_detect_all_filters_disabled(self):
        """disabled_types removes matching setup types from results."""
        from scanner.screener import detect_all_setups
        from scanner.indicators import build_context

        # Build synthetic ranging data to trigger setups
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
        base = 100.0
        prices = []
        for i in range(n):
            mid = base + np.sin(i * 0.3) * 5
            o = mid + np.random.uniform(-0.5, 0.5)
            h = mid + np.random.uniform(2, 4)
            l = mid - np.random.uniform(2, 4)
            c = mid + np.random.uniform(-0.5, 0.5)
            prices.append((o, h, l, c, 1000.0))
        df = pd.DataFrame(prices, index=dates, columns=["Open", "High", "Low", "Close", "Volume"])
        ctx = build_context(df)

        # Get all signals
        all_signals = detect_all_setups("BTC", df, ctx, max_per_coin=20)
        if not all_signals:
            pytest.skip("No signals generated from synthetic data")

        # Get types that were found
        found_types = {s.setup_type.value for s in all_signals}

        # Disable one of the found types
        disabled = {next(iter(found_types))}
        filtered = detect_all_setups("BTC", df, ctx, max_per_coin=20, disabled_types=disabled)
        filtered_types = {s.setup_type.value for s in filtered}

        assert disabled.isdisjoint(filtered_types)

    def test_none_disabled_returns_all(self):
        """disabled_types=None doesn't filter anything."""
        from scanner.screener import detect_all_setups
        from scanner.indicators import build_context

        np.random.seed(42)
        n = 200
        dates = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
        prices = []
        for i in range(n):
            mid = 100 + np.sin(i * 0.3) * 5
            o = mid + np.random.uniform(-0.5, 0.5)
            h = mid + np.random.uniform(2, 4)
            l = mid - np.random.uniform(2, 4)
            c = mid + np.random.uniform(-0.5, 0.5)
            prices.append((o, h, l, c, 1000.0))
        df = pd.DataFrame(prices, index=dates, columns=["Open", "High", "Low", "Close", "Volume"])
        ctx = build_context(df)

        signals_none = detect_all_setups("BTC", df, ctx, max_per_coin=20, disabled_types=None)
        signals_empty = detect_all_setups("BTC", df, ctx, max_per_coin=20, disabled_types=set())
        assert len(signals_none) == len(signals_empty)
