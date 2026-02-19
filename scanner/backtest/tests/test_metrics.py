"""Tests for metrics aggregation."""

import pytest

from scanner.backtest.resolver import TradeSetup, TradeOutcome
from scanner.backtest.metrics import compute_metrics, slice_metrics, bootstrap_expectancy, BootstrapCI


def _outcome(result="WIN", r_achieved=2.0, bars=5, mae=0.3, mfe=2.5, setup_type="C", direction="LONG", confidence=0.7, regime=""):
    setup = TradeSetup(
        coin="BTC", direction=direction, entry=100, stop=95, target=110,
        timestamp="2026-01-01", setup_type=setup_type, confidence=confidence,
        structure="BULLISH", sent_to_llm=False, source="screener", regime=regime,
    )
    return TradeOutcome(
        result=result, r_achieved=r_achieved, bars_to_exit=bars,
        exit_price=110 if result == "WIN" else 95,
        mae=mae, mfe=mfe, fees_paid=0.0, setup=setup,
    )


class TestComputeMetrics:
    """Aggregation math."""

    def test_empty_outcomes(self):
        report = compute_metrics([], "empty")
        assert report.sample_size == 0
        assert report.win_rate == 0.0
        assert report.expectancy == 0.0

    def test_all_wins(self):
        outcomes = [_outcome("WIN", 2.0) for _ in range(5)]
        report = compute_metrics(outcomes, "wins")
        assert report.sample_size == 5
        assert report.wins == 5
        assert report.losses == 0
        assert report.win_rate == 1.0
        assert report.avg_win_r == pytest.approx(2.0)
        assert report.profit_factor == float("inf")

    def test_all_losses(self):
        outcomes = [_outcome("LOSS", -1.0, mae=1.0, mfe=0.5) for _ in range(3)]
        report = compute_metrics(outcomes, "losses")
        assert report.wins == 0
        assert report.losses == 3
        assert report.win_rate == 0.0
        assert report.avg_loss_r == pytest.approx(-1.0)
        assert report.profit_factor == 0.0

    def test_mixed_outcomes(self):
        outcomes = [
            _outcome("WIN", 2.0),
            _outcome("WIN", 3.0),
            _outcome("LOSS", -1.0, mae=1.0, mfe=0.5),
            _outcome("LOSS", -1.0, mae=1.0, mfe=0.3),
        ]
        report = compute_metrics(outcomes, "mixed")
        assert report.sample_size == 4
        assert report.wins == 2
        assert report.losses == 2
        assert report.win_rate == pytest.approx(0.5)
        assert report.total_r == pytest.approx(3.0)  # 2+3-1-1
        assert report.profit_factor == pytest.approx(2.5)  # 5/2

    def test_expectancy_formula(self):
        # 2 wins at 2R, 2 losses at -1R
        # win_rate=0.5, avg_win=2.0, loss_rate=0.5, avg_loss=-1.0
        # expectancy = 0.5*2.0 + 0.5*(-1.0) = 0.5
        outcomes = [
            _outcome("WIN", 2.0),
            _outcome("WIN", 2.0),
            _outcome("LOSS", -1.0, mae=1.0, mfe=0.5),
            _outcome("LOSS", -1.0, mae=1.0, mfe=0.5),
        ]
        report = compute_metrics(outcomes, "exp")
        assert report.expectancy == pytest.approx(0.5)

    def test_avg_bars_to_exit(self):
        outcomes = [
            _outcome("WIN", 2.0, bars=10),
            _outcome("LOSS", -1.0, bars=5, mae=1.0, mfe=0.5),
        ]
        report = compute_metrics(outcomes, "bars")
        assert report.avg_bars_to_exit == pytest.approx(7.5)

    def test_expired_counted(self):
        outcomes = [
            _outcome("WIN", 2.0),
            _outcome("EXPIRED", 0.3, bars=168),
        ]
        report = compute_metrics(outcomes, "with_expired")
        assert report.expired == 1
        assert report.sample_size == 2

    def test_mae_mfe_averaged(self):
        outcomes = [
            _outcome("WIN", 2.0, mae=0.2, mfe=2.5),
            _outcome("WIN", 2.0, mae=0.4, mfe=3.0),
        ]
        report = compute_metrics(outcomes, "drawdown")
        assert report.avg_mae == pytest.approx(0.3)
        assert report.avg_mfe == pytest.approx(2.75)


class TestSliceMetrics:
    """Slicing by dimension."""

    def test_slice_by_setup_type(self):
        outcomes = [
            _outcome("WIN", 2.0, setup_type="A"),
            _outcome("WIN", 2.0, setup_type="A"),
            _outcome("LOSS", -1.0, setup_type="C", mae=1.0, mfe=0.5),
        ]
        sliced = slice_metrics(outcomes, {"setup_type": lambda o: o.setup.setup_type})
        assert "setup_type" in sliced
        labels = [r.label for r in sliced["setup_type"]]
        assert "A" in labels
        assert "C" in labels

        a_report = next(r for r in sliced["setup_type"] if r.label == "A")
        assert a_report.sample_size == 2
        assert a_report.win_rate == 1.0

    def test_slice_by_direction(self):
        outcomes = [
            _outcome("WIN", 2.0, direction="LONG"),
            _outcome("LOSS", -1.0, direction="SHORT", mae=1.0, mfe=0.5),
        ]
        sliced = slice_metrics(outcomes, {"direction": lambda o: o.setup.direction})
        assert len(sliced["direction"]) == 2


class TestBootstrapExpectancy:
    """Bootstrap confidence interval on expectancy."""

    def test_too_few_outcomes_returns_none(self):
        outcomes = [_outcome("WIN", 2.0) for _ in range(4)]
        assert bootstrap_expectancy(outcomes) is None

    def test_returns_bootstrap_ci(self):
        outcomes = [_outcome("WIN", 2.0) for _ in range(10)]
        ci = bootstrap_expectancy(outcomes, seed=42)
        assert isinstance(ci, BootstrapCI)
        assert ci.confidence == 0.95
        # All wins at 2.0R — CI should be tight around 2.0
        assert ci.ci_low == pytest.approx(2.0)
        assert ci.ci_high == pytest.approx(2.0)
        assert ci.excludes_zero is True

    def test_mixed_outcomes_ci_range(self):
        outcomes = (
            [_outcome("WIN", 2.0) for _ in range(6)]
            + [_outcome("LOSS", -1.0, mae=1.0, mfe=0.5) for _ in range(4)]
        )
        ci = bootstrap_expectancy(outcomes, seed=42)
        # Expectancy = 0.5*2 + 0.5*(-1) = 0.8 (6/10 * 2 + 4/10 * -1 = 0.8)
        assert ci.ci_low < ci.ci_high
        # With 60% wins at 2R and 40% losses at -1R, CI should be mostly positive
        assert ci.ci_low > -1.0
        assert ci.ci_high < 3.0

    def test_losing_system_ci_includes_negative(self):
        outcomes = (
            [_outcome("WIN", 2.0) for _ in range(2)]
            + [_outcome("LOSS", -1.0, mae=1.0, mfe=0.5) for _ in range(8)]
        )
        ci = bootstrap_expectancy(outcomes, seed=42)
        # 20% WR — should be clearly negative
        assert ci.ci_high <= 0.5  # upper bound shouldn't be strongly positive

    def test_deterministic_with_seed(self):
        outcomes = (
            [_outcome("WIN", 2.0) for _ in range(5)]
            + [_outcome("LOSS", -1.0, mae=1.0, mfe=0.5) for _ in range(5)]
        )
        ci1 = bootstrap_expectancy(outcomes, seed=123)
        ci2 = bootstrap_expectancy(outcomes, seed=123)
        assert ci1.ci_low == ci2.ci_low
        assert ci1.ci_high == ci2.ci_high

    def test_compute_metrics_includes_ci(self):
        outcomes = [_outcome("WIN", 2.0) for _ in range(10)]
        report = compute_metrics(outcomes, "test")
        assert report.bootstrap_ci is not None
        assert report.bootstrap_ci.excludes_zero is True

    def test_compute_metrics_no_ci_when_small(self):
        outcomes = [_outcome("WIN", 2.0) for _ in range(3)]
        report = compute_metrics(outcomes, "small")
        assert report.bootstrap_ci is None


class TestRegimeSlicer:
    """Regime slicer in DEFAULT_SLICERS."""

    def test_regime_in_default_slicers(self):
        from scanner.backtest.metrics import DEFAULT_SLICERS
        assert "regime" in DEFAULT_SLICERS

    def test_regime_appears_in_slice_output(self):
        outcomes = [
            _outcome("WIN", 2.0, regime="trending_up"),
            _outcome("LOSS", -1.0, mae=1.0, mfe=0.5, regime="ranging"),
        ]
        sliced = slice_metrics(outcomes)
        assert "regime" in sliced
        labels = {r.label for r in sliced["regime"]}
        assert "trending_up" in labels
        assert "ranging" in labels

    def test_unknown_regime_when_empty(self):
        outcomes = [_outcome("WIN", 2.0, regime="")]
        sliced = slice_metrics(outcomes)
        labels = {r.label for r in sliced["regime"]}
        assert "unknown" in labels
