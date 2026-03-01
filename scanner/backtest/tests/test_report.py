"""Tests for backtest report formatting — CLI and Discord embeds."""

from scanner.backtest.metrics import BootstrapCI, MetricsReport, compute_metrics, slice_metrics
from scanner.backtest.report import _ci_verdict, build_backtest_embeds
from scanner.backtest.resolver import TradeOutcome, TradeSetup


def _outcome(result="WIN", r_achieved=2.0, setup_type="C", direction="LONG"):
    setup = TradeSetup(
        coin="BTC", direction=direction, entry=100, stop=95, target=110,
        timestamp="2026-01-01", setup_type=setup_type, confidence=0.7,
        structure="BULLISH", sent_to_llm=False, source="screener", regime="",
    )
    return TradeOutcome(
        result=result, r_achieved=r_achieved, bars_to_exit=5,
        exit_price=110 if result == "WIN" else 95,
        mae=0.3, mfe=2.5, fees_paid=0.0, setup=setup,
    )


class TestBuildBacktestEmbeds:
    def test_returns_at_least_one_embed(self):
        outcomes = [_outcome()]
        overall = compute_metrics(outcomes, "Overall")
        sliced = slice_metrics(outcomes)
        embeds = build_backtest_embeds(overall, sliced, "screener", 30)
        assert len(embeds) >= 1

    def test_overall_embed_fields(self):
        outcomes = [_outcome("WIN", 2.0), _outcome("LOSS", -1.0)]
        overall = compute_metrics(outcomes, "Overall")
        sliced = slice_metrics(outcomes)
        embeds = build_backtest_embeds(overall, sliced, "screener", 7)
        overall_embed = embeds[0]
        assert "Backtest" in overall_embed["title"]
        assert "SCREENER" in overall_embed["title"]
        assert "7d" in overall_embed["title"]
        field_names = [f["name"] for f in overall_embed["fields"]]
        assert "Trades" in field_names
        assert "Win Rate" in field_names
        assert "Expectancy" in field_names

    def test_green_when_positive_expectancy(self):
        outcomes = [_outcome("WIN", 3.0)] * 5
        overall = compute_metrics(outcomes, "Overall")
        embeds = build_backtest_embeds(overall, {}, "llm", 30)
        assert embeds[0]["color"] == 0x26A69A  # green

    def test_red_when_negative_expectancy(self):
        outcomes = [_outcome("LOSS", -1.0)] * 5
        overall = compute_metrics(outcomes, "Overall")
        embeds = build_backtest_embeds(overall, {}, "screener", 30)
        assert embeds[0]["color"] == 0xEF5350  # red

    def test_setup_type_embed(self):
        outcomes = [_outcome("WIN", 2.0, "C"), _outcome("LOSS", -1.0, "A")]
        overall = compute_metrics(outcomes, "Overall")
        sliced = slice_metrics(outcomes)
        embeds = build_backtest_embeds(overall, sliced, "screener", 30)
        type_embed = next((e for e in embeds if e["title"] == "By Setup Type"), None)
        assert type_embed is not None
        assert "A" in type_embed["description"]
        assert "C" in type_embed["description"]

    def test_direction_embed(self):
        outcomes = [_outcome("WIN", 2.0, direction="LONG"), _outcome("LOSS", -1.0, direction="SHORT")]
        overall = compute_metrics(outcomes, "Overall")
        sliced = slice_metrics(outcomes)
        embeds = build_backtest_embeds(overall, sliced, "screener", 30)
        dir_embed = next((e for e in embeds if e["title"] == "By Direction"), None)
        assert dir_embed is not None
        assert "LONG" in dir_embed["description"]
        assert "SHORT" in dir_embed["description"]

    def test_no_type_embed_when_empty(self):
        outcomes = [_outcome()]
        overall = compute_metrics(outcomes, "Overall")
        embeds = build_backtest_embeds(overall, {}, "screener", 30)
        assert len(embeds) == 1


class TestCiVerdict:
    def test_edge(self):
        r = MetricsReport(
            label="x", sample_size=10, wins=8, losses=2, expired=0,
            win_rate=0.8, avg_r=1.0, expectancy=0.5, profit_factor=4.0,
            total_r=10.0, avg_bars_to_exit=5.0, avg_mae=0.2, avg_mfe=1.5,
            avg_win_r=1.5, avg_loss_r=-0.5,
            bootstrap_ci=BootstrapCI(ci_low=0.1, ci_high=0.9, confidence=0.95, excludes_zero=True),
        )
        assert _ci_verdict(r) == "edge"

    def test_negative(self):
        r = MetricsReport(
            label="x", sample_size=10, wins=2, losses=8, expired=0,
            win_rate=0.2, avg_r=-0.5, expectancy=-0.3, profit_factor=0.4,
            total_r=-5.0, avg_bars_to_exit=5.0, avg_mae=0.8, avg_mfe=0.3,
            avg_win_r=1.0, avg_loss_r=-0.8,
            bootstrap_ci=BootstrapCI(ci_low=-0.9, ci_high=-0.1, confidence=0.95, excludes_zero=True),
        )
        assert _ci_verdict(r) == "negative"

    def test_inconclusive(self):
        r = MetricsReport(
            label="x", sample_size=10, wins=5, losses=5, expired=0,
            win_rate=0.5, avg_r=0.0, expectancy=0.0, profit_factor=1.0,
            total_r=0.0, avg_bars_to_exit=5.0, avg_mae=0.5, avg_mfe=0.5,
            avg_win_r=1.0, avg_loss_r=-1.0,
            bootstrap_ci=BootstrapCI(ci_low=-0.3, ci_high=0.3, confidence=0.95, excludes_zero=False),
        )
        assert _ci_verdict(r) == "inconclusive"

    def test_small_sample(self):
        r = MetricsReport(
            label="x", sample_size=3, wins=2, losses=1, expired=0,
            win_rate=0.67, avg_r=0.5, expectancy=0.3, profit_factor=2.0,
            total_r=1.5, avg_bars_to_exit=5.0, avg_mae=0.3, avg_mfe=1.0,
            avg_win_r=1.0, avg_loss_r=-0.5,
        )
        assert _ci_verdict(r) == "n/a"
