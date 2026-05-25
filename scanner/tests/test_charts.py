"""Tests for chart generation."""


import numpy as np
import pandas as pd
import pytest

from scanner.charts import cleanup_charts, generate_chart, generate_charts_for_signal
from scanner.indicators import build_context
from scanner.ma_indicators import compute_stoch_rsi
from scanner.models import Direction, SetupSignal, SetupType


@pytest.fixture
def tmp_charts(tmp_path):
    return tmp_path / "charts"


def _osc_df(n: int, start: str = "2026-01-01") -> pd.DataFrame:
    """Oscillatory (trend + sine swing) OHLCV df of length ``n``.

    The swing keeps RSI from flat-lining so Stoch RSI is not all-NaN once enough bars
    exist (a monotonic series collapses RSI → flat → Stoch undefined).
    """
    rng = np.random.default_rng(7)
    cols: dict[str, list[float]] = {"Open": [], "High": [], "Low": [], "Close": []}
    for i in range(n):
        mid = 100 + i * 0.4 + np.sin(i * 0.35) * 5
        cols["Open"].append(mid + rng.uniform(-0.5, 0.5))
        cols["High"].append(mid + rng.uniform(1, 3))
        cols["Low"].append(mid - rng.uniform(1, 3))
        cols["Close"].append(mid + rng.uniform(-0.6, 0.6))
    idx = pd.date_range(start, periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({**cols, "Volume": [1000.0] * n}, index=idx)


@pytest.fixture
def long_df():
    """≥320-bar oscillatory df so the longer set + envelope have data and Stoch ≠ all-NaN."""
    df = _osc_df(320)
    # Anti-vacuity: the Stoch sub-pane test below is only meaningful if Stoch actually emits.
    assert compute_stoch_rsi(df["Close"]).k.notna().any()
    return df


@pytest.fixture
def signal(long_df):
    """One signal reused across renders — its context is independent of the rendered df."""
    return SetupSignal(
        coin="TEST",
        setup_type=SetupType.A,
        direction=Direction.LONG,
        confidence=0.6,
        key_level=100.0,
        context=build_context(long_df),
        description="Test signal",
    )


@pytest.fixture
def capture_plot(monkeypatch):
    """Record the kwargs passed to mpf.plot so tests assert render STRUCTURE, not PNG bytes."""
    recorded = {}

    def fake_plot(df, **kwargs):
        recorded.clear()
        recorded.update(kwargs)

    monkeypatch.setattr("scanner.charts.mpf.plot", fake_plot)
    return recorded


class TestGenerateChart:
    def test_creates_png(self, ranging_df, tmp_charts):
        ctx = build_context(ranging_df)
        signal = SetupSignal(
            coin="TEST",
            setup_type=SetupType.A,
            direction=Direction.LONG,
            confidence=0.6,
            key_level=100.0,
            context=ctx,
            description="Test signal",
        )
        path = generate_chart(ranging_df, "TEST", "4H", signal, output_dir=tmp_charts)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 1000  # Non-trivial file

    def test_filename_contains_coin_and_tf(self, ranging_df, tmp_charts):
        ctx = build_context(ranging_df)
        signal = SetupSignal(
            coin="ETH",
            setup_type=SetupType.C,
            direction=Direction.SHORT,
            confidence=0.5,
            key_level=2000.0,
            context=ctx,
            description="Test",
        )
        path = generate_chart(ranging_df, "ETH", "1H", signal, output_dir=tmp_charts)
        assert "ETH" in path.name
        assert "1H" in path.name


class TestCleanupCharts:
    def test_removes_pngs(self, tmp_charts):
        tmp_charts.mkdir(parents=True)
        (tmp_charts / "test1.png").write_bytes(b"fake")
        (tmp_charts / "test2.png").write_bytes(b"fake")
        count = cleanup_charts(tmp_charts)
        assert count == 2
        assert len(list(tmp_charts.glob("*.png"))) == 0

    def test_returns_zero_for_empty(self, tmp_path):
        count = cleanup_charts(tmp_path / "nonexistent")
        assert count == 0


def _panel(addplots, panel):
    return [a for a in addplots if a["panel"] == panel]


class TestEnrichmentStructure:
    """Render structure via mpf.plot kwarg capture — proves what is drawn, not PNG size."""

    def test_default_path_adds_no_enrichment_kwargs(self, long_df, signal, capture_plot, tmp_charts):
        # enrich=False (default) preserves the v1 blind chart: no overlays, no sub-pane.
        generate_chart(long_df, "TEST", "4H", signal, output_dir=tmp_charts)
        assert "addplot" not in capture_plot
        assert "fill_between" not in capture_plot
        assert "panel_ratios" not in capture_plot

    def test_enriched_full_overlays_and_placement(self, long_df, signal, capture_plot, tmp_charts):
        generate_chart(long_df, "TEST", "4H", signal, output_dir=tmp_charts, enrich=True)
        addplots = capture_plot["addplot"]
        assert len(addplots) == 9  # band 3 + longer set 3 + Stoch k/d/midline 3
        assert "fill_between" in capture_plot
        assert len(capture_plot["panel_ratios"]) == 3
        # Placement: price overlays on panel 0, Stoch on panel 2 with a 0-100 axis.
        assert len(_panel(addplots, 0)) == 6
        stoch = _panel(addplots, 2)
        assert len(stoch) == 3
        assert any(a["ylim"] == (0, 100) for a in stoch)

    def test_enriched_short_df_skips_longer_set(self, signal, capture_plot, tmp_charts):
        # 80 bars: SMA100/EMA200/SMA300 all-NaN → skipped; full band (3) + envelope render.
        df = _osc_df(80)
        generate_chart(df, "TEST", "4H", signal, output_dir=tmp_charts, enrich=True)
        addplots = capture_plot["addplot"]
        assert len(_panel(addplots, 0)) == 3  # band only, no longer set
        assert "fill_between" in capture_plot  # both band edges present → envelope
        assert len(_panel(addplots, 2)) == 3  # Stoch pane present
        assert len(capture_plot["panel_ratios"]) == 3

    def test_enriched_partial_warmup(self, signal, capture_plot, tmp_charts):
        # 30 bars: fast(13)+mid(25) emit, slow(36) all-NaN → no envelope, no longer set.
        df = _osc_df(30)
        generate_chart(df, "TEST", "4H", signal, output_dir=tmp_charts, enrich=True)
        addplots = capture_plot["addplot"]
        assert len(_panel(addplots, 0)) == 2  # fast + mid band only
        assert "fill_between" not in capture_plot  # envelope needs both band edges
        # Stoch pane is data-driven: present iff Stoch RSI actually emits.
        stoch_present = compute_stoch_rsi(df["Close"]).k.notna().any()
        assert bool(_panel(addplots, 2)) == bool(stoch_present)
        assert ("panel_ratios" in capture_plot) == bool(stoch_present)

    def test_enriched_full_degradation(self, signal, capture_plot, tmp_charts):
        # 10 bars: every overlay all-NaN → plain candles + volume, no enrichment kwargs.
        df = _osc_df(10)
        generate_chart(df, "TEST", "4H", signal, output_dir=tmp_charts, enrich=True)
        assert "addplot" not in capture_plot
        assert "fill_between" not in capture_plot
        assert "panel_ratios" not in capture_plot

    def test_htf_ema_aligned_adds_one_panel0_line(self, long_df, signal, capture_plot, tmp_charts):
        generate_chart(long_df, "TEST", "1H", signal, output_dir=tmp_charts, enrich=True)
        baseline_n = len(_panel(capture_plot["addplot"], 0))  # public-path baseline (no htf)
        htf = pd.Series(long_df["Close"].to_numpy(), index=long_df.index)  # aligned to df.index
        generate_chart(long_df, "TEST", "1H", signal, output_dir=tmp_charts, enrich=True, htf_ema=htf)
        assert len(_panel(capture_plot["addplot"], 0)) == baseline_n + 1

    def test_htf_ema_misaligned_is_dropped(self, long_df, signal, capture_plot, tmp_charts):
        generate_chart(long_df, "TEST", "1H", signal, output_dir=tmp_charts, enrich=True)
        baseline_n = len(_panel(capture_plot["addplot"], 0))  # public-path baseline (no htf)
        # Same length, fully disjoint index → reindex(df.index) is all-NaN → no positional line.
        disjoint = long_df.index + pd.Timedelta(days=10_000)
        htf = pd.Series(long_df["Close"].to_numpy(), index=disjoint)
        generate_chart(long_df, "TEST", "1H", signal, output_dir=tmp_charts, enrich=True, htf_ema=htf)
        assert len(_panel(capture_plot["addplot"], 0)) == baseline_n  # htf line absent

    def test_htf_ema_duplicate_index_is_skipped(self, long_df, signal, capture_plot, tmp_charts):
        # A non-unique index never equals df.index → the overlay is skipped (no raise, no
        # positional/partial line). Duplicate the first label across two rows.
        generate_chart(long_df, "TEST", "1H", signal, output_dir=tmp_charts, enrich=True)
        baseline_n = len(_panel(capture_plot["addplot"], 0))
        dup_index = long_df.index.delete(1).insert(0, long_df.index[0])  # index[0] appears twice
        htf = pd.Series(long_df["Close"].to_numpy(), index=dup_index)
        generate_chart(long_df, "TEST", "1H", signal, output_dir=tmp_charts, enrich=True, htf_ema=htf)
        assert len(_panel(capture_plot["addplot"], 0)) == baseline_n  # htf line absent


class TestEnrichmentRender:
    def test_enriched_png_smoke(self, long_df, signal, tmp_charts):
        # Real end-to-end render (no mpf.plot stub) — exercises the fill_between + panel API.
        path = generate_chart(long_df, "TEST", "4H", signal, output_dir=tmp_charts, enrich=True)
        assert path.exists()
        assert path.stat().st_size > 1000


class TestGenerateChartsForSignal:
    """The multi-chart bundle (tf_order) — renders whichever of 1d/4h/1h/15m are present."""

    def test_renders_15m_when_present(self, signal, capture_plot, tmp_charts):
        # v2 bundle: all four frames → four charts, HTF→LTF with 15m last (Chunk 8/8.2).
        df = _osc_df(320)
        candle_sets = {tf: df for tf in ("1d", "4h", "1h", "15m")}
        paths = generate_charts_for_signal(signal, candle_sets, output_dir=tmp_charts)
        assert [p.stem.split("_")[1] for p in paths] == ["1D", "4H", "1H", "15M"]

    def test_v1_shape_omits_15m(self, signal, capture_plot, tmp_charts):
        # v1 blind path passes only 1d/4h/1h → 15m never renders (byte-identical to pre-8.2).
        df = _osc_df(320)
        candle_sets = {tf: df for tf in ("1d", "4h", "1h")}
        paths = generate_charts_for_signal(signal, candle_sets, output_dir=tmp_charts)
        assert [p.stem.split("_")[1] for p in paths] == ["1D", "4H", "1H"]
