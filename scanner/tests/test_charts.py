"""Tests for chart generation."""


import numpy as np
import pandas as pd
import pytest

from scanner.charts import _sr_zones, cleanup_charts, generate_chart, generate_charts_for_signal
from scanner.indicators import build_context
from scanner.ma_indicators import compute_stoch_rsi
from scanner.models import Direction, SetupSignal, SetupType, SpineLevel


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


class TestZoneRendering:
    """v2 zoned S/R via `_sr_zones`/`axhspan`. Independent of `mpf.plot` — we render
    against a real matplotlib axes and assert the patches landed at the expected
    bounds with the expected colour/alpha."""

    def _zoned(self, price, zone_low, zone_high, *, quality=1.0, pivot_type="high"):
        return SpineLevel(
            price=price, degraded=False,
            zone_low=zone_low, zone_high=zone_high,
            quality=quality, pivot_type=pivot_type,
        )

    def test_axhspan_paints_zones_with_quality_alpha_and_pivot_color(self, long_df):
        import matplotlib.pyplot as plt
        levels = [
            self._zoned(price=110.0, zone_low=109.0, zone_high=110.0, quality=1.0, pivot_type="high"),
            self._zoned(price=95.0, zone_low=95.0, zone_high=96.0, quality=0.5, pivot_type="low"),
        ]
        fig, ax = plt.subplots()
        try:
            count = _sr_zones(levels, long_df, ax)
            assert count == 2
            patches = [
                {"y0": p.get_y(), "y1": p.get_y() + p.get_height(),
                 "color": p.get_facecolor(), "alpha": p.get_alpha()}
                for p in ax.patches
            ]
            # Patch bounds match the zones (mpl orders them in insertion order).
            assert (patches[0]["y0"], patches[0]["y1"]) == (109.0, 110.0)
            assert (patches[1]["y0"], patches[1]["y1"]) == (95.0, 96.0)
            # Vivid (1.0 quality) is the higher alpha; faded (0.5) is the lower.
            assert patches[0]["alpha"] > patches[1]["alpha"]
            # Pivot-type colour split: resistance red-tinted, support green-tinted.
            # Compare red vs green channel dominance per RGBA.
            res_r, res_g = patches[0]["color"][0], patches[0]["color"][1]
            sup_r, sup_g = patches[1]["color"][0], patches[1]["color"][1]
            assert res_r > res_g                              # resistance is red-dominant
            assert sup_g > sup_r                              # support is green-dominant
        finally:
            plt.close(fig)

    def test_zones_outside_visible_range_are_skipped(self, long_df):
        import matplotlib.pyplot as plt
        # `long_df` spans roughly 95–225; place a level WAY above the ±5% cushion.
        far = self._zoned(price=10_000.0, zone_low=9_990.0, zone_high=10_010.0)
        fig, ax = plt.subplots()
        try:
            assert _sr_zones([far], long_df, ax) == 0
            assert len(ax.patches) == 0
        finally:
            plt.close(fig)

    def test_partial_zone_fields_raises_at_dispatch(self, long_df, tmp_charts):
        """Codex architecture pass r2: a level with `zone_low` set but `zone_high`
        missing is malformed and must be rejected at the dispatch boundary — never
        slipped through to matplotlib where it would crash with a confusing TypeError.
        """
        partial = SpineLevel(price=104.0, zone_low=103.0)                 # zone_high missing
        context = build_context(long_df)
        signal = SetupSignal(
            coin="PART", setup_type=SetupType.A, direction=Direction.LONG,
            confidence=0.6, key_level=100.0, context=context, description="part",
        )
        with pytest.raises(ValueError, match="partial zone fields"):
            generate_chart(long_df, "PART", "4H", signal, sr_levels=[partial],
                           output_dir=tmp_charts)

    def test_mixed_zoned_and_line_levels_raises_at_dispatch(self, long_df, tmp_charts):
        """Codex architecture pass: mixing v1 line-only and v2 zoned levels in one
        chart payload is a producer bug — the dispatch must raise rather than silently
        drop the line-only entries on the zoned path."""
        zoned = self._zoned(price=110.0, zone_low=109.0, zone_high=110.0,
                            quality=1.0, pivot_type="high")
        line_only = SpineLevel(price=120.0, degraded=False)
        context = build_context(long_df)
        signal = SetupSignal(
            coin="MIX", setup_type=SetupType.A, direction=Direction.LONG,
            confidence=0.6, key_level=100.0, context=context, description="mix",
        )
        with pytest.raises(ValueError, match="homogeneous"):
            generate_chart(long_df, "MIX", "4H", signal, sr_levels=[zoned, line_only],
                           output_dir=tmp_charts)

    def test_generate_chart_routes_v2_levels_through_zones(self, long_df, tmp_charts):
        # End-to-end smoke: v2 SpineLevel with zones → file written via the
        # returnfig + axhspan + manual savefig path. The plain SetupSignal
        # context.sr_levels is ignored once the explicit kwarg supplies zoned levels.
        context = build_context(long_df)
        signal = SetupSignal(
            coin="ZONE", setup_type=SetupType.A, direction=Direction.LONG,
            confidence=0.6, key_level=100.0, context=context, description="zoned",
        )
        zoned = [
            self._zoned(price=110.0, zone_low=109.0, zone_high=110.0, quality=1.0, pivot_type="high"),
            self._zoned(price=95.0, zone_low=95.0, zone_high=96.0, quality=0.5, pivot_type="low"),
        ]
        path = generate_chart(long_df, "ZONE", "4H", signal, sr_levels=zoned, output_dir=tmp_charts)
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
