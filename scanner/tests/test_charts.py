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
    """v2 zoned S/R via `_sr_zones` — forward-only Rectangle + `ax.hlines` (Sub-G).
    Independent of `mpf.plot` — we render against a real matplotlib axes and assert the
    patches landed at the expected bounds with the expected colour/alpha."""

    def _zoned(self, price, zone_low, zone_high, *, quality=1.0, pivot_type="high",
               formed_at=None):
        # Default formed_at sits BEFORE the chart's start (`_osc_df` defaults to
        # 2026-01-01), so the rendered Rectangle spans the full chart width — the
        # pre-G axhspan behaviour for these specific tests' purposes.
        return SpineLevel(
            price=price, degraded=False,
            zone_low=zone_low, zone_high=zone_high,
            quality=quality, pivot_type=pivot_type,
            formed_at=formed_at if formed_at is not None else pd.Timestamp("2025-12-01", tz="UTC"),
        )

    def test_rectangle_paints_zones_with_quality_alpha_and_pivot_color(self, long_df):
        """Zones render as forward-extending Rectangles (Sub-G replaces axhspan): y-bounds
        are the zone, x-bounds run from the formation bar's left edge to chart's right
        edge. Quality drives alpha; pivot_type drives colour."""
        import matplotlib.pyplot as plt
        levels = [
            self._zoned(price=110.0, zone_low=109.0, zone_high=110.0, quality=1.0, pivot_type="high"),
            self._zoned(price=95.0, zone_low=95.0, zone_high=96.0, quality=0.5, pivot_type="low"),
        ]
        fig, ax = plt.subplots()
        try:
            count = _sr_zones(levels, long_df, ax)
            assert count == 2
            x_right = len(long_df.index) - 0.5
            patches = [
                {"x0": p.get_x(), "width": p.get_width(),
                 "y0": p.get_y(), "y1": p.get_y() + p.get_height(),
                 "color": p.get_facecolor(), "alpha": p.get_alpha()}
                for p in ax.patches
            ]
            # Both levels formed pre-window (default helper formed_at = 2025-12-01) so
            # both Rectangles span the full chart width: x0 = -0.5, right edge = x_right.
            for patch in patches:
                assert patch["x0"] == -0.5
                assert patch["x0"] + patch["width"] == x_right
            # Patch bounds match the zones (mpl orders them in insertion order). Insertion
            # order = quality-descending → resistance (q=1.0) first, support (q=0.5) second.
            assert (patches[0]["y0"], patches[0]["y1"]) == (109.0, 110.0)
            assert (patches[1]["y0"], patches[1]["y1"]) == (95.0, 96.0)
            # Vivid (1.0 quality) is the higher alpha; faded (0.5) is the lower.
            assert patches[0]["alpha"] > patches[1]["alpha"]
            # Pivot-type colour split: resistance red-tinted, support green-tinted.
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

    def test_sr_zones_rectangle_starts_at_bar_left_edge_when_formed_inside_chart(self, long_df):
        """Sub-G geometry: a zone whose `formed_at` matches bar 50 of the chart renders
        as a Rectangle starting at x = 49.5 (the LEFT edge of bar 50 on mpf's bar-centered
        categorical axis — `searchsorted - 0.5`), spanning to chart-right."""
        import matplotlib.pyplot as plt
        formed = long_df.index[50]
        lvl = self._zoned(price=110.0, zone_low=109.0, zone_high=110.0,
                          quality=1.0, pivot_type="high", formed_at=formed)
        fig, ax = plt.subplots()
        try:
            assert _sr_zones([lvl], long_df, ax) == 1
            (patch,) = ax.patches
            assert patch.get_x() == 49.5                          # bar 50's LEFT edge
            assert patch.get_x() + patch.get_width() == len(long_df.index) - 0.5
        finally:
            plt.close(fig)

    def test_sr_zones_rectangle_clipped_to_chart_left_when_formed_before_window(self, long_df):
        """A zone formed BEFORE chart-start renders from x = -0.5 (chart-left edge),
        spanning the full chart width."""
        import matplotlib.pyplot as plt
        before = long_df.index[0] - pd.Timedelta(days=30)
        lvl = self._zoned(price=110.0, zone_low=109.0, zone_high=110.0,
                          quality=1.0, pivot_type="high", formed_at=before)
        fig, ax = plt.subplots()
        try:
            assert _sr_zones([lvl], long_df, ax) == 1
            (patch,) = ax.patches
            assert patch.get_x() == -0.5
            assert patch.get_width() == float(len(long_df.index))
        finally:
            plt.close(fig)

    def test_sr_zones_renders_level_formed_at_last_visible_bar(self, long_df):
        """A zone whose `formed_at` is the LAST visible bar's timestamp renders as a
        thin Rectangle in the last bar's slot — not skipped (the `>` check excludes
        only strictly-after-window, not at-window-end)."""
        import matplotlib.pyplot as plt
        n = len(long_df.index)
        lvl = self._zoned(price=110.0, zone_low=109.0, zone_high=110.0,
                          quality=1.0, pivot_type="high", formed_at=long_df.index[-1])
        fig, ax = plt.subplots()
        try:
            assert _sr_zones([lvl], long_df, ax) == 1
            (patch,) = ax.patches
            assert patch.get_x() == float(n - 1) - 0.5            # last bar's LEFT edge
            assert patch.get_width() == 1.0                       # one bar's slot
        finally:
            plt.close(fig)

    def test_sr_zones_skips_level_formed_strictly_after_chart_window(self, long_df):
        """A zone whose `formed_at` STRICTLY exceeds the last visible bar is skipped —
        the zone hadn't formed yet in the rendered window."""
        import matplotlib.pyplot as plt
        after = long_df.index[-1] + pd.Timedelta(days=30)
        lvl = self._zoned(price=110.0, zone_low=109.0, zone_high=110.0,
                          quality=1.0, pivot_type="high", formed_at=after)
        fig, ax = plt.subplots()
        try:
            assert _sr_zones([lvl], long_df, ax) == 0
            assert len(ax.patches) == 0
        finally:
            plt.close(fig)

    def test_sr_zones_top_n_cap_applies_after_renderable_filter(self, long_df, monkeypatch):
        """Sub-G ordering: filter to RENDERABLE first, then cap top-N. Bad ordering
        (top-N first, filter second) would silently render below capacity whenever any
        of the top-N had formed after the window. Pin both the count AND the identities
        so the bad ordering fails on identity, not just count."""
        import matplotlib.pyplot as plt

        from scanner import charts
        monkeypatch.setattr(charts, "_MAX_ZONES_PER_CHART", 3)
        # 10 zones at DISTINCT in-window prices; quality A=10 (highest)..J=1 (lowest).
        # Top 6 by quality (A-F) have formed_at AFTER the chart's last bar → skipped.
        # Bottom 4 (G-J) have formed_at BEFORE chart-start → renderable.
        # Correct (renderable-first): renderable={G,H,I,J}; top-3 by quality={G,H,I}.
        # Bad (top-N-first): top-3 from full set={A,B,C}; renderable-filter drops them → 0.
        # Distinct prices let us pin IDENTITY (which labels survived), not just count.
        after = long_df.index[-1] + pd.Timedelta(days=30)
        before = long_df.index[0] - pd.Timedelta(days=30)
        labels = list("ABCDEFGHIJ")
        prices = [150.0 + i for i in range(len(labels))]                     # 150..159
        label_to_price = dict(zip(labels, prices))
        levels = []
        for i, (label, price) in enumerate(zip(labels, prices)):
            q = float(10 - i)                                                # A=10 down to J=1
            formed_at = after if i < 6 else before                           # A-F skip, G-J renderable
            levels.append(self._zoned(
                price=price, zone_low=price - 1.0, zone_high=price,
                quality=q, pivot_type="high", formed_at=formed_at,
            ))
        fig, ax = plt.subplots()
        try:
            count = _sr_zones(levels, long_df, ax)
            assert count == 3
            # Identity pin: surviving Rectangles' y-bounds must match G/H/I's price levels
            # exactly. Quality-descending insertion order = G(q=4), H(q=3), I(q=2).
            expected = [label_to_price[label] for label in ("G", "H", "I")]
            rendered_y_high = [p.get_y() + p.get_height() for p in ax.patches]
            assert rendered_y_high == expected
        finally:
            plt.close(fig)

    def test_sr_zones_zorder_below_candles(self, long_df, signal, tmp_charts):
        """Sub-G: zone artists (Rectangle fill + edge hlines) MUST sit at a zorder BELOW
        the candle collections so candles stay visible on top. Read mpf's actual zorder
        live; the literal default could shift across mplfinance bumps."""
        import matplotlib.pyplot as plt
        import mplfinance as mpf
        fig, axes = mpf.plot(long_df, type="candle", returnfig=True, volume=True)
        ax = axes[0]
        try:
            # Snapshot BEFORE _sr_zones so the candle baseline isn't poisoned by the new
            # edge LineCollections we are about to add.
            pre_collections = list(ax.collections)
            candle_z = min(a.zorder for a in pre_collections)
            lvl = self._zoned(price=110.0, zone_low=109.0, zone_high=110.0,
                              quality=1.0, pivot_type="high")
            _sr_zones([lvl], long_df, ax)
            new_collections = [a for a in ax.collections if a not in pre_collections]
            assert all(a.zorder < candle_z for a in new_collections)          # edge hlines
            assert all(p.zorder < candle_z for p in ax.patches)               # fill Rectangle
        finally:
            plt.close(fig)

    def test_sr_zones_no_axhspan_called(self, long_df, monkeypatch):
        """Regression guard: _sr_zones must NOT call ax.axhspan (the pre-G back-dating
        bug). Monkeypatch axhspan to raise; rendering valid zoned input should succeed."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        monkeypatch.setattr(ax, "axhspan", lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("axhspan must not be called post-G")
        ))
        try:
            lvl = self._zoned(price=110.0, zone_low=109.0, zone_high=110.0,
                              quality=1.0, pivot_type="high")
            assert _sr_zones([lvl], long_df, ax) == 1
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

    def test_zoned_with_missing_quality_raises_at_dispatch(self, long_df, tmp_charts):
        """Codex arch r3 MEDIUM: a zoned level with `quality=None` must be rejected at
        the dispatch boundary — `_sr_zones` reads quality for both top-N selection AND
        alpha; pre-r3 a None quality silently fell back to 0.0, masking a producer bug.
        Production `SRLevelV2.quality` is always populated by `_assign_quality`."""
        no_quality = SpineLevel(
            price=110.0, zone_low=109.0, zone_high=110.0,
            pivot_type="high", quality=None,
            formed_at=pd.Timestamp("2025-12-01", tz="UTC"),
        )
        context = build_context(long_df)
        signal = SetupSignal(
            coin="NOQ", setup_type=SetupType.A, direction=Direction.LONG,
            confidence=0.6, key_level=100.0, context=context, description="noq",
        )
        with pytest.raises(ValueError, match="partial zone fields"):
            generate_chart(long_df, "NOQ", "4H", signal, sr_levels=[no_quality],
                           output_dir=tmp_charts)

    def test_zoned_with_invalid_pivot_type_raises_at_dispatch(self, long_df, tmp_charts):
        """Codex arch r2 MEDIUM: a zoned level whose `pivot_type` is outside
        `{'high', 'low'}` must be rejected at the dispatch boundary — not silently
        rendered via `_sr_zones`'s colour fallback (`pivot_type == 'low'` → support
        else resistance). The boundary owns pivot_type-value validation, not the
        renderer."""
        bogus = SpineLevel(
            price=110.0, zone_low=109.0, zone_high=110.0,
            pivot_type="bogus", quality=1.0,
            formed_at=pd.Timestamp("2025-12-01", tz="UTC"),
        )
        context = build_context(long_df)
        signal = SetupSignal(
            coin="BOGUS", setup_type=SetupType.A, direction=Direction.LONG,
            confidence=0.6, key_level=100.0, context=context, description="bogus",
        )
        with pytest.raises(ValueError, match="partial zone fields"):
            generate_chart(long_df, "BOGUS", "4H", signal, sr_levels=[bogus],
                           output_dir=tmp_charts)

    def test_zoned_with_missing_formed_at_raises_at_dispatch(self, long_df, tmp_charts):
        """Sub-G: a level with zone fields BUT `formed_at=None` is malformed under the
        quadruple-shape contract. Without this, a producer bug would silently fall open
        to chart-left back-dating — the exact bug `_has_zoned_levels` is designed to
        prevent at the dispatch boundary (parallel to F arch-r2's triple-shape rejection)."""
        missing = SpineLevel(
            price=110.0, zone_low=109.0, zone_high=110.0,
            pivot_type="high", quality=1.0, formed_at=None,
        )
        context = build_context(long_df)
        signal = SetupSignal(
            coin="MISS", setup_type=SetupType.A, direction=Direction.LONG,
            confidence=0.6, key_level=100.0, context=context, description="miss",
        )
        with pytest.raises(ValueError, match="partial zone fields"):
            generate_chart(long_df, "MISS", "4H", signal, sr_levels=[missing],
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
        # returnfig + Rectangle + manual savefig path (Sub-G). The plain SetupSignal
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
