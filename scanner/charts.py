"""Chart generation via mplfinance — light theme with S/R + indicator overlays."""

import matplotlib

matplotlib.use("agg")

from pathlib import Path

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from matplotlib.patches import Rectangle

from .config import CHARTS_DIR
from .ma_indicators import compute_ema_band, compute_longer_set, compute_stoch_rsi
from .models import BoundarySignal, SpineLevel

# Light theme — better LLM visual acuity on white backgrounds
STYLE = mpf.make_mpf_style(
    base_mpf_style="charles",
    marketcolors=mpf.make_marketcolors(
        up="#26a69a",
        down="#ef5350",
        edge="inherit",
        wick="inherit",
        volume={"up": "#26a69a80", "down": "#ef535080"},
    ),
)

# v10 enrichment palette (light theme). The band fades fast→slow; the longer set is
# distinguished by linestyle so three lines stay legible in greyscale too. Colours only —
# each MA's period is owned by its ma_indicators field, never restated here.
_BAND_FAST, _BAND_MID, _BAND_SLOW = "#1565c0", "#42a5f5", "#90caf9"
_SMA100, _EMA200, _SMA300 = "#fb8c00", "#8e24aa", "#455a64"
_ENVELOPE = {"color": "#90a4ae", "alpha": 0.15}
_HTF_COLOR = "#d81b60"
_STOCH_K, _STOCH_D, _STOCH_MID = "#3949ab", "#ef6c00", "#9e9e9e"
_STOCH_PANEL = 2  # price=0, volume=1, Stoch RSI=2
_ENRICHED_FIGSIZE = (13.5, 9)  # taller than the plain (13.5, 8) to fit the sub-pane
_ENRICHED_PANEL_RATIOS = (6, 1.4, 2)  # price / volume / Stoch, only when the pane exists

# Chart presentation slice — the rendered window per chart. Detection upstream uses the
# full substrate (V2_CANDLE_COUNTS = 400 bars for MA warmup); this slice exists only so
# each candle gets enough pixels to be legible. At figsize 13.5" × 150 dpi = 2025px wide,
# 150 candles ≈ 13.5 px each (vs ~2.4 px each at 400 candles / 70 dpi). MAs/Stoch RSI
# stay computed on the full close, then reindexed to the chart slice.
_CHART_RENDER_BARS = 150


def _ma_addplots(close: pd.Series, chart_index: pd.Index | None = None) -> tuple[list, dict | None]:
    """Panel-0 MA overlays (EMA band + longer set) and the band's envelope fill.

    Returns ``(addplots, fill_between | None)``. Every line is skipped unless it has at
    least one non-NaN value — a too-short ``close`` warms up the band and longer-set edges
    at different bar counts, and mplfinance errors on an all-NaN addplot. The envelope is
    emitted only when both band edges have data, with a numpy-array payload (mplfinance's
    top-level ``fill_between`` rejects bare Series).

    ``chart_index`` — when supplied, every series is reindexed to it after compute, so
    mplfinance receives length-matched series for the chart slice without losing the
    full-history warmup the MAs need.
    """
    band = compute_ema_band(close)
    longer = compute_longer_set(close)

    def _s(series: pd.Series) -> pd.Series:
        return series.reindex(chart_index) if chart_index is not None else series

    # Each line pairs its named substrate field with a colour at the styling site — no
    # positional zip against a separate colour table whose order could silently drift.
    band_lines = ((_s(band.fast), _BAND_FAST), (_s(band.mid), _BAND_MID), (_s(band.slow), _BAND_SLOW))
    longer_lines = (
        (_s(longer.sma_100), _SMA100, "dotted"),
        (_s(longer.ema_200), _EMA200, "dashed"),
        (_s(longer.sma_300), _SMA300, "dashdot"),
    )

    addplots = []
    for series, colour in band_lines:
        if series.notna().any():
            addplots.append(mpf.make_addplot(series, panel=0, color=colour, width=0.8))
    for series, colour, linestyle in longer_lines:
        if series.notna().any():
            addplots.append(
                mpf.make_addplot(series, panel=0, color=colour, width=0.9, linestyle=linestyle)
            )

    fill = None
    fast_chart, slow_chart = band_lines[0][0], band_lines[2][0]
    if fast_chart.notna().any() and slow_chart.notna().any():
        fill = {"y1": fast_chart.to_numpy(), "y2": slow_chart.to_numpy(), **_ENVELOPE}
    return addplots, fill


def _stoch_addplots(close: pd.Series, panel: int, chart_index: pd.Index | None = None) -> list:
    """Stoch-RSI sub-pane addplots (%K, %D, and a constant 50 midline) on ``panel``.

    Returns ``[]`` when Stoch RSI is all-NaN (close too short / flat) so no empty
    sub-panel is created. The 50 midline carries no data, so it is added only when the
    pane already exists — it is not independently guarded. ``chart_index`` reslices the
    computed series to the chart window (same contract as ``_ma_addplots``).
    """
    stoch = compute_stoch_rsi(close)
    k = stoch.k.reindex(chart_index) if chart_index is not None else stoch.k
    d = stoch.d.reindex(chart_index) if chart_index is not None else stoch.d
    addplots = []
    if k.notna().any():
        addplots.append(
            mpf.make_addplot(
                k, panel=panel, color=_STOCH_K, width=0.9, ylabel="Stoch RSI", ylim=(0, 100)
            )
        )
    if d.notna().any():
        addplots.append(mpf.make_addplot(d, panel=panel, color=_STOCH_D, width=0.7))
    if addplots:
        midline_index = chart_index if chart_index is not None else close.index
        midline = pd.Series(50.0, index=midline_index)
        addplots.append(
            mpf.make_addplot(midline, panel=panel, color=_STOCH_MID, width=0.6, linestyle="dotted")
        )
    return addplots


def _sr_hlines(sr_levels: list, df: pd.DataFrame) -> dict | None:
    """Build a single hlines dict for mplfinance with S/R levels visible on the chart."""
    price_range = df["High"].max() - df["Low"].min()
    lo = df["Low"].min() - price_range * 0.05
    hi = df["High"].max() + price_range * 0.05

    prices = []
    colors = []
    styles = []
    for level in sr_levels:
        if lo <= level.price <= hi:
            prices.append(level.price)
            colors.append("#ef5350" if level.degraded else "#4caf50")
            styles.append("dashed" if level.degraded else "solid")

    if not prices:
        return None

    return dict(
        hlines=prices,
        colors=colors,
        linestyle=styles,
        linewidths=0.8,
        alpha=0.7,
    )


# v2 zone rendering palette — green-tinted support, red-tinted resistance (matches
# the candle marketcolors), alpha gradient `_ZONE_ALPHA_LO..._ZONE_ALPHA_HI` so the
# best-quality level reads vivid and the worst reads faded.
_ZONE_SUPPORT, _ZONE_RESISTANCE = "#26a69a", "#ef5350"
_ZONE_ALPHA_LO, _ZONE_ALPHA_HI = 0.18, 0.45
# Cap the on-chart zone count — detection consumes every level, but the chart should
# render only the few highest-quality ones in view. A chart marks 3–6 levels, not
# the whole substrate; rendering all leaves no whitespace.
_MAX_ZONES_PER_CHART = 6
# Each rendered zone gets a thin solid edge at zone_low/zone_high in addition to the
# faint Rectangle fill — TradingView-style SR box. The edge sharpens the boundary that
# the LLM (and the operator) reads as the touch/pierce price.
_ZONE_EDGE_LW = 0.6
_ZONE_EDGE_ALPHA = 0.6


_ZONED_PIVOT_TYPES = frozenset({"high", "low"})


def _has_zoned_levels(sr_levels: list) -> bool:
    """The v2 chart payload — every level carries a complete zone envelope.

    The QUINTUPLE-shape contract: ``zone_low`` + ``zone_high`` + ``quality`` +
    ``pivot_type ∈ {"high", "low"}`` + ``formed_at`` all set. A level with some but
    not all of those, OR a ``pivot_type`` outside the legal set, is malformed and
    rejected at this boundary, never later inside matplotlib. The quintuple covers
    every field ``_sr_zones`` reads, so a producer bug fails here rather than
    silently rendering with a fallback (mis-coloured zone, wrong alpha tier, etc.).
    Mixed-shape payloads (some zoned, some line-only) are likewise a producer bug and
    fail here so a bad call never silently drops the line-only entries on the zoned
    render path. v1 ``SpineLevel`` / ``SRLevel`` leave every zone field unset and the
    renderer falls through to the line-based ``_sr_hlines`` path; v2 ``SpineLevel``
    populates every field via ``from_sr_level_v2`` (``SRLevelV2.quality`` is always
    assigned by ``_assign_quality``'s final pass — ``quality=None`` here is a
    producer bug, not a line-shape)."""
    def _shape(lvl) -> str:
        zl = getattr(lvl, "zone_low", None)
        zh = getattr(lvl, "zone_high", None)
        pt = getattr(lvl, "pivot_type", "") or ""
        fa = getattr(lvl, "formed_at", None)
        q = getattr(lvl, "quality", None)
        pt_present = pt in _ZONED_PIVOT_TYPES
        pt_absent = pt == ""
        present = (zl is not None, zh is not None, pt_present, fa is not None, q is not None)
        absent = (zl is None, zh is None, pt_absent, fa is None, q is None)
        if all(present):
            return "zoned"
        if all(absent):
            return "line"
        return "malformed"

    shapes = {_shape(lvl) for lvl in sr_levels}
    if "malformed" in shapes:
        raise ValueError(
            "chart S/R payload contains a level with partial zone fields — "
            "every zoned level must set zone_low, zone_high, quality, formed_at, "
            "and a pivot_type in {'high', 'low'} together."
        )
    if shapes == {"zoned", "line"}:
        raise ValueError(
            "chart S/R payload mixes zoned and line-only levels — producers must "
            "supply a homogeneous list (all v1 SpineLevel or all v2 SpineLevel)."
        )
    return shapes == {"zoned"}


def _x_left_renderable(formed_at, chart_index) -> float | None:
    """Bar-edge x-coord where a forward-only zone starts on a mpf categorical axis.

    mpf renders bars centered on integer x: bar ``i`` spans visually from ``x = i - 0.5``
    (left edge) to ``x = i + 0.5`` (right edge); chart bounds are ``[-0.5, len(index) -
    0.5]``. ``chart_index.searchsorted(formed_at)`` returns the integer index of the bar
    CONTAINING ``formed_at`` — that's the bar's CENTER. To start a Rectangle at the LEFT
    edge of that bar, subtract 0.5.

    Returns:
      * ``-0.5`` — ``formed_at`` precedes the rendered window; clip to chart-left edge.
      * ``float(idx) - 0.5`` — ``formed_at`` falls inside the rendered window.
      * ``None`` — ``formed_at`` strictly exceeds the last visible bar; zone hadn't formed
        yet in this window, caller skips.

    Trusts the producer contract: zoned-payload validation lives at ``_has_zoned_levels``
    (the single dispatch boundary). A ``formed_at=None`` on this path is impossible if
    that boundary is honoured; if a future direct caller bypasses dispatch with bad
    input, the ``<=`` comparison raises ``TypeError`` loud-and-fast at the same line."""
    if formed_at <= chart_index[0]:
        return -0.5
    if formed_at > chart_index[-1]:
        return None
    return float(chart_index.searchsorted(formed_at)) - 0.5


def _sr_zones(sr_levels: list, df: pd.DataFrame, ax) -> int:
    """Render v2 zoned S/R as forward-extending Rectangles (faint fill) + ``ax.hlines``
    (sharp zone edges) — colour by ``pivot_type``, alpha by normalised ``quality``.
    Returns the count of zones drawn (0 when none are renderable). Same ±5% cushion as
    the line path. Input is guaranteed homogeneous by ``_has_zoned_levels`` at the
    dispatch boundary.

    Stage 1 — filter to RENDERABLE zones: in price window AND ``formed_at`` falls at or
    before the last visible bar (post-window zones skip). Stage 2 — keep the top
    ``_MAX_ZONES_PER_CHART`` by quality among the RENDERABLE set (not pre-filter, or the
    chart would render below capacity whenever any of the top-N had formed after the
    window). Stage 3 — paint each zone forward-only from ``searchsorted(formed_at) - 0.5``
    on mpf's bar-centered categorical x-axis (clipped to ``-0.5`` when formed before the
    window) to ``len(index) - 0.5`` (the right edge of the last bar). Zone artists sit at
    a zorder BELOW the candle collections — read live from ``ax.collections`` since
    ``_sr_zones`` is called AFTER ``mpf.plot`` has already added the candles, and mpf's
    default zorder could shift across versions.
    """
    price_range = df["High"].max() - df["Low"].min()
    lo = df["Low"].min() - price_range * 0.05
    hi = df["High"].max() + price_range * 0.05

    renderable: list[tuple] = []
    for lvl in sr_levels:
        if not (lo <= lvl.price <= hi):
            continue
        x_left = _x_left_renderable(lvl.formed_at, df.index)
        if x_left is None:
            continue
        renderable.append((lvl, x_left))
    if not renderable:
        return 0

    renderable.sort(
        key=lambda r: (r[0].quality if r[0].quality is not None else 0.0), reverse=True,
    )
    rendered = renderable[:_MAX_ZONES_PER_CHART]

    qs = [(r[0].quality if r[0].quality is not None else 0.0) for r in rendered]
    qmin, qmax = min(qs), max(qs)
    qspan = qmax - qmin
    x_right = len(df.index) - 0.5
    # Pin zone artists strictly under mpf's candle/wick collections so candles stay on
    # top. Read live: mpf's default zorder isn't a public contract and could shift.
    candle_z = min((a.zorder for a in ax.collections), default=2)
    fill_z, edge_z = candle_z - 1.0, candle_z - 0.5
    for (lvl, x_left), q in zip(rendered, qs):
        normalized = ((q - qmin) / qspan) if qspan > 0 else 1.0
        fill_alpha = _ZONE_ALPHA_LO + (_ZONE_ALPHA_HI - _ZONE_ALPHA_LO) * normalized
        color = _ZONE_SUPPORT if lvl.pivot_type == "low" else _ZONE_RESISTANCE
        width = x_right - x_left
        height = lvl.zone_high - lvl.zone_low
        ax.add_patch(Rectangle(
            (x_left, lvl.zone_low), width, height,
            facecolor=color, alpha=fill_alpha, lw=0, zorder=fill_z,
        ))
        ax.hlines(lvl.zone_high, x_left, x_right,
                  color=color, lw=_ZONE_EDGE_LW, alpha=_ZONE_EDGE_ALPHA, zorder=edge_z)
        ax.hlines(lvl.zone_low, x_left, x_right,
                  color=color, lw=_ZONE_EDGE_LW, alpha=_ZONE_EDGE_ALPHA, zorder=edge_z)
    return len(rendered)



def generate_chart(
    df: pd.DataFrame,
    coin: str,
    timeframe: str,
    signal: BoundarySignal,
    sr_levels: list | None = None,
    output_dir: Path | None = None,
    *,
    enrich: bool = False,
    htf_ema: pd.Series | None = None,
) -> Path:
    """Generate a single chart PNG. Returns the file path.

    ``enrich=False`` (default) renders the v1 blind chart — candles + volume + S/R lines,
    no indicator overlays (Decision #5). ``enrich=True`` adds the v10 substrate: the EMA-band
    "ribbon" + envelope, the longer MA set, an optional cross-TF anchor EMA, and a Stoch-RSI
    sub-pane. ``htf_ema`` is a pre-aligned higher-TF line (the 4H-200 EMA on LTF charts) that the
    data plane (Chunk 7a) forward-fills onto ``df.index``; the renderer plots it only on an exact
    index match and otherwise skips it — it never repairs alignment or plots a series positionally.
    """
    out_dir = output_dir or CHARTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # MAs / Stoch RSI need the full close to warm up (SMA300 + Stoch RSI smoothing);
    # the chart slice is presentation-only — render the last `_CHART_RENDER_BARS` for
    # legibility, with every addplot reindexed to match.
    full_close = df["Close"]
    if len(df) > _CHART_RENDER_BARS:
        chart_index = df.index[-_CHART_RENDER_BARS:]
        df = df.loc[chart_index]
    else:
        chart_index = df.index

    # Title shows coin + timeframe only — no setup type (avoids anchoring the LLM)
    title = f"{coin}/USDT {timeframe}"

    # `is not None`, not `or`: an explicit empty per-chart payload (`sr_levels=[]`, e.g. a
    # TF whose `derive_chart_levels` returned nothing) MUST render zero zones, not silently
    # fall back to the signal's macro context. None means "caller passed nothing" (v1 path).
    levels_for_render = sr_levels if sr_levels is not None else signal.context.sr_levels
    use_zones = _has_zoned_levels(levels_for_render)

    filename = f"{coin}_{timeframe}_{signal.setup_type.value}.png"
    filepath = out_dir / filename

    kwargs = dict(
        type="candle",
        style=STYLE,
        title=title,
        volume=True,
        figsize=(13.5, 8),
        warn_too_much_data=500,
    )
    if not use_zones:
        # v1/line path: mplfinance handles savefig + hlines together (existing behaviour).
        kwargs["savefig"] = dict(fname=str(filepath), dpi=150, bbox_inches="tight")
        hlines = _sr_hlines(levels_for_render, df)
        if hlines:
            kwargs["hlines"] = hlines
    else:
        # v2 zoned path: `_sr_zones` adds Rectangle + hlines artists post-plot on the
        # price axis (forward-only from formed_at — Sub-G), so we take fig/axes back via
        # `returnfig=True` and call `fig.savefig` ourselves. The line `hlines` kwarg is
        # omitted on this path — zones replace it.
        kwargs["returnfig"] = True

    if enrich:
        addplots, fill = _ma_addplots(full_close, chart_index)
        if htf_ema is not None and htf_ema.notna().any() and htf_ema.index.is_unique:
            # The data plane (Chunk 7a) owns alignment + forward-fill onto the substrate
            # index; the renderer reindexes to the chart slice. Skip when the source index
            # is non-unique (reindex would raise — alignment is upstream's responsibility,
            # never repaired here) or the reindexed series is empty (htf_ema source frame
            # too short / doesn't cover the chart window). Never plot positionally.
            htf_ema_chart = htf_ema.reindex(chart_index)
            if htf_ema_chart.notna().any():
                addplots.append(
                    mpf.make_addplot(htf_ema_chart, panel=0, color=_HTF_COLOR, width=1.0, linestyle="dashed")
                )
        stoch_addplots = _stoch_addplots(full_close, _STOCH_PANEL, chart_index)
        addplots.extend(stoch_addplots)
        if addplots:
            kwargs["addplot"] = addplots
        if fill is not None:
            kwargs["fill_between"] = fill
        if stoch_addplots:  # the sub-pane exists → size three panels + grow the figure
            kwargs["panel_ratios"] = _ENRICHED_PANEL_RATIOS
            kwargs["figsize"] = _ENRICHED_FIGSIZE

    if use_zones:
        result = mpf.plot(df, **kwargs)
        if result is not None:                          # test stubs may return None
            fig, axes = result
            price_ax = axes[0] if isinstance(axes, (list, tuple)) else axes
            _sr_zones(levels_for_render, df, price_ax)
            fig.savefig(str(filepath), dpi=150, bbox_inches="tight")
            plt.close(fig)
    else:
        mpf.plot(df, **kwargs)
    return filepath


def generate_charts_for_signal(
    signal: BoundarySignal,
    candle_sets: dict[str, pd.DataFrame],
    output_dir: Path | None = None,
    *,
    enrich: bool = False,
    htf_emas: dict[str, pd.Series] | None = None,
    per_chart_levels: dict[str, list[SpineLevel]] | None = None,
) -> list[Path]:
    """Generate all charts for a signal. Returns list of file paths.

    Renders whichever of 1D/4H/1H/15M are present in ``candle_sets`` (HTF→LTF). The v1
    blind path passes only 1d/4h/1h, so 15m never renders there; the v2 producer
    (`screener_v2.chart_inputs`) supplies all four (Chunk 8 / 8.2). ``enrich`` /
    ``htf_emas`` are the v2 dispatch's enriched-chart path (Chunk 7b.2b): when
    ``enrich=True`` each chart adds the v10 substrate overlays, and ``htf_emas`` (per-TF,
    pre-aligned by the data plane) supplies the next-HTF anchor EMA. ``per_chart_levels``
    (Sub-chunk H) is the per-chart-TF S/R map (`screener_v2.chart_inputs`): each chart
    renders ITS TF's locally-refined zones (`sr_levels=per_chart_levels[tf]`) rather than
    one shared macro set; an explicit empty list renders zero zones (no fallback). The v1
    path passes none of these — default-off keeps the blind contract (Decision #5), and
    `signal.context.sr_levels` flows through unchanged."""
    chart_paths = []
    htf_emas = htf_emas or {}

    # Render exactly what the producer supplied, in the order it supplied it (the
    # caller builds `candle_sets` HTF→LTF: v2 `screener_v2.chart_inputs`, v1
    # `pipeline._analyze_one`). The renderer owns no TF whitelist — the chartable-TF
    # set is the producer's decision, so a TF the producer intends is never silently
    # dropped here.
    for tf, df in candle_sets.items():
        # When the producer supplies per_chart_levels it MUST cover every rendered TF
        # (`chart_inputs` keys it identically to candle_sets); a missing key is a producer
        # bug and raises here rather than silently falling back to the macro context. An
        # explicit empty list is legal — it renders zero zones (the `is not None` dispatch).
        sr_for_tf = per_chart_levels[tf] if per_chart_levels is not None else None
        path = generate_chart(
            df,
            signal.coin,
            tf.upper(),
            signal,
            sr_levels=sr_for_tf,
            output_dir=output_dir,
            enrich=enrich,
            htf_ema=htf_emas.get(tf),
        )
        chart_paths.append(path)

    return chart_paths


def cleanup_charts(output_dir: Path | None = None) -> int:
    """Remove all chart PNGs from the output directory. Returns count removed."""
    out_dir = output_dir or CHARTS_DIR
    if not out_dir.exists():
        return 0
    count = 0
    for f in out_dir.glob("*.png"):
        f.unlink()
        count += 1
    return count
