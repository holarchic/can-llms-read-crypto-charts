"""Chart generation via mplfinance — light theme with S/R + indicator overlays."""

import matplotlib

matplotlib.use("agg")

from pathlib import Path

import mplfinance as mpf
import pandas as pd

from .config import CHARTS_DIR
from .ma_indicators import compute_ema_band, compute_longer_set, compute_stoch_rsi
from .models import SetupSignal

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


def _ma_addplots(close: pd.Series) -> tuple[list, dict | None]:
    """Panel-0 MA overlays (EMA band + longer set) and the band's envelope fill.

    Returns ``(addplots, fill_between | None)``. Every line is skipped unless it has at
    least one non-NaN value — a too-short ``close`` warms up the band and longer-set edges
    at different bar counts, and mplfinance errors on an all-NaN addplot. The envelope is
    emitted only when both band edges have data, with a numpy-array payload (mplfinance's
    top-level ``fill_between`` rejects bare Series).
    """
    band = compute_ema_band(close)
    longer = compute_longer_set(close)

    # Each line pairs its named substrate field with a colour at the styling site — no
    # positional zip against a separate colour table whose order could silently drift.
    band_lines = ((band.fast, _BAND_FAST), (band.mid, _BAND_MID), (band.slow, _BAND_SLOW))
    longer_lines = (
        (longer.sma_100, _SMA100, "dotted"),
        (longer.ema_200, _EMA200, "dashed"),
        (longer.sma_300, _SMA300, "dashdot"),
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
    if band.fast.notna().any() and band.slow.notna().any():
        fill = {"y1": band.fast.to_numpy(), "y2": band.slow.to_numpy(), **_ENVELOPE}
    return addplots, fill


def _stoch_addplots(close: pd.Series, panel: int) -> list:
    """Stoch-RSI sub-pane addplots (%K, %D, and a constant 50 midline) on ``panel``.

    Returns ``[]`` when Stoch RSI is all-NaN (close too short / flat) so no empty
    sub-panel is created. The 50 midline carries no data, so it is added only when the
    pane already exists — it is not independently guarded.
    """
    stoch = compute_stoch_rsi(close)
    addplots = []
    if stoch.k.notna().any():
        addplots.append(
            mpf.make_addplot(
                stoch.k, panel=panel, color=_STOCH_K, width=0.9, ylabel="Stoch RSI", ylim=(0, 100)
            )
        )
    if stoch.d.notna().any():
        addplots.append(mpf.make_addplot(stoch.d, panel=panel, color=_STOCH_D, width=0.7))
    if addplots:
        midline = pd.Series(50.0, index=close.index)
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



def generate_chart(
    df: pd.DataFrame,
    coin: str,
    timeframe: str,
    signal: SetupSignal,
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

    # Title shows coin + timeframe only — no setup type (avoids anchoring the LLM)
    title = f"{coin}/USDT {timeframe}"

    hlines = _sr_hlines(sr_levels or signal.context.sr_levels, df)

    filename = f"{coin}_{timeframe}_{signal.setup_type.value}.png"
    filepath = out_dir / filename

    kwargs = dict(
        type="candle",
        style=STYLE,
        title=title,
        volume=True,
        figsize=(13.5, 8),
        savefig=dict(fname=str(filepath), dpi=70, bbox_inches="tight"),
        warn_too_much_data=500,
    )

    if hlines:
        kwargs["hlines"] = hlines

    if enrich:
        close = df["Close"]
        addplots, fill = _ma_addplots(close)
        if htf_ema is not None and htf_ema.index.equals(df.index) and htf_ema.notna().any():
            # The data plane (Chunk 7a) owns alignment + forward-fill; the renderer only
            # plots an exact-index match. Anything else is skipped, never repaired here or
            # plotted positionally wrong — alignment stays a single upstream responsibility.
            addplots.append(
                mpf.make_addplot(htf_ema, panel=0, color=_HTF_COLOR, width=1.0, linestyle="dashed")
            )
        stoch_addplots = _stoch_addplots(close, _STOCH_PANEL)
        addplots.extend(stoch_addplots)
        if addplots:
            kwargs["addplot"] = addplots
        if fill is not None:
            kwargs["fill_between"] = fill
        if stoch_addplots:  # the sub-pane exists → size three panels + grow the figure
            kwargs["panel_ratios"] = _ENRICHED_PANEL_RATIOS
            kwargs["figsize"] = _ENRICHED_FIGSIZE

    mpf.plot(df, **kwargs)
    return filepath


def generate_charts_for_signal(
    signal: SetupSignal,
    candle_sets: dict[str, pd.DataFrame],
    output_dir: Path | None = None,
) -> list[Path]:
    """Generate all charts (1D, 4H, 1H) for a signal. Returns list of file paths."""
    chart_paths = []

    # Order: HTF to LTF
    tf_order = ["1d", "4h", "1h"]
    for tf in tf_order:
        if tf in candle_sets:
            path = generate_chart(
                candle_sets[tf],
                signal.coin,
                tf.upper(),
                signal,
                output_dir=output_dir,
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
