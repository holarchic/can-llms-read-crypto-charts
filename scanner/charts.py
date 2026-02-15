"""Chart generation via mplfinance — dark theme with S/R + indicator overlays."""

import matplotlib
matplotlib.use("agg")

from pathlib import Path

import mplfinance as mpf
import pandas as pd

from .config import CHARTS_DIR
from .models import MarketContext, SetupSignal, SETUP_NAMES


# Dark theme matching TradingView
STYLE = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    marketcolors=mpf.make_marketcolors(
        up="#26a69a",
        down="#ef5350",
        edge="inherit",
        wick="inherit",
        volume={"up": "#26a69a80", "down": "#ef535080"},
    ),
    rc={
        "axes.labelcolor": "#999999",
        "xtick.color": "#999999",
        "ytick.color": "#999999",
    },
)


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
) -> Path:
    """Generate a single chart PNG. Returns the file path."""
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
        volume=False,
        figsize=(14, 8),
        savefig=dict(fname=str(filepath), dpi=70, bbox_inches="tight"),
        warn_too_much_data=500,
    )

    if hlines:
        kwargs["hlines"] = hlines

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
