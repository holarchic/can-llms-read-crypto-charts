"""Multi-TF data plane for the v10 v2 framework (T-019 Chunk 7a).

Assembles, per coin, the multi-TF ``MarketContextV2`` the v2 detectors and the
enriched charts consume: fetch the canonical timeframes, resample the derived
ones, build a per-TF ``TFSubstrate``, and compute top-down S/R. Additive — nothing
calls this module yet (dispatch behind ``SETUP_VERSION`` is Chunk 7b), so v1 stays
byte-identical. ``config.candle_counts`` is NOT read or mutated; v2 fetch depths
live in the isolated ``V2_CANDLE_COUNTS`` constant.

Closed-bar contract (load-bearing): every frame the plane returns holds only
finalized bars. Native frames have their forming last bar dropped before any
assembly or resample; resampled frames drop the partial right-edge bucket. The v2
substrate (trend_state / regime / S-R) is right-edge-sensitive, so this
normalization is what stops it ingesting in-progress state — distinct from the v1
scanner path, which tolerates the forming bar.

Substrates only for the deep setup TFs (``SETUP_TFS``), fetched ≥400 bars so the
100/200/300 longer stack is real (non-FLAT ``longer_trend_direction``). The macro
TFs (``1w``/``1mo``/``3d``) are candle-only context: ``derive_sr_levels`` reads
their raw frames + fractal pivots, so their shallow depths are correct — they
never need a 300-bar stack.
"""

from typing import NamedTuple

import pandas as pd

from scanner.config import ScannerConfig
from scanner.data import fetch_candles
from scanner.ma_indicators import (
    compute_ema,
    compute_ema_band,
    compute_longer_set,
    compute_volume_profile,
)
from scanner.models import SRLevelV2
from scanner.regime import detect_regime
from scanner.setups_v2 import SetupContext, TFSubstrate, setup_context_from_df
from scanner.sr_levels import derive_sr_levels
from scanner.trend_state import derive_trend_state

__all__ = [
    "CANONICAL_TFS",
    "SETUP_TFS",
    "MIN_SUBSTRATE_BARS",
    "V2_CANDLE_COUNTS",
    "MarketContextV2",
    "resample_ohlcv",
    "fetch_multi_tf",
    "build_tf_substrate",
    "build_market_context_v2",
    "htf_ema_for_chart",
]

# The 7 canonical timeframe keys, HTF→LTF (Chunk-0 contract).
CANONICAL_TFS: tuple[str, ...] = ("1mo", "1w", "3d", "1d", "4h", "1h", "15m")

# The deep TFs the detectors run on. Substrates + setup_contexts are built only for
# these, and only when the frame clears ``MIN_SUBSTRATE_BARS``.
SETUP_TFS: tuple[str, ...] = ("1d", "4h", "1h", "15m")

# A setup TF earns a substrate iff its closed-bar frame has at least this many bars
# (the SMA300 floor). Below it, the longer stack is all-NaN and the read is junk, so
# the TF is simply absent from the dicts (define-errors-out, no raise).
MIN_SUBSTRATE_BARS = 300

# v2-specific fetch depths, ISOLATED from v1's ``config.candle_counts`` — mutating
# that would change v1 stage-1/2 behaviour. Setup TFs are deep enough (400) that the
# 100/200/300 longer stack is non-degenerate (~100 valid SMA300 points); macro TFs
# are shallow candle-only context. ``3d`` is the post-resample trim target: its
# ``1d=400`` source yields ~133 3d bars, trimmed to 130.
V2_CANDLE_COUNTS: dict[str, int] = {
    "15m": 400,
    "1h": 400,
    "4h": 400,
    "1d": 400,
    "3d": 130,
    "1w": 104,
    "1mo": 36,
}

# Canonical key → ccxt interval. Identity except ``1mo``→``1M``; ``3d`` is absent
# (no perp venue serves native 3D — it is always D1-resampled).
_CCXT_INTERVAL: dict[str, str] = {
    "1mo": "1M",
    "1w": "1w",
    "1d": "1d",
    "4h": "4h",
    "1h": "1h",
    "15m": "15m",
}

# Derived key → (source key, pandas resample rule, source bars per complete bucket).
# ``period_bars`` is the constituent count a complete derived bar requires; the
# right-edge bucket is dropped when its source-bar count falls short.
_RESAMPLE_RULE: dict[str, tuple[str, str, int]] = {"3d": ("1d", "3D", 3)}


class MarketContextV2(NamedTuple):
    """The per-coin multi-TF context the v2 detectors + enriched charts consume."""

    coin: str
    candles: dict[str, pd.DataFrame]  # all 7 canonical TF keys → closed-bar OHLCV
    substrates: dict[str, TFSubstrate]  # SETUP_TFS present at ≥ MIN_SUBSTRATE_BARS
    setup_contexts: dict[str, SetupContext]  # same keys as substrates
    sr_levels: list[SRLevelV2]  # top-down W/M → D/3D


def _drop_forming_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the in-progress last bar — the codebase's closed-bar convention.

    Deterministic (no wall-clock), so callers and tests stay time-independent.
    """
    return df.iloc[:-1]


def resample_ohlcv(df: pd.DataFrame, rule: str, count: int, period_bars: int) -> pd.DataFrame:
    """OHLCV-correct downsample of ``df`` to ``rule``, trimmed to the last ``count`` bars.

    Aggregation is open-time bucketed (``label=closed="left"``, matching the native
    candle index): O=first, H=max, L=min, C=last, V=sum. Empty buckets (a source gap)
    are dropped. The right-edge bucket is dropped when its source-bar count is below
    ``period_bars`` — a partial last bucket is forming-bar lookahead, whereas a partial
    *first/old* bucket is kept (no lookahead). Pure.
    """
    agg = df.resample(rule, label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    sizes = df.resample(rule, label="left", closed="left").size()
    agg = agg.dropna(how="all", subset=["Open", "High", "Low", "Close"])
    sizes = sizes.reindex(agg.index)
    if len(agg) and int(sizes.iloc[-1]) < period_bars:
        agg = agg.iloc[:-1]
    return agg.iloc[-count:]


def fetch_multi_tf(
    coin: str,
    config: ScannerConfig | None = None,
    *,
    fetcher=fetch_candles,
    counts: dict[str, int] = V2_CANDLE_COUNTS,
) -> dict[str, pd.DataFrame]:
    """Fetch + assemble the 7 canonical closed-bar frames for ``coin``.

    Native keys are fetched at the ``counts`` depth and have their forming last bar
    dropped before anything else; derived keys are resampled off the already-closed
    source. ``fetcher``/``counts`` are injectable so the plane is unit-testable
    against synthetic candle dicts without a live exchange.
    """
    out: dict[str, pd.DataFrame] = {}
    for key in CANONICAL_TFS:
        if key in _RESAMPLE_RULE:
            continue
        df = fetcher(coin, _CCXT_INTERVAL[key], counts[key], config)
        out[key] = _drop_forming_bar(df)
    for key, (src_key, rule, period_bars) in _RESAMPLE_RULE.items():
        out[key] = resample_ohlcv(out[src_key], rule, counts[key], period_bars)
    return {key: out[key] for key in CANONICAL_TFS}  # canonical HTF→LTF order


def build_tf_substrate(df: pd.DataFrame) -> TFSubstrate:
    """Assemble the per-TF substrate: band → longer set → trend state → regime.

    The volume profile is wired into ``detect_regime`` (the deferred VRVP→ROTATIONAL
    path for regime — distinct from the range-anchored S/R VRVP, which is Chunk 7b).
    """
    close = df["Close"]
    band = compute_ema_band(close)
    longer = compute_longer_set(close)
    trend_state = derive_trend_state(df, band, longer)
    vrvp = compute_volume_profile(df)
    regime = detect_regime(df, trend_state, longer, vrvp=vrvp)
    return TFSubstrate(bands=band, longer=longer, trend_state=trend_state, regime=regime)


def build_market_context_v2(coin: str, candles: dict[str, pd.DataFrame]) -> MarketContextV2:
    """Build the per-coin ``MarketContextV2`` from a closed-bar candle dict.

    Substrates + setup_contexts are built for each ``SETUP_TFS`` frame present at
    ``len ≥ MIN_SUBSTRATE_BARS``; a short/absent setup TF is simply omitted (no raise).
    S/R levels come top-down from the macro W/M/D/3D frames with ``vrvp=None`` — the
    range-anchored S/R confluence is deferred to Chunk 7b.
    """
    substrates: dict[str, TFSubstrate] = {}
    setup_contexts: dict[str, SetupContext] = {}
    for tf in SETUP_TFS:
        df = candles.get(tf)
        if df is None or len(df) < MIN_SUBSTRATE_BARS:
            continue
        setup_contexts[tf] = setup_context_from_df(df)
        substrates[tf] = build_tf_substrate(df)
    sr_levels = derive_sr_levels(
        weekly=candles.get("1w"),
        monthly=candles.get("1mo"),
        daily=candles.get("1d"),
        three_day=candles.get("3d"),
        vrvp=None,
    )
    return MarketContextV2(
        coin=coin,
        candles=candles,
        substrates=substrates,
        setup_contexts=setup_contexts,
        sr_levels=sr_levels,
    )


def _infer_interval(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Modal bar interval of ``index`` — the median gap between consecutive bars."""
    diffs = index.to_series().diff().dropna()
    return diffs.median() if len(diffs) else pd.Timedelta(0)


def htf_ema_for_chart(htf_df: pd.DataFrame, ltf_index: pd.DatetimeIndex, period: int = 200) -> pd.Series:
    """Align an HTF EMA onto ``ltf_index`` on a close-time↔close-time basis.

    Each HTF EMA value is keyed by its bar's close-time (open-time + one HTF interval)
    and each LTF bar by its own close-time; a backward as-of merge then gives every LTF
    bar the EMA of the most recent HTF bar that closed *by that LTF bar's own close*. So
    the step lands on the LTF bar during which the HTF bar closes — no 1-bar lag — and an
    unclosed last HTF bar (close-time in the future) never leaks onto earlier closed LTF
    bars — no lookahead. Returns a series exactly on ``ltf_index`` (the Chunk-6 renderer
    gates on ``index.equals``); LTF bars before the first valid HTF EMA close-time are NaN.
    """
    ema = compute_ema(htf_df["Close"], period)
    htf_close = htf_df.index + _infer_interval(htf_df.index)
    ltf_close = ltf_index + _infer_interval(ltf_index)
    left = pd.DataFrame({"ltf_close": ltf_close})
    right = pd.DataFrame({"htf_close": htf_close, "ema": ema.to_numpy()}).dropna(subset=["ema"])
    merged = pd.merge_asof(left, right, left_on="ltf_close", right_on="htf_close", direction="backward")
    return pd.Series(merged["ema"].to_numpy(), index=ltf_index)
