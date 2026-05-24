"""v10 the framework trend-family setup detectors (T-019 Chunk 5a).

Three pure detectors — ``detect_trend_continuation`` / ``detect_trend_reclaim`` /
``detect_deviation_reclaim`` — over the Chunk-1/2/3/4 substrate. Each is anchored on
the local trend (EMA 13/25/36) as both basis AND invalidation (the framework 2026-01-13). No
pipeline wiring, no LLM, no DB; each returns 0-N ``SetupSignalV2`` (in ``models``).
Rotational/counter-trend detectors are Chunk 5b (reuse this module's bundles + the
``models`` signal contract).

Two per-TF bundles live here (NOT in ``models`` — they reference the indicator/regime
layer, and ``models`` must take no upward deps):

- ``SetupContext`` — the setup-TF structural read PLUS the raw ``df``. It carries
  candle history (the band/impulse/deviation rules need it) and deliberately
  EXCLUDES the v1 ``sr_levels``/``regime`` fields ``MarketContext`` holds, so a
  detector cannot read v1 levels by accident. Build it with
  ``setup_context_from_df`` (which owns the literal-empty guard — ``build_context``
  is not total on an empty frame).
- ``TFSubstrate`` — the per-TF MA substrate (bands + longer set + ``TrendState`` +
  ``RegimeState``), already computed together upstream.

Cross-TF concerns stay out: ``detect_trend_reclaim`` takes the one-TF-lower 200EMA
reclaim as the explicit keyword ``lower_tf_reclaim`` (Chunk 7a computes it). All
thresholds are v1 defaults, calibrated Chunk 8b. Every detector degrades to ``[]`` on
short history / NaN substrate and never raises.
"""

from typing import NamedTuple

import pandas as pd

from .indicators import build_context
from .ma_indicators import EMABand, LongerMASet
from .models import (
    MSB,
    ConfluenceFactor,
    Direction,
    SetupSignalV2,
    SetupTypeV2,
    SoftStop,
    SoftStopMode,
    SRLevelV2,
    Structure,
    SwingPoint,
)
from .regime import Regime, RegimeState
from .trend_state import BandPosition, TrendDirection, TrendState

__all__ = [
    "SetupContext",
    "TFSubstrate",
    "setup_context_from_df",
    "detect_trend_continuation",
    "detect_trend_reclaim",
    "detect_deviation_reclaim",
]


class SetupContext(NamedTuple):
    """Setup-TF structural read + raw ``df`` (history); no v1 ``sr_levels``/``regime``."""

    df: pd.DataFrame
    structure: Structure
    swings: list[SwingPoint]
    msbs: list[MSB]
    range_bounds: tuple[float, float] | None
    atr: float
    current_price: float
    efficiency_ratio: float


class TFSubstrate(NamedTuple):
    """The per-TF MA substrate the detectors read (bands → trend_state → regime)."""

    bands: EMABand
    longer: LongerMASet
    trend_state: TrendState
    regime: RegimeState


def setup_context_from_df(df: pd.DataFrame) -> SetupContext:
    """Build a ``SetupContext`` from a setup-TF OHLCV frame via v1 ``build_context``.

    The literal-empty frame is rejected HERE (``build_context`` indexes
    ``df["Close"].iloc[-1]`` and is not total on empty) so detectors receive only a
    totalized context.
    """
    if df.empty:
        raise ValueError("setup_context_from_df requires a non-empty OHLCV frame")
    ctx = build_context(df)
    return SetupContext(
        df=df,
        structure=ctx.structure,
        swings=ctx.swings,
        msbs=ctx.msbs,
        range_bounds=ctx.range_bounds,
        atr=ctx.atr,
        current_price=ctx.current_price,
        efficiency_ratio=ctx.efficiency_ratio,
    )


def _isnan(x: float) -> bool:
    return bool(pd.isna(x))


def _latest(series: pd.Series) -> float:
    return float(series.iloc[-1]) if len(series) else float("nan")


def _direction(td: TrendDirection) -> Direction | None:
    """LONG/SHORT for an UP/DOWN local stack; None when FLAT (no trend setup)."""
    if td is TrendDirection.UP:
        return Direction.LONG
    if td is TrendDirection.DOWN:
        return Direction.SHORT
    return None


def _reclaimed_full_ribbon(direction: Direction, pos: BandPosition) -> bool:
    """The full-band reclaim (D4): price beyond the whole ribbon, not just EMA36."""
    if direction is Direction.LONG:
        return pos is BandPosition.ABOVE
    return pos is BandPosition.BELOW


def _has_impulse(
    df: pd.DataFrame, direction: Direction, lookback: int, impulse_k: float, atr: float,
) -> bool:
    """Vertical accumulation: a *same-direction* impulse candle (signed body beyond
    impulse_k·ATR) in the last ``lookback`` bars — a green body for a long, red for a
    short. Steady grind (or a counter-trend impulse) → False → no with-trend entry."""
    body = (df["Close"] - df["Open"]).iloc[-lookback:]
    if direction is Direction.LONG:
        return bool((body > impulse_k * atr).any())
    return bool((body < -impulse_k * atr).any())


def _deviated_within_lookback(
    df: pd.DataFrame, bands: EMABand, direction: Direction, lookback: int,
) -> bool:
    """Did price trade through to the opposite side of the ribbon in the last
    ``lookback`` bars (a deviation that has since reclaimed)? Recomputes the band
    position per bar from the raw band series — the history seam in use."""
    close = df["Close"]
    n = len(close)
    for i in range(max(0, n - lookback), n):
        f, m, s = bands.fast.iloc[i], bands.mid.iloc[i], bands.slow.iloc[i]
        if pd.isna(f) or pd.isna(m) or pd.isna(s):
            continue
        c = close.iloc[i]
        if direction is Direction.LONG and c < min(f, m, s):
            return True
        if direction is Direction.SHORT and c > max(f, m, s):
            return True
    return False


def _near_sr(price: float, sr_levels: list[SRLevelV2], atr: float, tol: float) -> bool:
    return any(abs(price - lvl.price) <= tol * atr for lvl in sr_levels)


def _recent_msb(msbs: list[MSB], n: int, direction: Direction, lookback: int) -> bool:
    want = "bullish" if direction is Direction.LONG else "bearish"
    cutoff = n - lookback
    return any(m.type == want and m.index >= cutoff for m in msbs)


def _htf_block(
    sr_levels: list[SRLevelV2], entry: float, direction: Direction, atr: float, tol: float,
) -> bool:
    """HTF resistance (long) / support (short) within ``tol·ATR`` of the reclaim point
    invalidates the reclaim (rule line 189)."""
    if direction is Direction.LONG:
        ahead = [lvl.price for lvl in sr_levels if lvl.price > entry]
        return bool(ahead) and (min(ahead) - entry) <= tol * atr
    behind = [lvl.price for lvl in sr_levels if lvl.price < entry]
    return bool(behind) and (entry - max(behind)) <= tol * atr


def _protected_stop(swings: list[SwingPoint], direction: Direction, entry: float) -> float:
    """The nearest protective swing on the stop side of ``entry`` (a swing low BELOW
    entry for a long / swing high ABOVE for a short) — the structure whose break
    invalidates the setup. NaN when no such structure exists: the detector then fails
    closed (a trend setup with no structural invalidation is not well-formed — no
    ATR-derived stop masquerading as structural)."""
    if direction is Direction.LONG:
        lows = [s.body_price for s in swings if s.type == "low" and s.body_price < entry]
        return max(lows) if lows else float("nan")
    highs = [s.body_price for s in swings if s.type == "high" and s.body_price > entry]
    return min(highs) if highs else float("nan")


def _weakness_flag(swings: list[SwingPoint], direction: Direction) -> bool:
    """First sign of weakness: the most recent same-direction swing failed to extend
    (no new HH for a long / no new LL for a short) vs the prior one."""
    want = "high" if direction is Direction.LONG else "low"
    same = [s for s in swings if s.type == want]
    if len(same) < 2:
        return False
    last, prev = same[-1].body_price, same[-2].body_price
    return last <= prev if direction is Direction.LONG else last >= prev


def _retest_factors(
    entry: float, price: float, atr: float, sr_levels: list[SRLevelV2],
    sr_tol: float, retest_tol: float,
) -> set[ConfluenceFactor]:
    factors: set[ConfluenceFactor] = set()
    if _near_sr(entry, sr_levels, atr, sr_tol):
        factors.add(ConfluenceFactor.AT_SR)
    if abs(price - entry) <= retest_tol * atr:
        factors.add(ConfluenceFactor.RETEST)
    return factors


def detect_trend_continuation(
    context: SetupContext,
    substrate: TFSubstrate,
    sr_levels: list[SRLevelV2],
    *,
    sr_tol: float = 0.5,
    retest_tol: float = 0.5,
    impulse_k: float = 1.5,
    lookback: int = 10,
    min_bars: int = 40,
) -> list[SetupSignalV2]:
    """With-trend retest in a TRENDING regime, the band holding as support/resistance."""
    ts = substrate.trend_state
    direction = _direction(ts.local_trend_direction)
    if direction is None or len(context.df) < min_bars:
        return []
    if substrate.regime.regime is not Regime.TRENDING:
        return []
    if ts.local_trend_direction is not ts.longer_trend_direction:  # aligned local == longer
        return []
    if not _reclaimed_full_ribbon(direction, ts.price_vs_ribbon):  # trend intact, band held
        return []
    ema36, price, atr = _latest(substrate.bands.slow), context.current_price, context.atr
    if _isnan(ema36) or _isnan(price) or _isnan(atr) or atr <= 0:
        return []
    if not _has_impulse(context.df, direction, lookback, impulse_k, atr):  # vertical accumulation
        return []
    hard = _protected_stop(context.swings, direction, ema36)
    if _isnan(hard):  # fail closed — no structural invalidation, not well-formed
        return []
    factors ={ConfluenceFactor.TREND_ALIGNED, ConfluenceFactor.VERTICAL_ACCUMULATION}
    factors |= _retest_factors(ema36, price, atr, sr_levels, sr_tol, retest_tol)
    return [SetupSignalV2(
        setup_type=SetupTypeV2.TREND_CONTINUATION,
        direction=direction,
        basis_price=ema36,
        entry=ema36,
        hard_stop=hard,
        soft_stop=SoftStop(SoftStopMode.TREND_RECLAIM_AGAINST, ema36),
        confluence=frozenset(factors),
        weakness_flag=_weakness_flag(context.swings, direction),
    )]


def detect_trend_reclaim(
    context: SetupContext,
    substrate: TFSubstrate,
    sr_levels: list[SRLevelV2],
    *,
    lower_tf_reclaim: bool,
    htf_res_tol: float = 1.0,
    sr_tol: float = 0.5,
    retest_tol: float = 0.5,
    msb_lookback: int = 10,
    min_bars: int = 40,
) -> list[SetupSignalV2]:
    """Local trend lost then reclaimed (full ribbon + lower-TF 200EMA), MSB-confirmed."""
    ts = substrate.trend_state
    direction = _direction(ts.local_trend_direction)
    if direction is None or len(context.df) < min_bars:
        return []
    if not _reclaimed_full_ribbon(direction, ts.price_vs_ribbon):
        return []
    if not lower_tf_reclaim:  # the one-TF-lower 200EMA reclaim (D3) — conjunction
        return []
    if not _recent_msb(context.msbs, len(context.df), direction, msb_lookback):
        return []
    ema36, price, atr = _latest(substrate.bands.slow), context.current_price, context.atr
    if _isnan(ema36) or _isnan(price) or _isnan(atr) or atr <= 0:
        return []
    if _htf_block(sr_levels, ema36, direction, atr, htf_res_tol):  # resistance too close
        return []
    hard = _protected_stop(context.swings, direction, ema36)
    if _isnan(hard):  # fail closed — no structural invalidation, not well-formed
        return []
    factors ={ConfluenceFactor.TREND_ALIGNED, ConfluenceFactor.LOWER_TF_CONFIRMED}
    factors |= _retest_factors(ema36, price, atr, sr_levels, sr_tol, retest_tol)
    return [SetupSignalV2(
        setup_type=SetupTypeV2.TREND_RECLAIM,
        direction=direction,
        basis_price=ema36,
        entry=ema36,
        hard_stop=hard,
        soft_stop=SoftStop(SoftStopMode.TREND_RECLAIM_AGAINST, ema36),
        confluence=frozenset(factors),
        weakness_flag=_weakness_flag(context.swings, direction),
    )]


def detect_deviation_reclaim(
    context: SetupContext,
    substrate: TFSubstrate,
    sr_levels: list[SRLevelV2],
    *,
    lookback: int = 10,
    sr_tol: float = 0.5,
    retest_tol: float = 0.5,
    min_bars: int = 40,
) -> list[SetupSignalV2]:
    """Price deviated through the band then reclaimed it at a v2 S/R level."""
    ts = substrate.trend_state
    direction = _direction(ts.local_trend_direction)
    if direction is None or len(context.df) < min_bars:
        return []
    if not _reclaimed_full_ribbon(direction, ts.price_vs_ribbon):  # reclaimed now
        return []
    if not _deviated_within_lookback(context.df, substrate.bands, direction, lookback):
        return []
    ema36, price, atr = _latest(substrate.bands.slow), context.current_price, context.atr
    if _isnan(ema36) or _isnan(price) or _isnan(atr) or atr <= 0:
        return []
    if not _near_sr(ema36, sr_levels, atr, sr_tol):  # reclaim must be S/R-supported
        return []
    hard = _protected_stop(context.swings, direction, ema36)
    if _isnan(hard):  # fail closed — no structural invalidation, not well-formed
        return []
    factors ={ConfluenceFactor.TREND_ALIGNED, ConfluenceFactor.AT_SR}
    if abs(price - ema36) <= retest_tol * atr:
        factors.add(ConfluenceFactor.RETEST)
    return [SetupSignalV2(
        setup_type=SetupTypeV2.DEVIATION_RECLAIM,
        direction=direction,
        basis_price=ema36,
        entry=ema36,
        hard_stop=hard,
        soft_stop=SoftStop(SoftStopMode.TREND_RECLAIM_AGAINST, ema36),
        confluence=frozenset(factors),
        weakness_flag=_weakness_flag(context.swings, direction),
    )]
