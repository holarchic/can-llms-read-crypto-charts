"""v10 the framework setup detectors (T-019 Chunk 5) over the Chunk-1/2/3/4 substrate.

Six pure detectors returning 0-N ``SetupSignalV2`` (in ``models``); no pipeline
wiring, no LLM, no DB. The trend family is anchored on the local trend (EMA 13/25/36)
as both basis AND invalidation (the framework 2026-01-13):

- ``detect_trend_continuation`` / ``detect_trend_reclaim`` / ``detect_deviation_reclaim``.

The rotational / counter-trend family reuses the same bundles + signal contract:

- ``detect_gap_fill`` — counter-trend reclaim toward the 100MA (geometry AND
  (RSI divergence OR the regime F2 stall)).
- ``detect_sfp_range`` — a swing-failure sweep of a significant level (an S/R level
  or external liquidity beyond the structural range) with a close-back + reaction.
- ``detect_rsi_divergence_range`` — a price/RSI divergence (no 50-reset) in a
  rotational / transitional regime.

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
from .ma_indicators import EMABand, LongerMASet, compute_rsi
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
    "detect_gap_fill",
    "detect_sfp_range",
    "detect_rsi_divergence_range",
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
    msb_lookback: int = 25,
    min_bars: int = 40,
) -> list[SetupSignalV2]:
    """Local trend lost then reclaimed (full ribbon + lower-TF 200EMA), MSB-confirmed.

    ``msb_lookback`` is deliberately wider than the F1/regime window (25, not 10): the
    confirming structure break registers when price first closes through a lower high,
    but the local 13/25/36 stack only *fans* back up — the reclaim this detector keys
    on — some bars later (the EMA-fanning latency), so by the time the reclaim is
    visible the break is already ~15-25 bars old. A 10-bar window would make
    trend_reclaim near-dead on real data. Calibrated in Chunk 8b.
    """
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


# --- Chunk 5b: rotational / counter-trend detectors ---------------------------


def _gap_fill_geometry(
    substrate: TFSubstrate, close: float, atr: float, *, min_gap: float,
) -> tuple[bool, Direction | None]:
    """Counter-trend reclaim toward the 100MA with a real remaining gap to fill.

    Direction is set by the dominant HTF trend (the fill runs against it): in a
    downtrend a bounce that reclaimed above the ribbon but is still below the 100MA
    (a LONG fill); in an uptrend the mirror (SHORT). ``htf == FLAT`` → no fill.
    """
    ts = substrate.trend_state
    htf = substrate.regime.htf_trend_direction
    sma_100 = _latest(substrate.longer.sma_100)
    if pd.isna(sma_100) or htf is TrendDirection.FLAT:
        return (False, None)
    pos = ts.price_vs_ribbon
    if htf is TrendDirection.DOWN:
        if pos is BandPosition.ABOVE and close < sma_100 and (sma_100 - close) / atr >= min_gap:
            return (True, Direction.LONG)
    elif htf is TrendDirection.UP:
        if pos is BandPosition.BELOW and close > sma_100 and (close - sma_100) / atr >= min_gap:
            return (True, Direction.SHORT)
    return (False, None)


def _rsi_divergence(
    df: pd.DataFrame, swings: list[SwingPoint], *, rsi_len: int, max_gap: int,
    want: Direction | None = None,
) -> tuple[Direction, SwingPoint, float] | None:
    """Price/RSI divergence on the last two confirmed same-type swings.

    Bearish (→ SHORT): price prints a higher high while RSI prints a *lower* high (the
    fading momentum IS the divergence), the swings are ≤ ``max_gap`` bars apart, and
    RSI did NOT reset back down through 50 between them. Bullish (→ LONG) is the mirror
    (lower low / higher RSI low / no upside 50-reset). Returns ``(direction,
    reacting_swing, reference_rsi)`` — the reference swing's RSI lets the caller decide
    the overbought/oversold confluence (a corroborating factor, NOT an emit gate: the
    reacting peak of a real divergence has fading, i.e. sub-extreme, RSI, and a
    structural range mechanically caps the 14-period RSI well below 70).

    Both directions are evaluated; ``want`` restricts to one (gap_fill asks for its
    geometry direction), else the candidate with the most-recent reacting swing wins
    (so a stale opposite-side pair never pre-empts a fresher one).
    """
    rsi = compute_rsi(df["Close"], rsi_len)
    highs = [s for s in swings if s.type == "high"]
    lows = [s for s in swings if s.type == "low"]
    cands: list[tuple[int, Direction, SwingPoint, float]] = []
    if len(highs) >= 2 and want in (None, Direction.SHORT):
        a, b = highs[-2], highs[-1]
        ra, rb = float(rsi.iloc[a.index]), float(rsi.iloc[b.index])
        if (not pd.isna(ra) and not pd.isna(rb) and b.index - a.index <= max_gap
                and b.body_price > a.body_price and rb < ra
                and float(rsi.iloc[a.index:b.index + 1].min()) >= 50.0):
            cands.append((b.index, Direction.SHORT, b, ra))
    if len(lows) >= 2 and want in (None, Direction.LONG):
        a, b = lows[-2], lows[-1]
        ra, rb = float(rsi.iloc[a.index]), float(rsi.iloc[b.index])
        if (not pd.isna(ra) and not pd.isna(rb) and b.index - a.index <= max_gap
                and b.body_price < a.body_price and rb > ra
                and float(rsi.iloc[a.index:b.index + 1].max()) <= 50.0):
            cands.append((b.index, Direction.LONG, b, ra))
    if not cands:
        return None
    _, direction, swing, ref = max(cands, key=lambda c: c[0])  # most-recent reacting swing
    return (direction, swing, ref)


def _significant_level(
    close_i: float, wick: float, side: str, sr_levels: list[SRLevelV2],
    range_bounds: tuple[float, float] | None,
) -> tuple[float | None, bool]:
    """The significant level a bar *pierced* — close on one side, wick beyond.

    A short sweep straddles a level ``L`` with ``close < L < wick``; a long sweep with
    ``wick < L < close``. Branch (a): an ``sr_levels`` member so pierced (``at_sr``
    True, the closest-to-the-wick one). Branch (b): external liquidity — the structural
    ``range_bounds`` extreme pierced (``at_sr`` False). A mid-range bar that pierces
    neither → ``(None, False)``.
    """
    if side == "short":
        pierced = [lvl.price for lvl in sr_levels if close_i < lvl.price < wick]
        if pierced:
            return (max(pierced), True)
        if range_bounds is not None and close_i < range_bounds[1] < wick:
            return (range_bounds[1], False)
    else:
        pierced = [lvl.price for lvl in sr_levels if wick < lvl.price < close_i]
        if pierced:
            return (min(pierced), True)
        if range_bounds is not None and wick < range_bounds[0] < close_i:
            return (range_bounds[0], False)
    return (None, False)


def _sweep(
    df: pd.DataFrame, sr_levels: list[SRLevelV2], range_bounds: tuple[float, float] | None,
    *, reaction_window: int, max_age: int,
) -> tuple[Direction, float, float, bool] | None:
    """Most recent significant-level sweep with a close-back and a reaction.

    A sweep bar pierces a significant level with its wick but closes back through it
    (``_significant_level``); the following ``reaction_window`` bars must stay on the
    reclaimed side (price continued away). Returns ``(direction, swept_level,
    sweep_wick_extreme, at_sr)`` or ``None``. Scans only recent bars (within
    ``max_age``) that leave room for the reaction window.
    """
    n = len(df)
    high, low, close = df["High"], df["Low"], df["Close"]
    for i in range(n - 1 - reaction_window, max(-1, n - 2 - max_age), -1):  # inclusive of age == max_age
        ci, hi, lo = float(close.iloc[i]), float(high.iloc[i]), float(low.iloc[i])
        level, at_sr = _significant_level(ci, hi, "short", sr_levels, range_bounds)
        if level is not None and all(close.iloc[j] < level for j in range(i + 1, i + 1 + reaction_window)):
            return (Direction.SHORT, level, hi, at_sr)
        level, at_sr = _significant_level(ci, lo, "long", sr_levels, range_bounds)
        if level is not None and all(close.iloc[j] > level for j in range(i + 1, i + 1 + reaction_window)):
            return (Direction.LONG, level, lo, at_sr)
    return None


def detect_gap_fill(
    context: SetupContext,
    substrate: TFSubstrate,
    sr_levels: list[SRLevelV2],
    *,
    min_gap: float = 1.0,
    rsi_len: int = 14,
    div_max_gap: int = 14,
    sr_tol: float = 0.5,
    min_bars: int = 40,
) -> list[SetupSignalV2]:
    """Counter-trend gap-fill toward the 100MA: geometry AND (divergence OR F2 stall)."""
    if len(context.df) < min_bars:
        return []
    price, atr = context.current_price, context.atr
    if _isnan(price) or _isnan(atr) or atr <= 0:
        return []
    geom_ok, direction = _gap_fill_geometry(substrate, price, atr, min_gap=min_gap)
    if not geom_ok:
        return []
    stall = substrate.regime.f2_gapfill_stall
    div = _rsi_divergence(context.df, context.swings, rsi_len=rsi_len, max_gap=div_max_gap, want=direction)
    has_div = div is not None
    if not (has_div or stall):
        return []
    ema36 = _latest(substrate.bands.slow)
    if _isnan(ema36):
        return []
    hard = _protected_stop(context.swings, direction, price)
    if _isnan(hard):
        return []
    factors = {ConfluenceFactor.COUNTER_TREND}
    if has_div:
        factors.add(ConfluenceFactor.DIVERGENCE)
    if stall:
        factors.add(ConfluenceFactor.GAPFILL_STALL)
    if _near_sr(price, sr_levels, atr, sr_tol):
        factors.add(ConfluenceFactor.AT_SR)
    return [SetupSignalV2(
        setup_type=SetupTypeV2.GAP_FILL,
        direction=direction,
        basis_price=ema36,
        entry=price,
        hard_stop=hard,
        soft_stop=SoftStop(SoftStopMode.TREND_RECLAIM_AGAINST, ema36),
        confluence=frozenset(factors),
        weakness_flag=_weakness_flag(context.swings, direction),
    )]


def detect_sfp_range(
    context: SetupContext,
    substrate: TFSubstrate,
    sr_levels: list[SRLevelV2],
    *,
    reaction_window: int = 2,
    max_age: int = 8,
    min_bars: int = 40,
) -> list[SetupSignalV2]:
    """Swing-failure pattern at a significant level (sr OR external liquidity)."""
    if len(context.df) < min_bars:
        return []
    atr = context.atr
    if _isnan(atr) or atr <= 0:
        return []
    if substrate.regime.regime not in (Regime.ROTATIONAL, Regime.TRANSITIONAL):
        return []
    swept = _sweep(context.df, sr_levels, context.range_bounds,
                   reaction_window=reaction_window, max_age=max_age)
    if swept is None:
        return []
    direction, level, extreme, at_sr = swept
    factors = {ConfluenceFactor.SWEEP_RECLAIM}
    if at_sr:
        factors.add(ConfluenceFactor.AT_SR)
    return [SetupSignalV2(
        setup_type=SetupTypeV2.SFP_RANGE,
        direction=direction,
        basis_price=level,
        entry=level,
        hard_stop=extreme,  # beyond the sweep-bar wick
        soft_stop=SoftStop(SoftStopMode.LEVEL_RECLAIM_AGAINST, level),
        confluence=frozenset(factors),
        weakness_flag=_weakness_flag(context.swings, direction),
    )]


def detect_rsi_divergence_range(
    context: SetupContext,
    substrate: TFSubstrate,
    sr_levels: list[SRLevelV2],
    *,
    rsi_len: int = 14,
    div_max_gap: int = 14,
    rsi_hi: float = 70.0,
    rsi_lo: float = 30.0,
    sr_tol: float = 0.5,
    min_bars: int = 40,
) -> list[SetupSignalV2]:
    """RSI divergence at a range extreme in a rotational/transitional regime."""
    if len(context.df) < min_bars:
        return []
    price, atr = context.current_price, context.atr
    if _isnan(price) or _isnan(atr) or atr <= 0:
        return []
    if substrate.regime.regime not in (Regime.ROTATIONAL, Regime.TRANSITIONAL):
        return []
    div = _rsi_divergence(context.df, context.swings, rsi_len=rsi_len, max_gap=div_max_gap)
    if div is None:
        return []
    direction, swing, ref_rsi = div
    # Reject a stale divergence: price must not have run past the reacting swing's
    # wick (else the hard stop would sit on the wrong side of entry — the divergence
    # is already invalidated).
    if direction is Direction.SHORT and not swing.price > price:
        return []
    if direction is Direction.LONG and not swing.price < price:
        return []
    # The rotation context is the regime gate above (not TRENDING); the divergence
    # swing is itself a local extreme, so no separate classify_structure==RANGING gate
    # — that requirement is jointly infeasible with the no-50-reset rule (a RANGING
    # structure's deep lows drive RSI through 50 between the swings). Matches the
    # parent rule, which scopes the divergence by candles/thresholds/50-reset, not by
    # structural range bounds.
    factors = {ConfluenceFactor.DIVERGENCE}
    overbought = direction is Direction.SHORT and ref_rsi > rsi_hi
    oversold = direction is Direction.LONG and ref_rsi < rsi_lo
    if overbought or oversold:  # corroborating extreme, not a gate
        factors.add(ConfluenceFactor.OVERSOLD_OVERBOUGHT)
    if _near_sr(swing.body_price, sr_levels, atr, sr_tol):
        factors.add(ConfluenceFactor.AT_SR)
    return [SetupSignalV2(
        setup_type=SetupTypeV2.RSI_DIVERGENCE_RANGE,
        direction=direction,
        basis_price=swing.body_price,
        entry=price,
        hard_stop=swing.price,  # beyond the reacting swing's wick extreme
        soft_stop=SoftStop(SoftStopMode.LEVEL_RECLAIM_AGAINST, swing.body_price),  # soft at the basis (body), hard at the wick
        confluence=frozenset(factors),
        weakness_flag=_weakness_flag(context.swings, direction),
    )]
