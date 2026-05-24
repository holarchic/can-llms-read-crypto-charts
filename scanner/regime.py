"""Market-regime classification for the v10 v2 framework (T-019 Chunk 4).

Reduces the Chunk-1/2 substrate (``TrendState``, ``LongerMASet``) plus an optional
volume profile to a typed ``RegimeState`` — the categorical market context the
Chunk-5 setup detectors read to weight setup-quality conviction. Pure functions;
no pipeline wiring, no LLM, no DB. Additive to v1: ``indicators.classify_regime``
(the string ``trending_up``/``ranging``/… classifier) is untouched.

Three regimes, from the framework's two frameworks plus a volume check:

- **F1 — trend break.** An abnormal change in trend dynamics: a recent structure
  break (an ``MSB`` inside ``msb_lookback`` bars) co-occurring with efficiency-ratio
  deterioration (``er < er_floor``). The MSB is the structural break; the sub-floor
  ER is the dynamics deterioration. Neither alone qualifies — a plain
  MA-disagreement proxy is too weak to mean "the trend's character changed".
- **F2 — gap-fill stall.** A *directional, counter-trend* reclaim toward the 100MA
  that stalls before reaching it. Gap-fills run counter to the dominant HTF trend,
  so the direction is set by ``htf_trend_direction``: in a downtrend, a bounce that
  reclaimed above the ribbon but is still below the 100MA and has gone flat; in an
  uptrend, the mirror. The 100MA is the gap-fill target, read from
  ``longer_set.sma_100``; "stalled before reaching it" means price sits between the
  ribbon and the un-reached 100MA with the local stack FLAT.
- **Range triangulation.** A structural ``RANGING`` whose volume value area sits
  *strictly inside* the structural bounds — the framework's confirmation that the range is a
  real rotation, not a mid-move pause. Needs a profile: ``vrvp=None`` makes
  ROTATIONAL unreachable by construction (a range can't be volume-validated
  without one).

The sub-signals (``f1_trend_break``, ``f2_gapfill_stall``, ``range_valid``) are
surfaced on ``RegimeState``, not just folded into the label, so the detectors and
tests read the *why*, not only the regime. F1 and F2 are disjoint-by-predicate from
TRENDING (F1 ⊆ ``{er < er_floor}``, F2 ⊆ ``{local == FLAT}``, both excluded by the
TRENDING condition), so they never contradict a TRENDING label — their value is
diagnostic and they are consumed directly by the Chunk-5 detectors.

Degradation (never raises). A frame shorter than ``max(min_bars, er_period + 1)``
short-circuits to TRANSITIONAL with all sub-flags ``False`` and ``htf = FLAT``
*before* any predicate runs: ``compute_efficiency_ratio(period=er_period)`` needs
``er_period + 1`` bars and returns ``0.0`` below that, so a warmup frame would
otherwise masquerade as a genuine low-ER F1 break. Empty / NaN substrate degrades
to neutral the same way. Thresholds are v1 defaults, calibrated in Chunk 8b.

Substrate-only: this consumes a ``VolumeProfile`` it is *handed*; it does not fetch
or anchor one. Wiring the range's pre-breakdown pivot to Chunk-3
``anchor_vrvp_for_range`` is deferred to Chunk 7 (pipeline).
"""

from enum import Enum
from typing import NamedTuple

import pandas as pd

from .indicators import (
    classify_structure,
    compute_efficiency_ratio,
    detect_msbs,
    find_swing_points,
)
from .ma_indicators import LongerMASet, VolumeProfile
from .models import Structure
from .trend_state import BandPosition, TrendDirection, TrendState

__all__ = ["Regime", "RegimeState", "detect_regime"]


class Regime(Enum):
    """The v10 market regime — the categorical context for setup conviction."""

    TRENDING = "trending"
    TRANSITIONAL = "transitional"
    ROTATIONAL = "rotational"


class RegimeState(NamedTuple):
    regime: Regime
    htf_trend_direction: TrendDirection  # the 100/200/300 bias (reused from trend_state)
    f1_trend_break: bool                 # recent MSB + ER deterioration
    f2_gapfill_stall: bool               # directional counter-trend reclaim that stalled
    range_valid: bool                    # RANGING with value area inside the structural bounds


def _latest(series: pd.Series) -> float:
    """Latest value of a series, or NaN if empty (never IndexError)."""
    return float(series.iloc[-1]) if len(series) else float("nan")


def _f1_trend_break(
    swings: list, df: pd.DataFrame, er: float, er_floor: float, msb_lookback: int,
) -> bool:
    """F1: a recent structure break co-occurring with ER deterioration.

    The ER drop is checked first (cheap, and the dominant gate); the MSB scan only
    runs when the dynamics have actually deteriorated. ``msb.index`` is the break
    bar, so "recent" = within ``msb_lookback`` bars of the latest bar.
    """
    if er >= er_floor:
        return False
    cutoff = len(df) - msb_lookback
    return any(msb.index >= cutoff for msb in detect_msbs(swings, df))


def _f2_gapfill_stall(
    trend_state: TrendState, sma_100: float, close: float, htf: TrendDirection,
) -> bool:
    """F2: a directional, counter-trend reclaim toward the 100MA that stalled.

    Direction is set by ``htf``; ``htf == FLAT`` has no dominant trend to fill
    against, so F2 does not apply. NaN substrate (no 100MA yet / empty close) → no
    stall.
    """
    if pd.isna(sma_100) or pd.isna(close) or htf == TrendDirection.FLAT:
        return False
    local = trend_state.local_trend_direction
    pos = trend_state.price_vs_ribbon
    if htf == TrendDirection.DOWN:
        # Bounce reclaimed above the ribbon but stalled below the 100MA.
        return pos == BandPosition.ABOVE and close < sma_100 and local == TrendDirection.FLAT
    # htf == UP: dip broke below the ribbon but stalled above the 100MA.
    return pos == BandPosition.BELOW and close > sma_100 and local == TrendDirection.FLAT


def _range_valid(
    structure: Structure, range_bounds: tuple[float, float] | None, vrvp: VolumeProfile | None,
) -> bool:
    """Range triangulation: the value area sits strictly inside the structural range.

    ``vrvp=None`` (or an all-NaN profile) → ``False``: a range can't be
    volume-validated without a profile, which keeps ROTATIONAL unreachable.
    """
    if structure != Structure.RANGING or range_bounds is None or vrvp is None:
        return False
    if pd.isna(vrvp.val) or pd.isna(vrvp.vah):
        return False
    range_low, range_high = range_bounds
    return vrvp.val > range_low and vrvp.vah < range_high


def detect_regime(
    df: pd.DataFrame,
    trend_state: TrendState,
    longer_set: LongerMASet,
    vrvp: VolumeProfile | None = None,
    *,
    er_period: int = 20,
    er_floor: float = 0.30,
    msb_lookback: int = 10,
    min_bars: int = 20,
) -> RegimeState:
    """Classify the market regime from the substrate + an optional volume profile.

    The decision tree is total and deterministic — every frame lands in exactly one
    regime — and records the sub-flags regardless of the branch taken:

    1. ``range_valid`` → ROTATIONAL.
    2. else ``f1_trend_break or f2_gapfill_stall`` → TRANSITIONAL.
    3. else aligned (``local == longer != FLAT``) and efficient (``er >= er_floor``)
       → TRENDING.
    4. else → TRANSITIONAL (disagreement / low efficiency / ambiguous default).
    """
    if len(df) < max(min_bars, er_period + 1):
        return RegimeState(Regime.TRANSITIONAL, TrendDirection.FLAT, False, False, False)

    htf = trend_state.longer_trend_direction
    swings = find_swing_points(df)
    structure, range_bounds = classify_structure(swings)
    er = compute_efficiency_ratio(df, er_period)

    f1 = _f1_trend_break(swings, df, er, er_floor, msb_lookback)
    f2 = _f2_gapfill_stall(trend_state, _latest(longer_set.sma_100), _latest(df["Close"]), htf)
    range_valid = _range_valid(structure, range_bounds, vrvp)

    if range_valid:
        regime = Regime.ROTATIONAL
    elif f1 or f2:
        regime = Regime.TRANSITIONAL
    elif trend_state.local_trend_direction == htf != TrendDirection.FLAT and er >= er_floor:
        regime = Regime.TRENDING
    else:
        regime = Regime.TRANSITIONAL

    return RegimeState(
        regime=regime,
        htf_trend_direction=htf,
        f1_trend_break=f1,
        f2_gapfill_stall=f2,
        range_valid=range_valid,
    )
