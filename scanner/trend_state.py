"""Derived MA-relative trend state for the v10 v2 framework (T-019 Chunk 2).

Consumes the Chunk-1 indicator substrate (``EMABand``, ``LongerMASet``) and
reduces it to a typed ``TrendState`` for the setup detectors (chunks 4-5, not yet
built) to read instead of re-deriving band relationships. Pure functions; no I/O.

Compression is a STATE detector, not a tightness number (decision 2026-05-16,
option a): the gate is the band's SHAPE — the local-trend EMAs converging (band
width contracting over a window) while their slopes no longer share a sign (they
stop fanning in one direction), i.e. a rounded convergence. ``width/ATR`` is kept
only as the soft annotation ``compression_tightness``, NEVER the act/wait gate.
The early-exit de-risk lives in ``compression_exit`` (the same chunk).

No length-based crash. Every band read goes through the total accessors
``_at`` / ``_latest`` / ``_slope`` (NaN when the requested bar is unavailable), so
a bad length yields NaN rather than an IndexError. A field degrades to neutral
(FLAT / INSIDE / not-compressed / NaN) only where its read is unavailable —
available latest values still produce real output. ``EMABand`` / ``LongerMASet``
are expected aligned (their constructors ``compute_ema_band`` /
``compute_longer_set`` return equal-length series).
Window thresholds are v1 defaults, calibrated in Chunk 8b.
"""

from enum import Enum
from typing import NamedTuple

import pandas as pd

from .indicators import compute_atr
from .ma_indicators import EMABand, LongerMASet


class TrendDirection(Enum):
    """Categorical trend (per the framework 2026-02-25: HTF trend is a discrete bias)."""

    UP = "up"
    DOWN = "down"
    FLAT = "flat"


class BandPosition(Enum):
    """Price relative to an MA band/envelope."""

    ABOVE = "above"
    BELOW = "below"
    INSIDE = "inside"


class TrendState(NamedTuple):
    local_trend_direction: TrendDirection  # the 13/25/36 ribbon stack
    price_vs_ribbon: BandPosition
    ribbon_compressed: bool                # the shape-based STATE predicate
    longer_trend_direction: TrendDirection  # the 100/200/300 stack
    price_vs_longer: BandPosition
    gap_to_next_ma: float                  # ATR-normalized distance to nearest longer MA
    compression_tightness: float           # width/ATR — soft annotation, NOT the gate


def stack_direction(top: float, mid: float, bot: float) -> TrendDirection:
    """UP when strictly fanned up (faster MA on top), DOWN when fanned down.

    A mixed/crossed stack or any NaN (insufficient history) → FLAT.
    """
    if any(pd.isna(v) for v in (top, mid, bot)):
        return TrendDirection.FLAT
    if top > mid > bot:
        return TrendDirection.UP
    if top < mid < bot:
        return TrendDirection.DOWN
    return TrendDirection.FLAT


def _band_bounds(*values: float) -> tuple[float, float]:
    """(low, high) of a band; (NaN, NaN) if any member is NaN (insufficient data)."""
    if any(pd.isna(v) for v in values):
        return (float("nan"), float("nan"))
    return (min(values), max(values))


def _position(price: float, low: float, high: float) -> BandPosition:
    if pd.isna(low) or pd.isna(high):
        return BandPosition.INSIDE  # insufficient data → neutral
    if price > high:
        return BandPosition.ABOVE
    if price < low:
        return BandPosition.BELOW
    return BandPosition.INSIDE


def _at(series: pd.Series, offset: int = 0) -> float:
    """Value ``offset`` bars from the end (0 = latest); NaN if unavailable.

    The total accessor for point reads — used by ``_latest`` (offset 0) and the
    now/past band reads in ``_is_compressed``. An empty/short/misaligned series
    yields NaN instead of an IndexError. (``_slope`` keeps its own length-guarded
    two-point read.)
    """
    return float(series.iloc[-1 - offset]) if len(series) > offset else float("nan")


def _latest(series: pd.Series) -> float:
    """Latest value, or NaN if the series is empty."""
    return _at(series, 0)


def _slope(series: pd.Series, window: int) -> float:
    """Latest minus `window`-bars-ago (sign = slope direction); NaN if too short."""
    if len(series) <= window:
        return float("nan")
    return float(series.iloc[-1] - series.iloc[-1 - window])


def _is_compressed(band: EMABand, contraction_window: int = 5, slope_window: int = 3) -> bool:
    """The shape-based compression predicate: band contracting AND slopes disagree.

    Contracting = the ribbon (max-min of the three EMAs) is narrower than
    ``contraction_window`` bars ago. Slope-disagreement = the three EMAs' slopes
    over ``slope_window`` do NOT all share a sign — a fanning (strong-trend) stack
    is all-same-sign and is NOT compression; a rounding/flattening convergence is.
    Reads via ``_at``/``_slope``, so a warmup / empty / short / misaligned band
    yields NaN → not compressed (never an IndexError).
    """
    f, m, s = band.fast, band.mid, band.slow
    now = _band_bounds(_at(f, 0), _at(m, 0), _at(s, 0))
    past = _band_bounds(_at(f, contraction_window), _at(m, contraction_window), _at(s, contraction_window))
    if pd.isna(now[0]) or pd.isna(past[0]):
        return False
    contracting = (now[1] - now[0]) < (past[1] - past[0])

    slopes = [_slope(series, slope_window) for series in (f, m, s)]
    if any(pd.isna(sl) for sl in slopes):
        return False
    fanning = all(sl > 0 for sl in slopes) or all(sl < 0 for sl in slopes)
    return bool(contracting and not fanning)  # Python bool, not np.bool_


def _gap_to_next_ma(price: float, longer_values: list[float], atr: float) -> float:
    """ATR-normalized distance from price to the nearest longer-set MA.

    Directional refinement (only MAs on the gap-fill side) is the ``gap_fill``
    detector's job (Chunk 5b); here it is the raw proximity feature.
    """
    valid = [v for v in longer_values if not pd.isna(v)]
    if not valid or pd.isna(atr) or atr == 0:
        return float("nan")
    nearest = min(valid, key=lambda v: abs(price - v))
    return float(abs(price - nearest) / atr)


def derive_trend_state(df: pd.DataFrame, fast_band: EMABand, longer_set: LongerMASet) -> TrendState:
    """Reduce the precomputed bands + latest price/ATR to a typed ``TrendState``.

    No length-based crash. A field degrades to neutral/NaN only where its band read
    is unavailable (warmup → NaN band latest → neutral field); an available latest
    value (e.g. a short but non-empty ``df``'s close) still produces real output.
    """
    price = _latest(df["Close"])
    atr = compute_atr(df) if not df.empty else float("nan")

    f, m, s = (_latest(fast_band.fast), _latest(fast_band.mid), _latest(fast_band.slow))
    ribbon_low, ribbon_high = _band_bounds(f, m, s)

    longer = [_latest(longer_set.sma_100), _latest(longer_set.ema_200), _latest(longer_set.sma_300)]
    longer_low, longer_high = _band_bounds(*longer)

    compression_tightness = (
        float((ribbon_high - ribbon_low) / atr)
        if not (pd.isna(ribbon_low) or pd.isna(atr) or atr == 0)
        else float("nan")
    )

    return TrendState(
        local_trend_direction=stack_direction(f, m, s),
        price_vs_ribbon=_position(price, ribbon_low, ribbon_high),
        ribbon_compressed=_is_compressed(fast_band),
        longer_trend_direction=stack_direction(*longer),
        price_vs_longer=_position(price, longer_low, longer_high),
        gap_to_next_ma=_gap_to_next_ma(price, longer, atr),
        compression_tightness=compression_tightness,
    )
