"""Compression early-exit monitor for the v10 v2 framework (T-019 Chunk 2).

the framework author's de-risk for acting BEFORE a local-trend band has fully compressed: it is
permitted iff this monitor trips — price breaks out of the compression AND the
local trend is *defended* on the breakout (the band re-fans in the breakout
direction and price holds beyond it for a confirming bar). Absent a trip, require
complete compression (MAs almost touching). Pure function; the held-bars/fan
criteria are v1 defaults calibrated in Chunk 8b.

The compression context is an explicit precondition: ``armed`` is the caller's
signal that a compression preceded this bar (the setup detector tracks it across
bars — at the breakout bar the band may already be re-fanning, so the monitor
cannot infer it from the current bar alone). Not armed → never trips; this keeps
the monitor from degrading into a generic breakout detector. Stack-direction
classification is delegated to ``trend_state.stack_direction`` (single owner).
"""

from typing import NamedTuple

import pandas as pd

from .ma_indicators import EMABand
from .trend_state import TrendDirection, stack_direction


class CompressionExit(NamedTuple):
    tripped: bool
    direction: TrendDirection  # breakout side when tripped, else FLAT


def evaluate_compression_exit(df: pd.DataFrame, fast_band: EMABand, armed: bool) -> CompressionExit:
    """Trip iff armed AND price broke out of the ribbon, held ≥2 bars, band re-fanned.

    ``armed`` = a compression context preceded this bar (caller-supplied).
    "Defended" = the band re-fans in the breakout direction (``stack_direction`` UP
    for an upside break / DOWN for a downside break) AND the breakout close held for
    the prior bar too — a single-bar poke the band hasn't followed is not defended.
    """
    if not armed or len(df) < 2 or min(len(fast_band.fast), len(fast_band.mid), len(fast_band.slow)) < 2:
        return CompressionExit(False, TrendDirection.FLAT)  # need ≥2 aligned bars to read a breakout

    f0, m0, s0 = (fast_band.fast.iloc[-1], fast_band.mid.iloc[-1], fast_band.slow.iloc[-1])
    f1, m1, s1 = (fast_band.fast.iloc[-2], fast_band.mid.iloc[-2], fast_band.slow.iloc[-2])
    if any(pd.isna(v) for v in (f0, m0, s0, f1, m1, s1)):
        return CompressionExit(False, TrendDirection.FLAT)

    close_now = float(df["Close"].iloc[-1])
    close_prev = float(df["Close"].iloc[-2])
    high_now, low_now = max(f0, m0, s0), min(f0, m0, s0)
    high_prev, low_prev = max(f1, m1, s1), min(f1, m1, s1)

    broke_up = close_now > high_now and close_prev > high_prev
    broke_down = close_now < low_now and close_prev < low_prev
    direction = stack_direction(f0, m0, s0)

    if broke_up and direction is TrendDirection.UP:
        return CompressionExit(True, TrendDirection.UP)
    if broke_down and direction is TrendDirection.DOWN:
        return CompressionExit(True, TrendDirection.DOWN)
    return CompressionExit(False, TrendDirection.FLAT)
