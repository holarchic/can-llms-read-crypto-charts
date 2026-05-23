"""MA-relative indicator substrate for the v10 v2 framework (T-019 Chunk 1).

Pure functions over OHLCV DataFrames (columns ``Open/High/Low/Close/Volume`` —
the project convention from ``scanner/data.py``). No pipeline wiring, no DB, no
I/O — the indicator building block the later v10 detection layers (trend state,
regime, top-down S/R, setup detection) build on.

Conventions (load-bearing):

- **EMA/RSI use SMA-seeded warmup** (ta-lib / Wilder convention, the standard-
  library default): the first emitted value at index ``period-1`` is the SMA of
  the first ``period`` bars, then the recursive update runs. The first
  ``period-1`` values are NaN. This is the convention every mainstream TA library
  shares; it differs from a first-observation seed (pandas ``ewm`` default) only
  across the warmup, decaying as ``(1-alpha)^n``. Downstream chunks read the
  *converged* latest values, where the seed choice is immaterial — but it is
  pinned here so the indicator surface is reproducible against ta-lib/TradingView.
- **RSI = Wilder smoothing**: ``RS = RMA(gain)/RMA(loss)`` with ``RMA`` the
  SMA-seeded recursion at ``alpha = 1/length`` (shares ``_seeded_ema``), seeded
  from the first ``length`` *real* price changes — the first RSI emits at index
  ``length`` (ta-lib alignment).

Insufficient data is not an error: warmup bars are NaN, matching the rest of the
indicator layer (``scanner/indicators.py``).

Malformed-input behavior differs by indicator KIND, deliberately — there is no
single "one NaN contract", and the substrate does NOT assume an upstream
validator (``scanner/data.py`` reshapes candles into a frame but does not reject
malformed ones; clean candles are the operational expectation, not an enforced
invariant):

- **Recursive indicators** (EMA, RSI, Stoch-RSI) propagate an interior NaN
  *permanently* from the gap onward — the recursion carries it forward (RSI uses
  ``.clip``, not ``.where(.., 0)``, to keep the interior NaN rather than fabricate
  a 0-delta).
- **Rolling-window indicators** (SMA, relative-volume) emit NaN only while the
  malformed row is inside the window, then recover once it ages out.
  (``compute_longer_set`` mixes both: its SMAs recover, its EMA does not.)
  Either way a gap surfaces as NaN — never a plausible-but-wrong number.
- **Binned summary** (``compute_volume_profile``) drops NaN typical-price/volume
  rows and profiles the valid remainder — a deliberate robustness choice for a
  window summary (one gap must not blank range detection); it returns an all-NaN
  ``VolumeProfile`` only when no valid rows remain. It drops rather than
  propagates because a NaN row would otherwise misbin its volume into an edge bin
  — itself a plausible-but-wrong level.

Compound results are named records
(``EMABand``/``LongerMASet``/``StochRSI``/``VolumeProfile``) so callers select by
field, never by tuple position.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd


class EMABand(NamedTuple):
    """The local-trend "ribbon" — three EMAs (13/25/36 by default)."""

    fast: pd.Series
    mid: pd.Series
    slow: pd.Series  # the 36 EMA — the framework's RSI-50 proxy, the band's slow anchor


class LongerMASet(NamedTuple):
    """HTF-on-LTF set; the SMA/EMA/SMA type mix is deliberate (the framework's script)."""

    sma_100: pd.Series
    ema_200: pd.Series
    sma_300: pd.Series


class StochRSI(NamedTuple):
    """Stochastic-RSI %K and %D lines (0-100)."""

    k: pd.Series
    d: pd.Series


class VolumeProfile(NamedTuple):
    """Value-area high/low and point-of-control for a fixed window."""

    vah: float
    val: float
    poc: float


def _seeded_ema(series: pd.Series, alpha: float, seed_period: int) -> pd.Series:
    """Recursive EMA seeded with the SMA of the first ``seed_period`` values.

    ta-lib / Wilder convention: output is NaN until index ``seed_period-1`` (the
    SMA seed), then ``ema[i] = alpha*x[i] + (1-alpha)*ema[i-1]``. Assumes a clean
    (NaN-free) input over the warmup window.
    """
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    n = len(series)
    if n < seed_period:
        return out
    values = series.to_numpy(dtype="float64")
    prev = float(values[:seed_period].mean())
    out.iloc[seed_period - 1] = prev
    for i in range(seed_period, n):
        prev = alpha * values[i] + (1.0 - alpha) * prev
        out.iloc[i] = prev
    return out


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    """EMA, SMA-seeded (ta-lib/standard convention); first ``period``-1 = NaN."""
    return _seeded_ema(close, alpha=2.0 / (period + 1.0), seed_period=period)


def compute_sma(close: pd.Series, period: int) -> pd.Series:
    """SMA; the first ``period``-1 values are NaN."""
    return close.rolling(window=period).mean()


def compute_ema_band(close: pd.Series, fast: int = 13, mid: int = 25, slow: int = 36) -> EMABand:
    """Local-trend "ribbon": the (fast, mid, slow) EMAs — 13/25/36 by default."""
    return EMABand(compute_ema(close, fast), compute_ema(close, mid), compute_ema(close, slow))


def compute_longer_set(close: pd.Series) -> LongerMASet:
    """HTF-on-LTF set: SMA 100, EMA 200, SMA 300 (the deliberate MA-type mix)."""
    return LongerMASet(compute_sma(close, 100), compute_ema(close, 200), compute_sma(close, 300))


def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder RSI (0-100); RMA seeded from the first ``length`` real price changes.

    First RSI emits at index ``length`` (ta-lib alignment). All-gain windows → 100
    (``RS`` is +inf, handled naturally); a fully-flat window yields NaN (RSI
    genuinely undefined); an interior NaN close propagates as NaN.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Drop the leading diff() NaN and seed the RMA over the first `length` REAL
    # deltas. Seeding across the synthetic index-0 slot would shift the warmup one
    # bar early; `.clip` (not `.where(..,0)`) keeps interior NaN as NaN so a gap
    # propagates instead of becoming a fake 0-delta.
    avg_gain = _seeded_ema(gain.iloc[1:], alpha=1.0 / length, seed_period=length)
    avg_loss = _seeded_ema(loss.iloc[1:], alpha=1.0 / length, seed_period=length)
    rs = avg_gain / avg_loss
    return (100.0 - 100.0 / (1.0 + rs)).reindex(close.index)


def compute_stoch_rsi(close: pd.Series, length: int = 14, k: int = 3, d: int = 3) -> StochRSI:
    """Stochastic RSI %K and %D, 0-100 (TradingView default 3/3/14/14).

    ``length`` is both the RSI length and the stochastic lookback. A flat RSI
    window (zero range) yields NaN — no oscillator signal to read.
    """
    rsi = compute_rsi(close, length)
    low = rsi.rolling(window=length).min()
    high = rsi.rolling(window=length).max()
    stoch = 100.0 * (rsi - low) / (high - low)
    k_line = stoch.rolling(window=k).mean()
    d_line = k_line.rolling(window=d).mean()
    return StochRSI(k_line, d_line)


def compute_volume_profile(
    df: pd.DataFrame, bars: int = 60, n_bins: int = 24, value_area_pct: float = 0.70,
) -> VolumeProfile:
    """Fixed volume profile over the last ``bars`` rows → ``VolumeProfile``.

    Each bar's volume is assigned to the price bin holding its typical price
    (HLC3); POC is the centre of the highest-volume bin; the value area grows
    outward from the POC (taking the heavier adjacent bin each step) until it
    holds ``value_area_pct`` of total volume. VAH/VAL are that area's edges.

    Rows with a NaN typical price or volume are dropped before binning — a
    malformed OHLCV row must not silently misbin its volume into an edge bin.
    ``n_bins`` (24) sets the profile's price resolution — a reasonable default
    for the H4/D1 ranges this runs on. Used for range validation only, never as a
    primary entry trigger.
    """
    window = df.iloc[-bars:]
    typical = (window["High"] + window["Low"] + window["Close"]) / 3.0
    volume = window["Volume"]
    valid = typical.notna() & volume.notna()
    typical, volume = typical[valid], volume[valid]
    if typical.empty:
        return VolumeProfile(float("nan"), float("nan"), float("nan"))

    prices = typical.to_numpy()
    vols = volume.to_numpy()
    lo, hi = float(prices.min()), float(prices.max())
    if hi == lo:  # single traded price across the window — degenerate but valid
        return VolumeProfile(hi, lo, hi)

    edges = np.linspace(lo, hi, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    bin_idx = np.clip(np.digitize(prices, edges) - 1, 0, n_bins - 1)
    bin_vol = np.zeros(n_bins)
    np.add.at(bin_vol, bin_idx, vols)

    poc_bin = int(bin_vol.argmax())
    target = bin_vol.sum() * value_area_pct
    lo_b = hi_b = poc_bin
    captured = bin_vol[poc_bin]
    while captured < target and (lo_b > 0 or hi_b < n_bins - 1):
        below = bin_vol[lo_b - 1] if lo_b > 0 else -1.0
        above = bin_vol[hi_b + 1] if hi_b < n_bins - 1 else -1.0
        if above >= below:
            hi_b += 1
            captured += bin_vol[hi_b]
        else:
            lo_b -= 1
            captured += bin_vol[lo_b]

    return VolumeProfile(float(edges[hi_b + 1]), float(edges[lo_b]), float(centres[poc_bin]))


def compute_relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """Relative volume: each bar's volume / its trailing ``window``-bar average.

    >1.0 = above-average participation. v1 stand-in for the framework's "Volume Suite by
    Leviathan" relative-volume candles. Takes the Volume series directly (column
    extraction stays at the caller boundary).
    """
    return volume / volume.rolling(window=window).mean()
