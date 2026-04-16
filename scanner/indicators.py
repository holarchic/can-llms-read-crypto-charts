"""Technical indicators: swing detection, S/R, market structure, MSBs, fibs."""

import pandas as pd

from .models import (
    MSB,
    FibLevels,
    MarketContext,
    SRLevel,
    Structure,
    SwingPoint,
)


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute Average True Range. Returns the latest ATR value."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_series = tr.rolling(window=period).mean()
    return float(atr_series.iloc[-1])


def compute_atr_percentile(df: pd.DataFrame, atr_period: int = 14, lookback: int = 100) -> float:
    """Current ATR as percentile of trailing ATR history. Returns 0-100."""
    clean = df[["High", "Low", "Close"]].dropna()
    if len(clean) < 2:
        return 50.0
    high = clean["High"]
    low = clean["Low"]
    close = clean["Close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_series = tr.rolling(window=atr_period).mean().dropna()
    if len(atr_series) < 2:
        return 50.0  # Not enough data — return neutral

    current_atr = atr_series.iloc[-1]
    trailing = atr_series.iloc[-lookback:] if len(atr_series) >= lookback else atr_series
    rank = (trailing < current_atr).sum()
    return round(float(rank / len(trailing)) * 100, 1)


def compute_efficiency_ratio(df: pd.DataFrame, period: int = 20) -> float:
    """Kaufman Efficiency Ratio. Returns 0.0-1.0.

    ER = abs(net_move) / sum(abs(bar_moves)) over `period` bars.
    Near 1.0 = clean trend, near 0.0 = chop.
    """
    if len(df) < period + 1:
        return 0.0

    close = df["Close"].dropna().iloc[-(period + 1):]
    if len(close) < period + 1:
        return 0.0
    net_move = abs(close.iloc[-1] - close.iloc[0])
    bar_moves = close.diff().abs().iloc[1:]  # drop first NaN
    total_move = bar_moves.sum()

    if total_move == 0:
        return 0.0

    er = float(net_move / total_move)
    return round(min(er, 1.0), 4)  # clamp to [0, 1]


def classify_regime(structure: Structure, vol_percentile: float, efficiency_ratio: float) -> str:
    """Combine indicators into regime label.

    | Regime         | Condition                                    |
    |---------------|----------------------------------------------|
    | trending_up   | ER >= 0.4 AND structure = BULLISH            |
    | trending_down | ER >= 0.4 AND structure = BEARISH            |
    | volatile      | vol_pct >= 75 AND ER < 0.4                   |
    | ranging       | everything else                              |
    """
    if efficiency_ratio >= 0.4 and structure == Structure.BULLISH:
        return "trending_up"
    if efficiency_ratio >= 0.4 and structure == Structure.BEARISH:
        return "trending_down"
    if vol_percentile >= 75 and efficiency_ratio < 0.4:
        return "volatile"
    return "ranging"


def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> list[SwingPoint]:
    """Find swing highs and lows using rolling window on body prices (preferred per ruleset).

    A swing high: body high is the max of ±lookback bars' body highs.
    A swing low: body low is the min of ±lookback bars' body lows.
    """
    # Body prices: max/min of open and close
    body_high = pd.concat([df["Open"], df["Close"]], axis=1).max(axis=1)
    body_low = pd.concat([df["Open"], df["Close"]], axis=1).min(axis=1)

    swings = []
    timestamps = df.index

    for i in range(lookback, len(df) - lookback):
        window_high = body_high.iloc[i - lookback: i + lookback + 1]
        if body_high.iloc[i] == window_high.max() and (window_high == body_high.iloc[i]).sum() == 1:
            swings.append(SwingPoint(
                index=i,
                timestamp=timestamps[i],
                price=float(df["High"].iloc[i]),
                type="high",
                body_price=float(body_high.iloc[i]),
            ))

        window_low = body_low.iloc[i - lookback: i + lookback + 1]
        if body_low.iloc[i] == window_low.min() and (window_low == body_low.iloc[i]).sum() == 1:
            swings.append(SwingPoint(
                index=i,
                timestamp=timestamps[i],
                price=float(df["Low"].iloc[i]),
                type="low",
                body_price=float(body_low.iloc[i]),
            ))

    swings.sort(key=lambda s: s.index)
    return swings


def cluster_sr_levels(
    swings: list[SwingPoint],
    pct: float = 0.015,
) -> list[SRLevel]:
    """Group swing points within pct% of each other into S/R levels.

    Each cluster becomes an SRLevel with strength = touch count.
    Levels with 3+ same-direction tests are marked degraded.
    """
    if not swings:
        return []

    prices = sorted([(s.body_price, s) for s in swings], key=lambda x: x[0])
    clusters: list[list[SwingPoint]] = []
    current_cluster = [prices[0][1]]
    current_anchor = prices[0][0]

    for price, swing in prices[1:]:
        if abs(price - current_anchor) / current_anchor <= pct:
            current_cluster.append(swing)
        else:
            clusters.append(current_cluster)
            current_cluster = [swing]
            current_anchor = price

    clusters.append(current_cluster)

    levels = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue

        avg_price = sum(s.body_price for s in cluster) / len(cluster)
        dir_tests = {"from_above": 0, "from_below": 0}
        for s in cluster:
            if s.type == "high":
                dir_tests["from_below"] += 1
            else:
                dir_tests["from_above"] += 1

        max_dir = max(dir_tests.values())
        degraded = max_dir >= 3

        timestamps = [s.timestamp for s in cluster]
        levels.append(SRLevel(
            price=round(avg_price, 6),
            strength=len(cluster),
            source="swing_cluster",
            first_seen=min(timestamps),
            last_tested=max(timestamps),
            direction_tests=dir_tests,
            degraded=degraded,
        ))

    levels.sort(key=lambda l: l.price)
    return levels


def classify_structure(swings: list[SwingPoint]) -> tuple[Structure, tuple[float, float] | None]:
    """Classify market structure from swing sequence.

    Uses majority-vote: if highs are predominantly LH and lows predominantly LL,
    that's BEARISH even with one counter-trend swing (e.g. a pullback bounce).
    Requires both highs and lows to agree on direction.

    Returns (structure, range_bounds). range_bounds is (low, high) if RANGING, else None.
    """
    highs = [s for s in swings if s.type == "high"]
    lows = [s for s in swings if s.type == "low"]

    if len(highs) < 2 or len(lows) < 2:
        return Structure.RANGING, None

    # Use the most recent swings for classification
    recent_highs = highs[-3:] if len(highs) >= 3 else highs[-2:]
    recent_lows = lows[-3:] if len(lows) >= 3 else lows[-2:]

    hh_count = sum(1 for i in range(1, len(recent_highs))
                   if recent_highs[i].body_price > recent_highs[i - 1].body_price)
    lh_count = sum(1 for i in range(1, len(recent_highs))
                   if recent_highs[i].body_price < recent_highs[i - 1].body_price)
    hl_count = sum(1 for i in range(1, len(recent_lows))
                   if recent_lows[i].body_price > recent_lows[i - 1].body_price)
    ll_count = sum(1 for i in range(1, len(recent_lows))
                   if recent_lows[i].body_price < recent_lows[i - 1].body_price)

    # Majority vote: both highs and lows must lean the same way
    highs_bullish = hh_count > lh_count
    highs_bearish = lh_count > hh_count
    lows_bullish = hl_count > ll_count
    lows_bearish = ll_count > hl_count

    if highs_bullish and lows_bullish:
        return Structure.BULLISH, None
    if highs_bearish and lows_bearish:
        return Structure.BEARISH, None

    # Tie-break: one side agrees unanimously + latest swing on tied side confirms
    highs_tied = hh_count == lh_count and hh_count > 0
    lows_tied = hl_count == ll_count and hl_count > 0

    if highs_tied and lows_bullish:
        if recent_highs[-1].body_price >= max(s.body_price for s in recent_highs):
            return Structure.BULLISH, None
    if highs_tied and lows_bearish:
        if recent_highs[-1].body_price <= min(s.body_price for s in recent_highs):
            return Structure.BEARISH, None
    if lows_tied and highs_bullish:
        if recent_lows[-1].body_price >= max(s.body_price for s in recent_lows):
            return Structure.BULLISH, None
    if lows_tied and highs_bearish:
        if recent_lows[-1].body_price <= min(s.body_price for s in recent_lows):
            return Structure.BEARISH, None

    # RANGING — compute bounds from recent swings
    all_recent = recent_highs + recent_lows
    range_low = min(s.body_price for s in all_recent)
    range_high = max(s.body_price for s in all_recent)
    return Structure.RANGING, (range_low, range_high)


def detect_msbs(swings: list[SwingPoint], df: pd.DataFrame) -> list[MSB]:
    """Detect Market Structure Breaks — price closes above a lower high or below a higher low."""
    msbs = []
    highs = [s for s in swings if s.type == "high"]
    lows = [s for s in swings if s.type == "low"]

    # Bullish MSB: price closes above a prior lower high
    for i in range(2, len(highs)):
        # Confirm this is actually a lower high (structure break, not continuation)
        if highs[i - 1].body_price < highs[i - 2].body_price:
            lh = highs[i - 1]
            # Check if any candle after this swing closes above it
            for j in range(lh.index + 1, min(lh.index + 30, len(df))):
                if df["Close"].iloc[j] > lh.body_price:
                    msbs.append(MSB(
                        index=j,
                        timestamp=df.index[j],
                        type="bullish",
                        level=lh.body_price,
                        swing_broken=lh,
                    ))
                    break

    # Bearish MSB: price closes below a prior higher low
    for i in range(2, len(lows)):
        # Confirm this is actually a higher low (structure break, not continuation)
        if lows[i - 1].body_price > lows[i - 2].body_price:
            hl = lows[i - 1]
            for j in range(hl.index + 1, min(hl.index + 30, len(df))):
                if df["Close"].iloc[j] < hl.body_price:
                    msbs.append(MSB(
                        index=j,
                        timestamp=df.index[j],
                        type="bearish",
                        level=hl.body_price,
                        swing_broken=hl,
                    ))
                    break

    msbs.sort(key=lambda m: m.index)
    return msbs


def calculate_fibs(swing_low: float, swing_high: float) -> FibLevels:
    """Calculate Fibonacci retracement levels. OTE zone = 0.618-0.786 retracement."""
    diff = swing_high - swing_low
    fib_618 = swing_high - diff * 0.618
    fib_786 = swing_high - diff * 0.786
    return FibLevels(
        swing_low=swing_low,
        swing_high=swing_high,
        fib_618=round(fib_618, 6),
        fib_786=round(fib_786, 6),
        ote_zone=(round(fib_618, 6), round(fib_786, 6)),
    )


def build_context(df: pd.DataFrame, lookback: int = 5, sr_pct: float = 0.015) -> MarketContext:
    """Build full market context from a DataFrame of candles."""
    swings = find_swing_points(df, lookback)
    sr_levels = cluster_sr_levels(swings, sr_pct)
    structure, range_bounds = classify_structure(swings)
    msbs = detect_msbs(swings, df)
    atr = compute_atr(df)
    current_price = float(df["Close"].iloc[-1])

    # Fib levels from the most recent significant swing
    fib_levels = None
    highs = [s for s in swings if s.type == "high"]
    lows = [s for s in swings if s.type == "low"]
    if highs and lows:
        recent_high = max(highs[-3:], key=lambda s: s.body_price) if len(highs) >= 3 else highs[-1]
        recent_low = min(lows[-3:], key=lambda s: s.body_price) if len(lows) >= 3 else lows[-1]
        if recent_high.body_price > recent_low.body_price:
            fib_levels = calculate_fibs(recent_low.body_price, recent_high.body_price)

    vol_pct = compute_atr_percentile(df)
    er = compute_efficiency_ratio(df)
    regime = classify_regime(structure, vol_pct, er)

    return MarketContext(
        structure=structure,
        swings=swings,
        sr_levels=sr_levels,
        msbs=msbs,
        range_bounds=range_bounds,
        atr=atr,
        current_price=current_price,
        fib_levels=fib_levels,
        vol_percentile=vol_pct,
        efficiency_ratio=er,
        regime=regime,
    )
