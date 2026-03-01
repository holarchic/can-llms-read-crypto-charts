"""Setup detection A-G from UNIFIED_TA_RULESET. Sensitive — flag anything plausible, let LLM decide."""

from collections import Counter

import pandas as pd

from .models import (
    Direction,
    MarketContext,
    SetupSignal,
    SetupType,
    Structure,
)

# Weights for composite scoring
W_BASE = 0.30
W_ALIGNMENT = 0.25
W_STRENGTH = 0.15
W_PROXIMITY = 0.15
W_CONFLUENCE = 0.15


def _near_level(price: float, level: float, atr: float, mult: float = 1.5) -> bool:
    """Check if price is within mult * ATR of a level."""
    return abs(price - level) <= atr * mult


def _structure_alignment(direction: Direction, structure: Structure) -> float:
    """Score how well a signal direction aligns with market structure. 0.0–1.0."""
    if structure == Structure.RANGING:
        return 0.5
    with_trend = (
        (direction == Direction.LONG and structure == Structure.BULLISH)
        or (direction == Direction.SHORT and structure == Structure.BEARISH)
    )
    return 1.0 if with_trend else 0.0


def score_signal(signal: SetupSignal, confluence_count: int, total_detectors: int) -> float:
    """Compute composite confidence score from multiple quality factors.

    Also stores individual components on the signal for calibration experiments.

    Args:
        signal: The setup signal with its base confidence and context.
        confluence_count: Number of detectors that fired the same direction for this coin.
        total_detectors: Total number of detector functions (for normalizing confluence).
    """
    from .models import ScoreComponents

    ctx = signal.context

    # Base prior — the detector's intrinsic confidence
    base = min(signal.confidence, 1.0)

    # Structure alignment — with-trend beats counter-trend
    alignment = _structure_alignment(signal.direction, ctx.structure)

    # Level strength — more touches on the key level = stronger
    strength = 0.5  # default for setups without explicit S/R level
    for level in ctx.sr_levels:
        if abs(level.price - signal.key_level) < ctx.atr * 0.5:
            strength = min(level.strength, 5) / 5.0
            break

    # Proximity — how close price is to the key level, normalized by ATR
    if ctx.atr > 0:
        distance_ratio = abs(ctx.current_price - signal.key_level) / (2 * ctx.atr)
        proximity = max(0.0, 1.0 - distance_ratio)
    else:
        proximity = 0.5

    # Confluence — fraction of detectors agreeing on this coin+direction
    confluence = min(confluence_count, total_detectors) / total_detectors

    # Store components for calibration experiments
    signal.score_components = ScoreComponents(
        base=round(base, 4),
        alignment=round(alignment, 4),
        strength=round(strength, 4),
        proximity=round(proximity, 4),
        confluence=round(confluence, 4),
    )

    composite = (
        W_BASE * base
        + W_ALIGNMENT * alignment
        + W_STRENGTH * strength
        + W_PROXIMITY * proximity
        + W_CONFLUENCE * confluence
    )
    return round(min(composite, 1.0), 4)


def _level_recently_breached(df: pd.DataFrame, level_price: float, direction: Direction,
                             atr: float, lookback: int = 10) -> bool:
    """Check if recent candles closed decisively through a level, invalidating it.

    For LONG (support buy): breached if a recent candle closed > 0.5 ATR below the level.
    For SHORT (resistance sell): breached if a recent candle closed > 0.5 ATR above the level.
    """
    recent_closes = df["Close"].iloc[-lookback:]
    threshold = atr * 0.5
    if direction == Direction.LONG:
        # Support is breached if price closed well below it
        return bool((recent_closes < level_price - threshold).any())
    else:
        # Resistance is breached if price closed well above it
        return bool((recent_closes > level_price + threshold).any())


def detect_a_same_way_level(
    coin: str, df: pd.DataFrame, ctx: MarketContext, proximity_mult: float = 1.5,
) -> list[SetupSignal]:
    """Setup A: RANGING + price at non-degraded S/R boundary.

    Filters: skip levels recently breached by a decisive close, and skip
    signals that contradict recent MSBs (e.g. buying support when a bearish
    MSB just fired).
    """
    if ctx.structure != Structure.RANGING:
        return []

    # Check for recent MSBs that indicate directional momentum
    recent_msb_direction = None
    if ctx.msbs:
        recent = [m for m in ctx.msbs if m.index >= len(df) - 20]
        if recent:
            latest_msb = max(recent, key=lambda m: m.index)
            recent_msb_direction = latest_msb.type  # "bullish" or "bearish"

    signals = []
    for level in ctx.sr_levels:
        if level.degraded:
            continue
        if not _near_level(ctx.current_price, level.price, ctx.atr, proximity_mult):
            continue

        # Direction based on whether price is at support or resistance
        if ctx.current_price <= level.price:
            direction = Direction.LONG
            desc = f"Price at support ${level.price:,.2f} in ranging market. Level tested {level.strength}x."
        else:
            direction = Direction.SHORT
            desc = f"Price at resistance ${level.price:,.2f} in ranging market. Level tested {level.strength}x."

        # Fix 2: Skip if level was recently breached by a decisive close
        if _level_recently_breached(df, level.price, direction, ctx.atr):
            continue

        # Fix 3: Skip if recent MSB contradicts the trade direction
        # Bearish MSB = don't buy support (it's likely to break)
        # Bullish MSB = don't sell resistance (it's likely to break)
        if recent_msb_direction == "bearish" and direction == Direction.LONG:
            continue
        if recent_msb_direction == "bullish" and direction == Direction.SHORT:
            continue

        signals.append(SetupSignal(
            coin=coin,
            setup_type=SetupType.A,
            direction=direction,
            confidence=0.5 + (level.strength - 2) * 0.1,
            key_level=level.price,
            context=ctx,
            description=f"Setup A (Same-Way Level) for {coin}. {desc}",
        ))
    return signals


def detect_b_breakout(
    coin: str, df: pd.DataFrame, ctx: MarketContext, proximity_mult: float = 1.5,
) -> list[SetupSignal]:
    """Setup B: Recent close crosses nearby S/R with conviction (>0.25 ATR beyond)."""
    signals = []
    closes_before = df["Close"].iloc[-10:-5]
    last_close = df["Close"].iloc[-1]
    min_break = ctx.atr * 0.25

    if len(closes_before) == 0:
        return []

    # Breakout uses wider proximity than base (proportional scaling)
    b_mult = proximity_mult * (2.0 / 1.5)

    for level in ctx.sr_levels:
        # Only consider levels near current price
        if not _near_level(ctx.current_price, level.price, ctx.atr, mult=b_mult):
            continue

        was_below = (closes_before < level.price).all()
        was_above = (closes_before > level.price).all()

        if was_below and (last_close - level.price) >= min_break:
            signals.append(SetupSignal(
                coin=coin,
                setup_type=SetupType.B,
                direction=Direction.LONG,
                confidence=0.5,
                key_level=level.price,
                context=ctx,
                description=(
                    f"Setup B (Breakout) for {coin}. Price broke above ${level.price:,.2f} "
                    f"resistance (strength={level.strength}). Looking for continuation."
                ),
            ))
        elif was_above and (level.price - last_close) >= min_break:
            signals.append(SetupSignal(
                coin=coin,
                setup_type=SetupType.B,
                direction=Direction.SHORT,
                confidence=0.5,
                key_level=level.price,
                context=ctx,
                description=(
                    f"Setup B (Breakdown) for {coin}. Price broke below ${level.price:,.2f} "
                    f"support (strength={level.strength}). Looking for continuation."
                ),
            ))

    # Cap at best signal per direction (closest broken level)
    best = {}
    for s in signals:
        dist = abs(ctx.current_price - s.key_level)
        if s.direction not in best or dist < abs(ctx.current_price - best[s.direction].key_level):
            best[s.direction] = s
    return list(best.values())


def detect_c_sr_flip(
    coin: str, df: pd.DataFrame, ctx: MarketContext, proximity_mult: float = 1.5,
) -> list[SetupSignal]:
    """Setup C: Recent MSB + price pulling back to broken level (S/R flip)."""
    if not ctx.msbs:
        return []

    signals = []
    recent_msbs = [m for m in ctx.msbs if m.index >= len(df) - 30]

    for msb in recent_msbs:
        if not _near_level(ctx.current_price, msb.level, ctx.atr, proximity_mult):
            continue

        if msb.type == "bullish":
            direction = Direction.LONG
            desc = f"Bullish MSB at ${msb.level:,.2f}. Price pulling back to broken resistance — potential S/R flip support."
        else:
            direction = Direction.SHORT
            desc = f"Bearish MSB at ${msb.level:,.2f}. Price pulling back to broken support — potential S/R flip resistance."

        signals.append(SetupSignal(
            coin=coin,
            setup_type=SetupType.C,
            direction=direction,
            confidence=0.65,
            key_level=msb.level,
            context=ctx,
            description=f"Setup C (S/R Flip Continuation) for {coin}. {desc}",
        ))

    # Cap at closest MSB per direction
    best = {}
    for s in signals:
        dist = abs(ctx.current_price - s.key_level)
        if s.direction not in best or dist < abs(ctx.current_price - best[s.direction].key_level):
            best[s.direction] = s
    return list(best.values())


def detect_d_deeper_retrace(coin: str, df: pd.DataFrame, ctx: MarketContext) -> list[SetupSignal]:
    """Setup D: Recent MSB + SR flip failed + price in OTE zone (0.618-0.786)."""
    if not ctx.msbs or not ctx.fib_levels:
        return []

    signals = []
    recent_msbs = [m for m in ctx.msbs if m.index >= len(df) - 30]
    ote_low, ote_high = sorted(ctx.fib_levels.ote_zone)

    if not (ote_low <= ctx.current_price <= ote_high):
        return []

    for msb in recent_msbs:
        if msb.type == "bullish":
            direction = Direction.LONG
            desc = f"Bullish MSB. Price retraced past flip level into OTE zone (${ote_low:,.2f}-${ote_high:,.2f}). Deeper retrace long."
        else:
            direction = Direction.SHORT
            desc = f"Bearish MSB. Price retraced past flip level into OTE zone (${ote_low:,.2f}-${ote_high:,.2f}). Deeper retrace short."

        signals.append(SetupSignal(
            coin=coin,
            setup_type=SetupType.D,
            direction=direction,
            confidence=0.6,
            key_level=ctx.current_price,
            context=ctx,
            description=f"Setup D (Deeper Retrace) for {coin}. {desc}",
        ))
        break  # One signal per MSB cluster
    return signals


def detect_e_reversal(coin: str, df: pd.DataFrame, ctx: MarketContext) -> list[SetupSignal]:
    """Setup E: 2+ opposing MSBs in recent history — failed continuation / reversal."""
    if len(ctx.msbs) < 2:
        return []

    signals = []
    recent_msbs = [m for m in ctx.msbs if m.index >= len(df) - 30]

    bullish_msbs = [m for m in recent_msbs if m.type == "bullish"]
    bearish_msbs = [m for m in recent_msbs if m.type == "bearish"]

    if bullish_msbs and bearish_msbs:
        # Most recent MSB determines direction
        latest = max(recent_msbs, key=lambda m: m.index)
        if latest.type == "bullish":
            direction = Direction.LONG
            desc = f"Multiple MSBs detected — latest is bullish at ${latest.level:,.2f}. Potential reversal to upside."
        else:
            direction = Direction.SHORT
            desc = f"Multiple MSBs detected — latest is bearish at ${latest.level:,.2f}. Potential reversal to downside."

        signals.append(SetupSignal(
            coin=coin,
            setup_type=SetupType.E,
            direction=direction,
            confidence=0.45,
            key_level=latest.level,
            context=ctx,
            description=f"Setup E (Reversal / Failed Continuation) for {coin}. {desc}",
        ))
    return signals


def detect_f_deviation(
    coin: str, df: pd.DataFrame, ctx: MarketContext, proximity_mult: float = 1.5,
) -> list[SetupSignal]:
    """Setup F: Recent candles — significant wick beyond nearby S/R + close back inside."""
    signals = []
    recent = df.iloc[-5:]
    min_pierce = ctx.atr * 0.25  # wick must exceed level by at least 0.25 ATR

    for level in ctx.sr_levels:
        # Only consider levels near current price
        if not _near_level(ctx.current_price, level.price, ctx.atr, proximity_mult):
            continue

        # Bearish deviation: wick meaningfully above level, close back below
        max_wick_above = (recent["High"] - level.price).max()
        closed_below = recent["Close"].iloc[-1] < level.price

        # Bullish deviation: wick meaningfully below level, close back above
        max_wick_below = (level.price - recent["Low"]).max()
        closed_above = recent["Close"].iloc[-1] > level.price

        if max_wick_above >= min_pierce and closed_below:
            signals.append(SetupSignal(
                coin=coin,
                setup_type=SetupType.F,
                direction=Direction.SHORT,
                confidence=0.55,
                key_level=level.price,
                context=ctx,
                description=(
                    f"Setup F (Deviation) for {coin}. Price wicked above ${level.price:,.2f} "
                    "resistance but closed back below. Failed breakout — short."
                ),
            ))
        elif max_wick_below >= min_pierce and closed_above:
            signals.append(SetupSignal(
                coin=coin,
                setup_type=SetupType.F,
                direction=Direction.LONG,
                confidence=0.55,
                key_level=level.price,
                context=ctx,
                description=(
                    f"Setup F (Deviation) for {coin}. Price wicked below ${level.price:,.2f} "
                    "support but closed back above. Failed breakdown — long."
                ),
            ))

    # Cap at best signal per direction
    best = {}
    for s in signals:
        key = s.direction
        dist = abs(ctx.current_price - s.key_level)
        if key not in best or dist < abs(ctx.current_price - best[key].key_level):
            best[key] = s
    return list(best.values())


def detect_g_spike_sfp(coin: str, df: pd.DataFrame, ctx: MarketContext) -> list[SetupSignal]:
    """Setup G: Last 5 candles — wick beyond recent swing H/L + close back inside + wick > 1.5x body."""
    signals = []
    recent = df.iloc[-5:]

    # Only consider swings from the last 30 bars
    cutoff = len(df) - 30
    swing_highs = [s for s in ctx.swings if s.type == "high" and s.index >= cutoff]
    swing_lows = [s for s in ctx.swings if s.type == "low" and s.index >= cutoff]

    for _, candle in recent.iterrows():
        body = abs(candle["Close"] - candle["Open"])
        upper_wick = candle["High"] - max(candle["Close"], candle["Open"])
        lower_wick = min(candle["Close"], candle["Open"]) - candle["Low"]

        if body == 0:
            continue

        wick_ratio = upper_wick / body

        # SFP above swing high
        for sh in swing_highs:
            if candle["High"] > sh.price and candle["Close"] < sh.body_price and wick_ratio > 1.5:
                signals.append((wick_ratio, SetupSignal(
                    coin=coin,
                    setup_type=SetupType.G,
                    direction=Direction.SHORT,
                    confidence=0.6,
                    key_level=sh.price,
                    context=ctx,
                    description=f"Setup G (Spike/SFP) for {coin}. Wick above swing high ${sh.price:,.2f} with close back below. Stop run — short.",
                )))

        wick_ratio_low = lower_wick / body if body > 0 else 0

        # SFP below swing low
        for sl in swing_lows:
            if candle["Low"] < sl.price and candle["Close"] > sl.body_price and wick_ratio_low > 1.5:
                signals.append((wick_ratio_low, SetupSignal(
                    coin=coin,
                    setup_type=SetupType.G,
                    direction=Direction.LONG,
                    confidence=0.6,
                    key_level=sl.price,
                    context=ctx,
                    description=f"Setup G (Spike/SFP) for {coin}. Wick below swing low ${sl.price:,.2f} with close back above. Stop run — long.",
                )))

    # Cap at best signal per direction (strongest wick ratio)
    best: dict[Direction, tuple[float, SetupSignal]] = {}
    for ratio, sig in signals:
        if sig.direction not in best or ratio > best[sig.direction][0]:
            best[sig.direction] = (ratio, sig)
    return [sig for _, sig in best.values()]


DETECTORS = [
    detect_a_same_way_level,
    detect_b_breakout,
    detect_c_sr_flip,
    detect_d_deeper_retrace,
    detect_e_reversal,
    detect_f_deviation,
    detect_g_spike_sfp,
]


def detect_all_setups(
    coin: str,
    df: pd.DataFrame,
    ctx: MarketContext,
    max_per_coin: int = 3,
    proximity_mult: float = 1.5,
    disabled_types: set[str] | None = None,
) -> list[SetupSignal]:
    """Run all setup detectors on a coin. Score with composite confidence, cap at max_per_coin."""
    # Detectors that accept proximity_mult
    _proximity_detectors = {detect_a_same_way_level, detect_b_breakout, detect_c_sr_flip, detect_f_deviation}

    all_signals = []
    for detector in DETECTORS:
        if detector in _proximity_detectors:
            all_signals.extend(detector(coin, df, ctx, proximity_mult=proximity_mult))
        else:
            all_signals.extend(detector(coin, df, ctx))

    # Filter disabled types before scoring
    if disabled_types:
        all_signals = [s for s in all_signals if s.setup_type.value not in disabled_types]

    # Count how many detectors fired per direction (for confluence scoring)
    direction_counts = Counter(s.direction for s in all_signals)

    # Replace raw confidence with composite score
    for signal in all_signals:
        signal.confidence = score_signal(
            signal,
            confluence_count=direction_counts[signal.direction],
            total_detectors=len(DETECTORS),
        )

    all_signals.sort(key=lambda s: s.confidence, reverse=True)
    return all_signals[:max_per_coin]
