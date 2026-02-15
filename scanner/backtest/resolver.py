"""Walk-forward trade outcome resolution + price parsing."""

import re
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class TradeSetup:
    """Mode-agnostic trade setup — built from either signals or scans table."""

    coin: str
    direction: str  # LONG or SHORT
    entry: float
    stop: float
    target: float
    timestamp: str  # ISO format

    # Metadata for slicing
    setup_type: str = ""
    confidence: float = 0.0
    structure: str = ""
    sent_to_llm: bool = False
    source: str = ""  # "screener" or "llm"


@dataclass
class TradeOutcome:
    """Result of walking forward through price data."""

    result: str  # WIN, LOSS, EXPIRED
    r_achieved: float = 0.0
    bars_to_exit: int = 0
    exit_price: float = 0.0
    mae: float = 0.0  # Max Adverse Excursion in R
    mfe: float = 0.0  # Max Favorable Excursion in R
    fees_paid: float = 0.0  # In R-multiples

    # Carry forward for slicing
    setup: TradeSetup | None = None


def resolve_signal(
    setup: TradeSetup,
    candles: pd.DataFrame,
    timeout_bars: int = 168,
    fee_rate: float = 0.00035,
) -> TradeOutcome:
    """Walk forward through 1H candles from signal time to determine outcome.

    Args:
        setup: Trade setup with entry/stop/target.
        candles: DataFrame with DatetimeIndex + Open/High/Low/Close columns.
        timeout_bars: Max bars before EXPIRED (default 168 = 7 days of 1H).
        fee_rate: Taker fee per side (default 0.035% for HyperLiquid).

    Returns:
        TradeOutcome with result, R-multiple achieved, MAE/MFE, etc.
    """
    risk = abs(setup.entry - setup.stop)
    if risk == 0:
        return TradeOutcome(result="EXPIRED", setup=setup)

    reward = abs(setup.target - setup.entry)
    is_long = setup.direction == "LONG"

    # Fee impact in R-multiples (round-trip: entry + exit)
    fee_r = (fee_rate * 2 * setup.entry) / risk

    # Filter candles to after signal time
    signal_time = pd.Timestamp(setup.timestamp, tz="UTC")
    future = candles[candles.index > signal_time]

    if future.empty:
        return TradeOutcome(result="EXPIRED", setup=setup)

    mae = 0.0  # worst drawdown in R
    mfe = 0.0  # best excursion in R

    for i, (ts, bar) in enumerate(future.iterrows()):
        if i >= timeout_bars:
            break

        if is_long:
            # Adverse = how far price dropped below entry
            adverse_r = (setup.entry - bar["Low"]) / risk
            # Favorable = how far price rose above entry
            favor_r = (bar["High"] - setup.entry) / risk

            hit_stop = bar["Low"] <= setup.stop
            hit_target = bar["High"] >= setup.target
        else:
            # SHORT: adverse = price rising above entry
            adverse_r = (bar["High"] - setup.entry) / risk
            # Favorable = price dropping below entry
            favor_r = (setup.entry - bar["Low"]) / risk

            hit_stop = bar["High"] >= setup.stop
            hit_target = bar["Low"] <= setup.target

        mae = max(mae, adverse_r)
        mfe = max(mfe, favor_r)

        # Both hit same candle → conservative: LOSS
        if hit_stop and hit_target:
            return TradeOutcome(
                result="LOSS",
                r_achieved=-1.0 - fee_r,
                bars_to_exit=i + 1,
                exit_price=setup.stop,
                mae=mae,
                mfe=mfe,
                fees_paid=fee_r,
                setup=setup,
            )

        if hit_stop:
            return TradeOutcome(
                result="LOSS",
                r_achieved=-1.0 - fee_r,
                bars_to_exit=i + 1,
                exit_price=setup.stop,
                mae=mae,
                mfe=mfe,
                fees_paid=fee_r,
                setup=setup,
            )

        if hit_target:
            target_r = reward / risk
            return TradeOutcome(
                result="WIN",
                r_achieved=target_r - fee_r,
                bars_to_exit=i + 1,
                exit_price=setup.target,
                mae=mae,
                mfe=mfe,
                fees_paid=fee_r,
                setup=setup,
            )

    # EXPIRED — close at last available bar
    last_close = future.iloc[min(timeout_bars - 1, len(future) - 1)]["Close"]
    if is_long:
        unrealized_r = (last_close - setup.entry) / risk
    else:
        unrealized_r = (setup.entry - last_close) / risk

    return TradeOutcome(
        result="EXPIRED",
        r_achieved=unrealized_r - fee_r,
        bars_to_exit=min(timeout_bars, len(future)),
        exit_price=last_close,
        mae=mae,
        mfe=mfe,
        fees_paid=fee_r,
        setup=setup,
    )


# --- Price parser for LLM text fields ---

_PRICE_RE = re.compile(r"\$?([\d,]+\.?\d*)")


def parse_price(text: str | None) -> float | None:
    """Extract a price from LLM text output.

    Handles: "$1,234.56", "1234.56", "$1234", "N/A", ranges like "$100-$110".
    For ranges, returns the midpoint.
    """
    if not text or text.strip().upper() in ("N/A", "NONE", "-", ""):
        return None

    text = text.strip()

    # Check for range pattern: "$X - $Y" or "$X-$Y" or "X to Y"
    range_match = re.match(
        r"\$?([\d,]+\.?\d*)\s*[-–—]\s*\$?([\d,]+\.?\d*)", text
    )
    if not range_match:
        range_match = re.match(
            r"\$?([\d,]+\.?\d*)\s+to\s+\$?([\d,]+\.?\d*)", text, re.IGNORECASE
        )

    if range_match:
        low = float(range_match.group(1).replace(",", ""))
        high = float(range_match.group(2).replace(",", ""))
        return (low + high) / 2

    # Single price
    match = _PRICE_RE.search(text)
    if match:
        return float(match.group(1).replace(",", ""))

    return None


# --- Setup builders ---


def build_screener_setup(
    row: dict,
    atr_stop_mult: float = 1.5,
    atr_target_mult: float = 3.0,
) -> TradeSetup | None:
    """Build a TradeSetup from a signals table row.

    Uses key_level as entry, ATR multiples for stop/target.
    """
    key_level = row.get("key_level")
    atr = row.get("atr")
    direction = row.get("direction", "").upper()

    if not key_level or not atr or atr == 0:
        return None
    if direction not in ("LONG", "SHORT"):
        return None

    if direction == "LONG":
        stop = key_level - atr * atr_stop_mult
        target = key_level + atr * atr_target_mult
    else:
        stop = key_level + atr * atr_stop_mult
        target = key_level - atr * atr_target_mult

    return TradeSetup(
        coin=row.get("coin", ""),
        direction=direction,
        entry=key_level,
        stop=stop,
        target=target,
        timestamp=row.get("timestamp", ""),
        setup_type=row.get("setup_type", ""),
        confidence=row.get("confidence", 0.0),
        structure=row.get("structure", ""),
        sent_to_llm=bool(row.get("sent_to_llm", 0)),
        source="screener",
    )


def build_llm_setup(row: dict) -> TradeSetup | None:
    """Build a TradeSetup from a scans table row (CONFIRM only).

    Parses entry_zone, stop_loss, take_profit from LLM text.
    """
    entry = parse_price(row.get("entry_zone"))
    stop = parse_price(row.get("stop_loss"))
    target = parse_price(row.get("take_profit"))

    if not entry or not stop or not target:
        return None

    direction = row.get("direction", "").upper()
    if direction not in ("LONG", "SHORT"):
        return None

    return TradeSetup(
        coin=row.get("coin", ""),
        direction=direction,
        entry=entry,
        stop=stop,
        target=target,
        timestamp=row.get("timestamp", ""),
        setup_type=row.get("setup_type", ""),
        confidence=row.get("screener_confidence", 0.0),
        structure="",
        sent_to_llm=True,
        source="llm",
    )
