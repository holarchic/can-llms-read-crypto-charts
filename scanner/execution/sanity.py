"""Pre-execution price sanity checks."""

from dataclasses import dataclass, field

from scanner.models import MachineBlock


@dataclass
class SanityResult:
    passed: bool
    errors: list[str] = field(default_factory=list)


def is_price_in_entry_zone(
    current_price: float,
    entry_low: float,
    entry_high: float,
    direction: str = "",
    buffer_pct: float = 0.0,
) -> bool:
    """Check if current price falls within the entry zone bounds.

    Direction-aware buffer: price that has moved past the zone is still
    accepted up to buffer_pct of the zone midpoint in either direction.
    - LONG:  allow price up to buffer% above entry_high  (momentum entry)
    - SHORT: allow price up to buffer% below entry_low   (momentum entry)
    Both directions also accept price slightly past the zone on the favorable
    side (SHORT above entry_high, LONG below entry_low).
    """
    if entry_low <= current_price <= entry_high:
        return True

    if buffer_pct <= 0 or not direction:
        return False

    mid = (entry_low + entry_high) / 2
    buffer = mid * buffer_pct / 100

    direction = direction.upper()
    if direction == "SHORT":
        if current_price < entry_low:
            return current_price >= entry_low - buffer
        if current_price > entry_high:
            return current_price <= entry_high + buffer
    if direction == "LONG":
        if current_price > entry_high:
            return current_price <= entry_high + buffer
        if current_price < entry_low:
            return current_price >= entry_low - buffer

    return False


def check_price_sanity(
    machine_block: MachineBlock,
    current_price: float,
    atr: float,
    config,
) -> SanityResult:
    """Run all pre-execution sanity checks on machine_block values.

    Checks: entry zone proximity, stop distance vs ATR, R:R bounds.
    """
    errors = []

    entry_low = machine_block.entry_low
    entry_high = machine_block.entry_high
    stop_loss = machine_block.stop_loss
    tp1 = machine_block.tp1
    tp2 = machine_block.tp2
    direction = machine_block.direction.upper()

    if not entry_low or not entry_high or not stop_loss:
        errors.append("Missing required price fields (entry_low, entry_high, stop_loss)")
        return SanityResult(passed=False, errors=errors)

    entry_mid = (entry_low + entry_high) / 2

    # 1. Entry zone proximity — entry_mid within tolerance of current price
    entry_deviation_pct = abs(entry_mid - current_price) / current_price * 100
    if entry_deviation_pct > config.entry_price_tolerance_pct:
        errors.append(
            f"Entry zone midpoint {entry_mid:.2f} is {entry_deviation_pct:.1f}% from "
            f"current price {current_price:.2f} (max {config.entry_price_tolerance_pct}%)"
        )

    # 2. Stop distance vs ATR
    if atr > 0:
        stop_distance = abs(entry_mid - stop_loss)
        atr_mult = stop_distance / atr
        if atr_mult < config.min_atr_stop_mult:
            errors.append(
                f"Stop distance {stop_distance:.2f} is {atr_mult:.2f}x ATR "
                f"(min {config.min_atr_stop_mult}x)"
            )
        if atr_mult > config.max_atr_stop_mult:
            errors.append(
                f"Stop distance {stop_distance:.2f} is {atr_mult:.2f}x ATR "
                f"(max {config.max_atr_stop_mult}x)"
            )

    # 3. R:R ratio
    stop_distance = abs(entry_mid - stop_loss)
    if stop_distance > 0:
        # Use TP1 for R:R check (primary target)
        tp_price = tp1 or tp2
        if tp_price:
            if direction == "LONG":
                reward = tp_price - entry_mid
            else:
                reward = entry_mid - tp_price
            rr = reward / stop_distance
            if rr < config.min_rr_ratio:
                errors.append(f"R:R ratio {rr:.2f} below minimum {config.min_rr_ratio}")
            if rr > config.max_rr_ratio:
                errors.append(f"R:R ratio {rr:.2f} above maximum {config.max_rr_ratio}")

    return SanityResult(passed=len(errors) == 0, errors=errors)
