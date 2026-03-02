"""Position sizing from machine_block parameters."""

from dataclasses import dataclass


@dataclass
class PositionSize:
    quantity: float
    notional_value: float
    risk_amount: float
    stop_distance: float
    rr_ratio: float
    leverage: int
    margin_required: float
    tp1_quantity: float
    tp2_quantity: float


def calculate_position(
    machine_block: dict,
    balance: float,
    risk_pct: float,
    max_leverage: int | None = None,
) -> PositionSize | None:
    """Calculate position size from machine_block values.

    Returns None if inputs are invalid (zero risk, missing fields).
    """
    entry_low = machine_block.get("entry_low", 0)
    entry_high = machine_block.get("entry_high", 0)
    stop_loss = machine_block.get("stop_loss", 0)
    tp1 = machine_block.get("tp1")
    tp2 = machine_block.get("tp2")
    direction = machine_block.get("direction", "").upper()
    leverage = machine_block.get("leverage", 1)

    if not entry_low or not entry_high or not stop_loss:
        return None

    entry_mid = (entry_low + entry_high) / 2
    stop_distance = abs(entry_mid - stop_loss)

    if stop_distance == 0:
        return None

    # Ensure integer leverage (LLM returns float like 5.0)
    leverage = int(leverage)

    # Cap leverage
    if max_leverage is not None:
        leverage = min(leverage, max_leverage)

    # Risk-based sizing: risk_amount = balance * risk_pct / 100
    risk_amount = balance * risk_pct / 100
    quantity = risk_amount / stop_distance

    notional_value = quantity * entry_mid
    margin_required = notional_value / leverage

    # R:R calculation
    tp_price = tp1 or tp2
    if tp_price:
        if direction == "LONG":
            reward = tp_price - entry_mid
        else:
            reward = entry_mid - tp_price
        rr_ratio = reward / stop_distance if stop_distance > 0 else 0
    else:
        rr_ratio = 0

    # TP quantity split — normalize to sum to 100% (LLM may produce 60+30=90 etc)
    tp1_exit_pct = machine_block.get("tp1_exit_pct", 50)
    tp2_exit_pct = machine_block.get("tp2_exit_pct", 50)

    if tp1 and tp2:
        total_pct = tp1_exit_pct + tp2_exit_pct
        if total_pct > 0:
            tp1_quantity = quantity * tp1_exit_pct / total_pct
            tp2_quantity = quantity - tp1_quantity  # remainder — no floating point gap
        else:
            tp1_quantity = quantity / 2
            tp2_quantity = quantity / 2
    elif tp1:
        tp1_quantity = quantity
        tp2_quantity = 0
    else:
        tp1_quantity = 0
        tp2_quantity = quantity if tp2 else 0

    return PositionSize(
        quantity=quantity,
        notional_value=notional_value,
        risk_amount=risk_amount,
        stop_distance=stop_distance,
        rr_ratio=rr_ratio,
        leverage=leverage,
        margin_required=margin_required,
        tp1_quantity=tp1_quantity,
        tp2_quantity=tp2_quantity,
    )
