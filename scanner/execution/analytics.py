"""Execution analytics — pure compute over the storage-owned closed-trade
read model (T-016 Phase 2a, SPEC #22). No DB access here: `get_closed_trades`
/ `llm_cost_summary` live in `scanner.execution.storage`."""

from scanner.models import TradeRecord


def compute_live_metrics(trades: list[TradeRecord]) -> dict:
    """Compute aggregate performance metrics from closed trades."""
    if not trades:
        return {
            "count": 0, "win_rate": 0, "expectancy_r": 0, "profit_factor": 0,
            "total_pnl": 0, "max_drawdown_pct": 0,
            "avg_holding_hours": 0, "avg_slippage_r": 0, "total_funding_r": 0,
            "fill_rate": 0, "data_source": "db_only",
            "quality_warnings": _compute_quality_warnings([]),
        }

    wins = [t for t in trades if t.realized_r > 0]
    losses = [t for t in trades if t.realized_r <= 0]

    win_rate = len(wins) / len(trades) if trades else 0
    expectancy_r = sum(t.realized_r for t in trades) / len(trades)
    total_pnl = sum(t.realized_pnl for t in trades)

    gross_profit = sum(t.realized_pnl for t in wins)
    gross_loss = abs(sum(t.realized_pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

    total_risk = sum(t.risk_amount for t in trades)

    # Max drawdown (simple: track running PnL, find max peak-to-trough)
    running = 0
    peak = 0
    max_dd = 0
    for t in trades:
        running += t.realized_pnl
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd
    # Express as % of peak, or % of total risk if equity never went positive
    if peak > 0:
        max_dd_pct = max_dd / peak * 100
    elif total_risk > 0:
        max_dd_pct = max_dd / total_risk * 100
    else:
        max_dd_pct = 0

    avg_holding = sum(t.holding_hours for t in trades) / len(trades)
    avg_slippage = sum(t.slippage_r for t in trades) / len(trades)
    total_funding = sum(t.funding_paid for t in trades)
    total_funding_r = total_funding / total_risk if total_risk > 0 else 0

    return {
        "count": len(trades),
        "win_rate": round(win_rate, 4),
        "expectancy_r": round(expectancy_r, 4),
        "profit_factor": round(profit_factor, 4),
        "total_pnl": round(total_pnl, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "avg_holding_hours": round(avg_holding, 1),
        "avg_slippage_r": round(avg_slippage, 4),
        "total_funding_r": round(total_funding_r, 4),
        "fill_rate": 0,  # Populated by caller with CONFIRM count context
        "data_source": "db_only",
        "quality_warnings": _compute_quality_warnings(trades),
    }


def _compute_quality_warnings(trades: list[TradeRecord]) -> list[str]:
    """Flag data gaps that affect metric accuracy."""
    warnings = ["All metrics estimated from DB — not reconciled with exchange"]
    if trades:
        no_funding = sum(1 for t in trades if t.funding_paid == 0)
        if 0 < no_funding < len(trades):
            warnings.append(f"{no_funding}/{len(trades)} trades missing funding data")
        elif no_funding == len(trades):
            warnings.append("Funding costs not captured — PnL overstated")
        approx_fills = sum(1 for t in trades if t.entry_fill == 0 or t.exit_fill == 0)
        if approx_fills:
            warnings.append(f"{approx_fills} trades with missing fill prices — using trigger prices")
    return warnings


def rolling_expectancy(trades: list[TradeRecord], window: int = 20) -> list[tuple[int, float]]:
    """Sliding window R mean for trend detection.

    Returns list of (trade_index, expectancy) tuples.

    Runtime-dead (C5 audit: "mainly serves tests") — retained unchanged;
    deleting runtime-dead reporting/strategy surface is a separate scope call.
    """
    if len(trades) < window:
        return []

    results = []
    for i in range(window, len(trades) + 1):
        window_trades = trades[i - window:i]
        avg_r = sum(t.realized_r for t in window_trades) / window
        results.append((i, round(avg_r, 4)))

    return results
