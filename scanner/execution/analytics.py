"""Execution analytics — realized performance, sim vs live comparison."""

import json as _json
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TradeRecord:
    scan_id: str
    coin: str
    direction: str
    entry_fill: float
    exit_fill: float
    exit_type: str  # SL, TP1, TP2
    realized_pnl: float
    realized_r: float
    risk_amount: float
    holding_hours: float
    funding_paid: float
    slippage_r: float
    execution_mode: str
    exchange: str = "hyperliquid"


def get_closed_trades(conn, execution_mode: str | None = None, days: int | None = None) -> list[TradeRecord]:
    """Join orders table to build trade-level records for closed positions.

    A closed trade = entry FILLED + at least one exit (SL/TP) FILLED.
    """
    clauses = [
        "order_type = 'entry'",
        "status = 'FILLED'",
    ]
    params = []

    if execution_mode:
        clauses.append("execution_mode = ?")
        params.append(execution_mode)
    if days:
        clauses.append(f"datetime(created_at) > datetime('now', '-{days} days')")

    where = " AND ".join(clauses)

    entries = conn.execute(
        f"SELECT * FROM orders WHERE {where} ORDER BY created_at",
        params,
    ).fetchall()

    trades = []
    for entry in entries:
        entry = dict(entry)
        # Find ALL filled exits for this scan
        exits = conn.execute(
            """SELECT * FROM orders
            WHERE scan_id = ? AND order_type IN ('stop_loss', 'tp1', 'tp2', 'timeout_close', 'emergency_close')
            AND status = 'FILLED'
            AND execution_mode = ? AND exchange = ?
            ORDER BY filled_at""",
            (entry["scan_id"], entry["execution_mode"], entry["exchange"]),
        ).fetchall()

        if not exits:
            continue  # Still open

        exits = [dict(x) for x in exits]
        entry_fill = entry.get("fill_price") or entry.get("entry_price") or 0
        entry_qty = entry.get("quantity", 0)
        filled_exit_qty = sum(x.get("quantity", 0) for x in exits)

        # Skip partially open positions (not fully closed yet)
        # Use epsilon to handle IEEE 754 artifacts (e.g. 0.043+0.042 = 0.08499...99)
        if entry_qty - filled_exit_qty > 1e-9:
            continue

        # Sum PnL across all exit legs
        realized_pnl = 0
        for exit_order in exits:
            exit_fill = exit_order.get("fill_price") or exit_order.get("trigger_price") or 0
            exit_qty = exit_order.get("quantity", 0)
            if entry["direction"] == "LONG":
                realized_pnl += (exit_fill - entry_fill) * exit_qty
            else:
                realized_pnl += (entry_fill - exit_fill) * exit_qty

        # Use last exit for labels and holding hours
        last_exit = exits[-1]
        last_exit_fill = last_exit.get("fill_price") or last_exit.get("trigger_price") or 0

        # Funding paid (sum across all orders for this scan)
        funding_row = conn.execute(
            """SELECT COALESCE(SUM(funding_paid), 0) as total FROM orders
            WHERE scan_id = ? AND execution_mode = ? AND exchange = ?""",
            (entry["scan_id"], entry["execution_mode"], entry["exchange"]),
        ).fetchone()
        funding_paid = float(funding_row["total"])

        # Risk amount: uses full entry quantity (that was the original risk)
        stop_order = conn.execute(
            """SELECT trigger_price FROM orders
            WHERE scan_id = ? AND order_type = 'stop_loss'
            AND execution_mode = ? AND exchange = ?
            LIMIT 1""",
            (entry["scan_id"], entry["execution_mode"], entry["exchange"]),
        ).fetchone()
        stop_price = float(stop_order["trigger_price"]) if stop_order else 0
        stop_distance = abs(entry_fill - stop_price)
        risk_amount = stop_distance * entry_qty if stop_distance > 0 else 0

        # Realized R (scaled by risk_mult for counter-trend position-weighted R)
        realized_r = realized_pnl / risk_amount if risk_amount > 0 else 0
        r_mult = float(entry.get("risk_mult") or 1.0)
        if r_mult < 1.0 - 1e-9:
            realized_r *= r_mult

        # Holding hours (entry to last exit)
        entry_time = entry.get("filled_at") or entry.get("placed_at") or entry.get("created_at")
        exit_time = last_exit.get("filled_at") or last_exit.get("created_at")
        holding_hours = 0
        if entry_time and exit_time:
            try:
                t1 = datetime.fromisoformat(entry_time)
                t2 = datetime.fromisoformat(exit_time)
                holding_hours = (t2 - t1).total_seconds() / 3600
            except (ValueError, TypeError):
                pass

        # Slippage: compare actual fill to machine_block expected
        slippage_r = _calculate_slippage_r(entry, last_exit, entry_fill, last_exit_fill, risk_amount, conn)

        # Map order_type to exit_type label (last exit determines label)
        exit_type_map = {
            "stop_loss": "SL", "tp1": "TP1", "tp2": "TP2",
            "timeout_close": "TIMEOUT", "emergency_close": "EMERGENCY",
            "drift_close": "DRIFT",
        }
        exit_type = exit_type_map.get(last_exit["order_type"], last_exit["order_type"])

        trades.append(TradeRecord(
            scan_id=entry["scan_id"],
            coin=entry["coin"],
            direction=entry["direction"],
            entry_fill=entry_fill,
            exit_fill=last_exit_fill,
            exit_type=exit_type,
            realized_pnl=realized_pnl,
            realized_r=realized_r,
            risk_amount=risk_amount,
            holding_hours=holding_hours,
            funding_paid=funding_paid,
            slippage_r=slippage_r,
            execution_mode=entry.get("execution_mode", "testnet"),
            exchange=entry.get("exchange", "hyperliquid"),
        ))

    return trades


def _calculate_slippage_r(entry, exit_order, entry_fill, exit_fill, risk_amount, conn) -> float:
    """Calculate slippage in R terms by comparing fills to machine_block expected prices."""
    if risk_amount <= 0:
        return 0

    # Get machine_block from scans table
    scan_row = conn.execute(
        "SELECT machine_block FROM scans WHERE scan_id = ?",
        (entry["scan_id"],),
    ).fetchone()

    if not scan_row or not scan_row["machine_block"]:
        return 0

    try:
        mb = _json.loads(scan_row["machine_block"])
    except (ValueError, TypeError):
        return 0

    expected_entry = (mb.get("entry_low", 0) + mb.get("entry_high", 0)) / 2
    if expected_entry <= 0:
        return 0

    entry_slip = abs(entry_fill - expected_entry) * (entry.get("quantity", 0) or 0)
    return entry_slip / risk_amount if risk_amount > 0 else 0


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


def compare_sim_vs_live(trades: list[TradeRecord], conn) -> list[dict]:
    """For each closed trade, compare live R to theoretical (machine_block) R."""
    comparisons = []

    for trade in trades:
        scan_row = conn.execute(
            "SELECT machine_block FROM scans WHERE scan_id = ?",
            (trade.scan_id,),
        ).fetchone()

        if not scan_row or not scan_row["machine_block"]:
            continue

        try:
            mb = _json.loads(scan_row["machine_block"])
        except (ValueError, TypeError):
            continue

        entry_mid = (mb.get("entry_low", 0) + mb.get("entry_high", 0)) / 2
        stop_loss = mb.get("stop_loss", 0)
        tp_prices = {"TP1": mb.get("tp1", 0), "TP2": mb.get("tp2", 0)}

        if not entry_mid or not stop_loss:
            continue

        stop_dist = abs(entry_mid - stop_loss)
        if stop_dist == 0:
            continue

        # Fetch all filled exits for quantity-weighted sim_r
        exits = conn.execute(
            """SELECT order_type, quantity FROM orders
            WHERE scan_id = ? AND order_type IN ('stop_loss', 'tp1', 'tp2', 'timeout_close', 'emergency_close')
            AND status = 'FILLED'
            AND execution_mode = ? AND exchange = ?""",
            (trade.scan_id, trade.execution_mode, trade.exchange),
        ).fetchall()

        if not exits:
            continue

        total_exit_qty = sum(e["quantity"] for e in exits)
        if total_exit_qty <= 0:
            continue

        weighted_sim_r = 0
        for ex in exits:
            qty = ex["quantity"]
            if ex["order_type"] == "stop_loss":
                leg_r = -1.0
            else:
                tp_key = "TP1" if ex["order_type"] == "tp1" else "TP2"
                tp_price = tp_prices.get(tp_key, 0)
                if tp_price:
                    if trade.direction == "LONG":
                        leg_r = (tp_price - entry_mid) / stop_dist
                    else:
                        leg_r = (entry_mid - tp_price) / stop_dist
                else:
                    leg_r = 0
            weighted_sim_r += leg_r * qty

        sim_r = weighted_sim_r / total_exit_qty

        comparisons.append({
            "scan_id": trade.scan_id,
            "coin": trade.coin,
            "sim_r": round(sim_r, 4),
            "live_r": round(trade.realized_r, 4),
            "delta_r": round(trade.realized_r - sim_r, 4),
            "slippage_r": round(trade.slippage_r, 4),
        })

    return comparisons


def rolling_expectancy(trades: list[TradeRecord], window: int = 20) -> list[tuple[int, float]]:
    """Sliding window R mean for trend detection.

    Returns list of (trade_index, expectancy) tuples.
    """
    if len(trades) < window:
        return []

    results = []
    for i in range(window, len(trades) + 1):
        window_trades = trades[i - window:i]
        avg_r = sum(t.realized_r for t in window_trades) / window
        results.append((i, round(avg_r, 4)))

    return results
