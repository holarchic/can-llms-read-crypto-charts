"""Drift invalidation — hybrid_inv strategy on 4H candles.

Detects four drift patterns at each 4H candle close and takes mechanical action:
- adverse/stagnation → move SL to breakeven (let market decide)
- no_momentum/reversal → force close (free up the slot)

No LLM re-evaluation needed. Drift zone derived mechanically from machine_block.
"""

import json
import logging
import time
from datetime import datetime, timezone

import requests

from scanner.config import get_exchange_meta
from scanner.storage import get_connection as get_scanner_connection

from .config import ExecutionConfig
from .exchange import cancel_order, get_auth_exchange
from .monitor import (
    _calc_r,
    _get_remaining_qty,
    cancel_remaining_orders,
    market_close_position,
)
from .storage import (
    get_open_entry_orders,
    get_orders_for_scan,
    make_cloid,
    update_order_quantity,
    update_order_status,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure functions — no side effects
# ---------------------------------------------------------------------------

def is_4h_boundary() -> bool:
    """Check if current UTC time is within the first 35 minutes of a 4H boundary.

    4H boundaries: 00:xx, 04:xx, 08:xx, 12:xx, 16:xx, 20:xx UTC.
    Monitor runs every 5 min, so one of the 7 cycles in this window will match.
    """
    now = datetime.now(timezone.utc)
    return now.hour % 4 == 0 and now.minute < 35


def compute_drift_zone(entry_price: float, stop_loss: float, direction: str) -> tuple[float, float]:
    """Return (drift_low, drift_high) — the entry-to-SL range where price is drifting.

    SHORT: price stuck between entry and stop (above entry)
    LONG: price stuck between stop and entry (below entry)
    """
    if direction == "SHORT":
        return (entry_price, stop_loss)
    return (stop_loss, entry_price)


def compute_progress(direction: str, entry_price: float, tp1: float, price: float) -> float:
    """Compute progress toward TP1 as a percentage (0-100+). Can exceed 100 if past TP."""
    tp_distance = abs(tp1 - entry_price)
    if tp_distance == 0:
        return 0.0
    if direction == "SHORT":
        return (entry_price - price) / tp_distance * 100
    return (price - entry_price) / tp_distance * 100


def evaluate_triggers(
    entry_price: float, stop_loss: float, tp1: float, direction: str,
    filled_at: str, candles_4h: list[dict], config: ExecutionConfig,
) -> tuple[str | None, str]:
    """Evaluate all drift triggers for a position against 4H candle history.

    Returns (trigger_type, action):
    - ("adverse", "sl_move") | ("stagnation", "sl_move")
    - ("no_momentum", "force_close") | ("reversal", "force_close")
    - (None, "none")
    """
    try:
        entry_time = datetime.fromisoformat(filled_at)
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return (None, "none")

    if not candles_4h:
        return (None, "none")

    drift_low, drift_high = compute_drift_zone(entry_price, stop_loss, direction)
    now = datetime.now(timezone.utc)
    age_hours = (now - entry_time).total_seconds() / 3600

    # Grace period
    if age_hours < config.drift_grace_hours:
        return (None, "none")

    # Walk candles from entry to latest, tracking state
    consecutive_in_zone = 0
    best_progress = 0.0
    momentum_checked = False

    for candle in candles_4h:
        try:
            candle_time = datetime.fromisoformat(str(candle["timestamp"]))
            if candle_time.tzinfo is None:
                candle_time = candle_time.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

        # Skip candles before entry
        if candle_time < entry_time:
            continue

        close = float(candle["close"])
        in_zone = drift_low <= close <= drift_high

        if in_zone:
            consecutive_in_zone += 1
        else:
            consecutive_in_zone = 0

        # Track best progress (use candle extremes for accuracy)
        high = float(candle["high"])
        low = float(candle["low"])
        if direction == "SHORT":
            best_progress = max(best_progress, compute_progress(direction, entry_price, tp1, low))
        else:
            best_progress = max(best_progress, compute_progress(direction, entry_price, tp1, high))

    # Current state from latest candle
    latest = candles_4h[-1]
    current_close = float(latest["close"])
    current_progress = compute_progress(direction, entry_price, tp1, current_close)
    current_in_zone = drift_low <= current_close <= drift_high

    # --- Trigger checks (priority order) ---

    # 1. No momentum — one-shot check at momentum_check_hours
    if not momentum_checked and age_hours >= config.drift_momentum_check_hours:
        momentum_checked = True
        if best_progress < config.drift_momentum_pct:
            return ("no_momentum", "force_close")

    # 2. Reversal — was working, gave it back
    if best_progress > config.drift_reversal_best_pct and current_progress < config.drift_reversal_current_pct:
        return ("reversal", "force_close")

    # 3. Adverse — in drift zone on wrong side of entry
    if current_in_zone:
        if direction == "SHORT" and current_close >= entry_price:
            return ("adverse", "sl_move")
        if direction == "LONG" and current_close <= entry_price:
            return ("adverse", "sl_move")

    # 4. Stagnation — stuck in drift zone for too long
    if consecutive_in_zone >= config.drift_consec_threshold:
        return ("stagnation", "sl_move")

    return (None, "none")


# ---------------------------------------------------------------------------
# Actions — exchange operations with side effects
# ---------------------------------------------------------------------------

def _move_sl_to_breakeven(
    entry: dict, sl_order: dict, symbol: str, exchange,
    config: ExecutionConfig, conn,
) -> bool:
    """Cancel current SL, place new SL at breakeven (entry fill price).

    Only tightens — if current SL is already tighter than entry, skip.
    Returns True on success, False on failure.
    """
    entry_price = float(entry.get("fill_price") or entry.get("entry_price") or 0)
    current_trigger = float(sl_order.get("trigger_price", 0))
    direction = entry["direction"]

    if entry_price <= 0 or current_trigger <= 0:
        log.error("Drift SL move: invalid prices entry=%s sl=%s for %s", entry_price, current_trigger, entry["coin"])
        return False

    new_sl = entry_price

    # Tighten-only: don't widen the SL
    if direction == "SHORT" and new_sl >= current_trigger:
        log.info("Drift: SL already tighter than entry for %s SHORT (current=%s, entry=%s)", entry["coin"], current_trigger, entry_price)
        return False
    if direction == "LONG" and new_sl <= current_trigger:
        log.info("Drift: SL already tighter than entry for %s LONG (current=%s, entry=%s)", entry["coin"], current_trigger, entry_price)
        return False

    # Cancel old SL on exchange
    old_exch_oid = sl_order.get("exchange_order_id")
    if old_exch_oid and not cancel_order(old_exch_oid, symbol, config):
        log.error("Drift: failed to cancel old SL %s for %s", old_exch_oid, entry["coin"])
        return False

    update_order_status(sl_order["order_id"], "CANCELLED", conn)

    # Place new SL at breakeven
    exit_side = "sell" if direction == "LONG" else "buy"
    qty = float(sl_order.get("quantity", 0))
    rounded_qty = float(exchange.amount_to_precision(symbol, qty))
    cloid = make_cloid(sl_order["order_id"] + ":drift")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            resp = exchange.create_order(
                symbol, "market", exit_side, rounded_qty, new_sl,
                {"stopLossPrice": new_sl, "reduceOnly": True, "clientOrderId": cloid},
            )
            new_exch_oid = resp.get("id", "")
            update_order_quantity(sl_order["order_id"], qty, new_exch_oid, conn)
            update_order_status(sl_order["order_id"], "PLACED", conn)
            log.info("Drift SL moved for %s %s: %s → %s (breakeven)",
                      entry["coin"], direction, current_trigger, new_sl)
            return True
        except Exception as exc:
            log.error("Drift SL replacement attempt %d/%d for %s failed: %s",
                      attempt, max_retries, entry["coin"], exc)
            if attempt < max_retries:
                time.sleep(attempt)

    log.error("All drift SL replacement attempts exhausted for %s — position unprotected", entry["coin"])
    _send_urgent_alert(f"DRIFT SL MOVE FAILED: {entry['coin']} {direction} — position unprotected after SL cancel", config)
    return False


# ---------------------------------------------------------------------------
# Discord alerts
# ---------------------------------------------------------------------------

def _send_drift_sl_alert(
    entry: dict, trigger: str, old_sl: float, new_sl: float, config: ExecutionConfig,
) -> bool:
    """Send Discord embed for drift SL move to breakeven."""
    webhook = config.discord_ops_webhook_url
    if not webhook:
        return False

    embed = {
        "title": f"\U0001f6a7 DRIFT SL \u2192 BE — {entry['coin']} {entry['direction']}",
        "color": 0xFFA726,
        "fields": [
            {"name": "Trigger", "value": trigger, "inline": True},
            {"name": "Old SL", "value": f"${old_sl:,.5f}", "inline": True},
            {"name": "New SL", "value": f"${new_sl:,.5f}", "inline": True},
        ],
        "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
    }
    try:
        resp = requests.post(webhook, json={"embeds": [embed]}, timeout=10)
        return resp.status_code == 204
    except Exception as exc:
        log.error("Drift SL alert failed: %s", exc)
        return False


def _send_drift_close_alert(
    entry: dict, trigger: str, fill_price: float, r_val: float | None, config: ExecutionConfig,
) -> bool:
    """Send Discord embed for drift force close."""
    webhook = config.discord_ops_webhook_url
    if not webhook:
        return False

    fields = [
        {"name": "Trigger", "value": trigger, "inline": True},
        {"name": "Exit", "value": f"${fill_price:,.5f}", "inline": True},
    ]
    if r_val is not None:
        fields.append({"name": "R Value", "value": f"{r_val:+.2f}R", "inline": True})

    embed = {
        "title": f"\U0001f6d1 DRIFT CLOSE — {entry['coin']} {entry['direction']}",
        "color": 0xEF5350,
        "fields": fields,
        "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
    }
    try:
        resp = requests.post(webhook, json={"embeds": [embed]}, timeout=10)
        return resp.status_code == 204
    except Exception as exc:
        log.error("Drift close alert failed: %s", exc)
        return False


def _send_urgent_alert(message: str, config: ExecutionConfig) -> bool:
    """Send urgent alert for critical drift failures."""
    webhook = config.discord_urgent_webhook_url
    if not webhook:
        return False
    embed = {
        "title": "\u26a0\ufe0f DRIFT URGENT",
        "description": message,
        "color": 0xFF0000,
        "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
    }
    try:
        resp = requests.post(webhook, json={"content": "@everyone", "embeds": [embed]}, timeout=10)
        return resp.status_code == 204
    except Exception as exc:
        log.error("Drift urgent alert failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Orchestrator — called from monitor cycle
# ---------------------------------------------------------------------------

def _get_position_data(entry: dict, conn) -> dict | None:
    """Build position data dict by joining entry order with scans table machine_block."""
    scanner_conn = get_scanner_connection()
    try:
        row = scanner_conn.execute(
            "SELECT machine_block FROM scans WHERE scan_id = ?",
            (entry["scan_id"],),
        ).fetchone()
    finally:
        scanner_conn.close()

    if not row or not row["machine_block"]:
        return None

    try:
        mb = json.loads(row["machine_block"])
    except (json.JSONDecodeError, TypeError):
        return None

    tp1 = mb.get("tp1")
    sl = mb.get("stop_loss")
    if tp1 is None or sl is None:
        return None

    return {
        "entry_price": float(entry.get("fill_price") or entry.get("entry_price") or 0),
        "stop_loss": float(sl),
        "tp1": float(tp1),
        "direction": entry["direction"],
        "filled_at": entry.get("filled_at") or entry.get("placed_at") or entry.get("created_at"),
    }


def _fetch_4h_candles(coin: str) -> list[dict]:
    """Fetch recent 4H candles for a coin. Returns list of dicts or empty list on failure."""
    try:
        from scanner.data import fetch_candles
        df = fetch_candles(coin, "4h", 20)
        if df is None or df.empty:
            return []
        records = []
        for ts, row in df.iterrows():
            records.append({
                "timestamp": ts.isoformat(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
            })
        return records
    except Exception as exc:
        log.error("Failed to fetch 4H candles for %s: %s", coin, exc)
        return []


def check_drift_positions(exec_config: ExecutionConfig, conn) -> int:
    """Top-level drift check. Called from monitor cycle at 4H boundaries.

    Returns count of positions acted on.
    """
    if not exec_config.drift_enabled:
        return 0

    if not is_4h_boundary():
        return 0

    open_entries = get_open_entry_orders(conn, exec_config.exchange, exec_config.mode)
    if not open_entries:
        return 0

    meta = get_exchange_meta(exec_config.exchange)
    exchange = get_auth_exchange(exec_config)
    acted = 0

    for entry in open_entries:
        try:
            acted += _process_drift_entry(entry, meta, exchange, exec_config, conn)
        except Exception as exc:
            log.error("Drift check failed for %s %s: %s", entry.get("coin"), entry.get("direction"), exc)

    return acted


def _process_drift_entry(entry: dict, meta: dict, exchange, config: ExecutionConfig, conn) -> int:
    """Process a single entry for drift triggers. Returns 1 if acted, 0 otherwise."""
    pos_data = _get_position_data(entry, conn)
    if not pos_data:
        return 0

    candles = _fetch_4h_candles(entry["coin"])
    if not candles:
        return 0

    trigger, action = evaluate_triggers(
        pos_data["entry_price"], pos_data["stop_loss"], pos_data["tp1"],
        pos_data["direction"], pos_data["filled_at"], candles, config,
    )

    if trigger is None:
        return 0

    symbol = meta["symbol_template"].format(coin=entry["coin"])
    log.info("Drift trigger: %s → %s for %s %s", trigger, action, entry["coin"], entry["direction"])

    if action == "sl_move":
        return _execute_sl_move(entry, trigger, symbol, exchange, config, conn)
    if action == "force_close":
        return _execute_force_close(entry, trigger, pos_data, symbol, exchange, config, conn)

    return 0


def _execute_sl_move(entry: dict, trigger: str, symbol: str, exchange, config: ExecutionConfig, conn) -> int:
    """Execute SL move to breakeven for adverse/stagnation triggers."""
    scan_orders = get_orders_for_scan(entry["scan_id"], conn)
    sl_order = next(
        (o for o in scan_orders if o["order_type"] == "stop_loss" and o["status"] == "PLACED"),
        None,
    )
    if not sl_order:
        log.error("Drift: no active SL found for %s — cannot move", entry["coin"])
        return 0

    old_trigger = float(sl_order.get("trigger_price", 0))
    entry_price = float(entry.get("fill_price") or entry.get("entry_price") or 0)

    success = _move_sl_to_breakeven(entry, dict(sl_order), symbol, exchange, config, conn)
    if success:
        _send_drift_sl_alert(entry, trigger, old_trigger, entry_price, config)
        return 1
    return 0


def _execute_force_close(
    entry: dict, trigger: str, pos_data: dict, symbol: str,
    exchange, config: ExecutionConfig, conn,
) -> int:
    """Execute force close for no_momentum/reversal triggers."""
    remaining_qty = _get_remaining_qty(entry, conn)
    if remaining_qty <= 0:
        return 0

    exit_side = "sell" if entry["direction"] == "LONG" else "buy"

    fill_price = market_close_position(
        entry, symbol, exit_side, remaining_qty, exchange, config, conn,
        order_type="drift_close",
    )

    if fill_price is not None:
        # Get SL price for R calculation
        sl_row = conn.execute(
            "SELECT trigger_price FROM orders WHERE scan_id = ? AND order_type = 'stop_loss' LIMIT 1",
            (entry["scan_id"],),
        ).fetchone()
        sl_price = float(sl_row["trigger_price"]) if sl_row and sl_row["trigger_price"] else None

        r_val = None
        if sl_price and entry.get("fill_price"):
            r_val = _calc_r(float(entry["fill_price"]), fill_price, sl_price, entry["direction"])

        cancel_remaining_orders(entry["scan_id"], "", symbol, config, conn)
        _send_drift_close_alert(entry, trigger, fill_price, r_val, config)
        return 1

    _send_urgent_alert(f"DRIFT CLOSE FAILED: {entry['coin']} {entry['direction']} — {trigger}", config)
    return 0
