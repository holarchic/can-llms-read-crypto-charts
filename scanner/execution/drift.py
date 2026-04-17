"""Drift invalidation + trailing stops on 4H candles.

Drift: detects four drift patterns at each 4H candle close and takes mechanical action:
- adverse/stagnation → move SL to breakeven (let market decide)
- no_momentum/reversal → force close (free up the slot)

Trailing: after TP1 fills, trail the SL to recent swing points (ruleset 5.2 System A).
No LLM re-evaluation needed. All triggers derived mechanically.
"""

import json
import logging
from datetime import datetime, timezone

import pandas as pd
import requests

from scanner.config import get_exchange_meta
from scanner.indicators import find_swing_points
from scanner.storage import get_connection as get_scanner_connection

from .config import ExecutionConfig
from .exchange import get_auth_exchange
from .monitor import (
    _calc_r,
    _get_remaining_qty,
    _place_replacement_exit_order,
    cancel_remaining_orders,
    market_close_position,
)
from .storage import (
    get_open_entry_orders,
    get_orders_for_scan,
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

    # 1. No momentum — fires if still below threshold after momentum_check_hours
    if age_hours >= config.drift_momentum_check_hours:
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
# Trailing stop — pure functions
# ---------------------------------------------------------------------------

def _candles_to_df(candles: list[dict]) -> pd.DataFrame:
    """Convert candle dicts (from _fetch_4h_candles) to DataFrame for swing detection."""
    df = pd.DataFrame(candles)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    return df


def find_trailing_sl(
    direction: str,
    current_sl: float,
    candles_4h: list[dict],
    lookback: int = 3,
) -> float | None:
    """Find the trailing SL level from swing points in 4H candles.

    LONG: most recent swing low above current SL (tighten upward).
    SHORT: most recent swing high below current SL (tighten downward).
    Returns new SL price or None if no qualifying swing found.
    """
    if len(candles_4h) < 2 * lookback + 1:
        return None

    df = _candles_to_df(candles_4h)
    swings = find_swing_points(df, lookback)
    if not swings:
        return None

    if direction == "LONG":
        candidates = [s for s in swings if s.type == "low" and s.price > current_sl]
    else:
        candidates = [s for s in swings if s.type == "high" and s.price < current_sl]

    if not candidates:
        return None

    # Most recent qualifying swing
    return candidates[-1].price


# ---------------------------------------------------------------------------
# Actions — exchange operations with side effects
# ---------------------------------------------------------------------------

def _move_sl_to_price(
    entry: dict, sl_order: dict, new_sl: float, symbol: str, exchange,
    config: ExecutionConfig, conn, cloid_suffix: str = ":drift",
) -> bool:
    """Place a new SL at the given price, then cancel the old one.

    Tighten-only — if the current SL is already closer to price than new_sl,
    skip. Delegates to the shared insert-new-then-cancel-old helper so the
    position is never naked across the swap.
    """
    current_trigger = float(sl_order.get("trigger_price", 0))
    direction = entry["direction"]

    if new_sl <= 0 or current_trigger <= 0:
        log.error("SL move: invalid prices new=%s current=%s for %s", new_sl, current_trigger, entry["coin"])
        return False

    # Tighten-only: don't widen the SL
    if direction == "SHORT" and new_sl >= current_trigger:
        log.info("SL already tighter for %s SHORT (current=%s, target=%s)", entry["coin"], current_trigger, new_sl)
        return False
    if direction == "LONG" and new_sl <= current_trigger:
        log.info("SL already tighter for %s LONG (current=%s, target=%s)", entry["coin"], current_trigger, new_sl)
        return False

    qty = float(sl_order.get("quantity", 0))
    ok = _place_replacement_exit_order(
        sl_order, new_sl, qty,
        entry, symbol, exchange, config, conn,
        suffix=cloid_suffix,
    )
    if ok:
        log.info("SL moved for %s %s: %s → %s",
                 entry["coin"], direction, current_trigger, new_sl)
    else:
        _send_urgent_alert(
            f"SL MOVE FAILED: {entry['coin']} {direction} — replacement not placed",
            config,
        )
    return ok


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


def _send_trailing_sl_alert(
    entry: dict, old_sl: float, new_sl: float, config: ExecutionConfig,
) -> bool:
    """Send Discord embed for trailing SL move."""
    webhook = config.discord_ops_webhook_url
    if not webhook:
        return False

    embed = {
        "title": f"\U0001f4c8 TRAIL SL — {entry['coin']} {entry['direction']}",
        "color": 0x42A5F5,
        "fields": [
            {"name": "Old SL", "value": f"${old_sl:,.5f}", "inline": True},
            {"name": "New SL", "value": f"${new_sl:,.5f}", "inline": True},
        ],
        "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
    }
    try:
        resp = requests.post(webhook, json={"embeds": [embed]}, timeout=10)
        return resp.status_code == 204
    except Exception as exc:
        log.error("Trailing SL alert failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Trailing stop — helpers
# ---------------------------------------------------------------------------

def _has_tp1_filled(scan_id: str, conn) -> bool:
    """Check if TP1 has filled for this position."""
    row = conn.execute(
        "SELECT 1 FROM orders WHERE scan_id = ? AND order_type = 'tp1' AND status = 'FILLED' LIMIT 1",
        (scan_id,),
    ).fetchone()
    return row is not None


def _get_current_sl(scan_id: str, conn) -> dict | None:
    """Get the active SL order for a position."""
    row = conn.execute(
        "SELECT * FROM orders WHERE scan_id = ? AND order_type = 'stop_loss' AND status = 'PLACED' LIMIT 1",
        (scan_id,),
    ).fetchone()
    return dict(row) if row else None


def _process_trailing_entry(
    entry: dict, meta: dict, exchange, config: ExecutionConfig, conn,
) -> int:
    """Trail SL to recent swing point for positions where TP1 has filled.

    Returns 1 if SL was moved, 0 otherwise.
    """
    candles = _fetch_4h_candles(entry["coin"])
    if not candles:
        return 0

    sl_order = _get_current_sl(entry["scan_id"], conn)
    if not sl_order:
        return 0

    current_sl = float(sl_order["trigger_price"])
    new_sl = find_trailing_sl(
        entry["direction"], current_sl, candles, config.trailing_sl_lookback,
    )
    if new_sl is None:
        return 0

    symbol = meta["symbol_template"].format(coin=entry["coin"])
    success = _move_sl_to_price(
        entry, dict(sl_order), new_sl, symbol, exchange, config, conn,
        cloid_suffix=":trail",
    )
    if success:
        log.info("Trailing SL moved for %s %s: %s → %s",
                 entry["coin"], entry["direction"], current_sl, new_sl)
        _send_trailing_sl_alert(entry, current_sl, new_sl, config)
        return 1
    return 0


# ---------------------------------------------------------------------------
# Orchestrator — called from monitor cycle
# ---------------------------------------------------------------------------

def _get_position_data(entry: dict, conn) -> dict | None:
    """Build position data dict. Prefers reanchored SL/TP1 from orders table, falls back to scans machine_block."""
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

    mb_tp1 = mb.get("tp1")
    mb_sl = mb.get("stop_loss")
    if mb_tp1 is None or mb_sl is None:
        return None

    # Prefer reanchored values from orders table (actual exchange prices)
    sl_row = conn.execute(
        "SELECT trigger_price FROM orders WHERE scan_id = ? AND order_type = 'stop_loss' AND status = 'PLACED' LIMIT 1",
        (entry["scan_id"],),
    ).fetchone()
    tp1_row = conn.execute(
        "SELECT trigger_price FROM orders WHERE scan_id = ? AND order_type = 'tp1' AND status = 'PLACED' LIMIT 1",
        (entry["scan_id"],),
    ).fetchone()

    sl = float(sl_row["trigger_price"]) if sl_row and sl_row["trigger_price"] is not None else float(mb_sl)
    tp1 = float(tp1_row["trigger_price"]) if tp1_row and tp1_row["trigger_price"] is not None else float(mb_tp1)

    return {
        "entry_price": float(entry.get("fill_price") or entry.get("entry_price") or 0),
        "stop_loss": sl,
        "tp1": tp1,
        "direction": entry["direction"],
        "filled_at": entry.get("filled_at") or entry.get("placed_at") or entry.get("created_at"),
    }


def _fetch_4h_candles(coin: str) -> list[dict]:
    """Fetch recent completed 4H candles for a coin.

    Requests 21 candles (20 completed + 1 in-progress) and drops the last
    (in-progress) candle so all returned candles have final OHLC values.
    Returns list of dicts or empty list on failure.
    """
    try:
        from scanner.data import fetch_candles
        df = fetch_candles(coin, "4h", 21)
        if df is None or len(df) < 2:
            return []
        # Drop the in-progress candle (last row)
        df = df.iloc[:-1]
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
    """Process a single entry for drift or trailing. Returns 1 if acted, 0 otherwise."""
    # After TP1 fills, trail SL instead of running drift triggers
    if config.trailing_sl_enabled and _has_tp1_filled(entry["scan_id"], conn):
        return _process_trailing_entry(entry, meta, exchange, config, conn)

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

    success = _move_sl_to_price(entry, dict(sl_order), entry_price, symbol, exchange, config, conn)
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
