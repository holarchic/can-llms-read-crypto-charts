"""Conditional entry monitor — watch 1H candle closes for trigger/invalidation.

Separates setup identification (LLM) from entry execution (mechanical).
The LLM defines entry trigger + invalidation conditions. This module watches
1H candle closes and enters when the trigger fires, or kills the setup if
invalidation fires or expiry elapses.
"""

import json
import logging
from datetime import datetime, timezone

import requests

from scanner.config import ScannerConfig
from scanner.models import ScanResult

from .config import ExecutionConfig
from .exchange import fetch_current_price
from .storage import (
    get_active_pending_entries,
    has_active_pending,
    has_open_position,
    increment_candles_checked,
    new_order_id,
    save_pending_entry,
    update_pending_status,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure functions — no side effects
# ---------------------------------------------------------------------------

def is_1h_boundary() -> bool:
    """Check if current UTC time is within the first 15 minutes of an hour.

    Monitor runs every 5 min, so 3 cycles in this window will match.
    """
    now = datetime.now(timezone.utc)
    return now.minute < 15


def evaluate_entry_trigger(
    entry_trigger: str,
    entry_trigger_price: float,
    invalidation_price: float,
    direction: str,
    candle_close: float,
) -> tuple[str | None, str]:
    """Evaluate a 1H candle close against entry trigger and invalidation.

    Returns (event_type, action):
    - ("triggered", "execute") — candle close confirmed the trigger
    - ("invalidated", "kill") — candle close breached invalidation
    - (None, "none") — neither fired, continue waiting

    Priority: invalidation checked first (safety).
    """
    # Check invalidation first
    if direction == "LONG" and candle_close <= invalidation_price:
        return ("invalidated", "kill")
    if direction == "SHORT" and candle_close >= invalidation_price:
        return ("invalidated", "kill")

    # Check trigger
    if entry_trigger == "candle_close_above" and candle_close >= entry_trigger_price:
        return ("triggered", "execute")
    if entry_trigger == "candle_close_below" and candle_close <= entry_trigger_price:
        return ("triggered", "execute")

    return (None, "none")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_1h_candle(coin: str) -> dict | None:
    """Fetch the latest completed 1H candle for a coin.

    Returns dict with timestamp, open, high, low, close or None on failure.
    """
    try:
        from scanner.data import fetch_candles
        df = fetch_candles(coin, "1h", 2)
        if df is None or df.empty:
            return None
        # Last row is the most recent completed candle
        ts = df.index[-1]
        row = df.iloc[-1]
        return {
            "timestamp": ts.isoformat(),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
        }
    except Exception as exc:
        log.error("Failed to fetch 1H candle for %s: %s", coin, exc)
        return None


def _reconstruct_scan_result(pending: dict, trigger_close: float) -> ScanResult:
    """Rebuild a ScanResult from a pending entry for execute_trade.

    Sets entry_low/entry_high to bracket the trigger close price (+/- 0.5%)
    and detected_at to now so the staleness check passes.
    """
    mb = json.loads(pending["machine_block"])

    # Bracket the trigger close price for the entry zone
    buffer = trigger_close * 0.005
    mb["entry_low"] = trigger_close - buffer
    mb["entry_high"] = trigger_close + buffer

    return ScanResult(
        scan_id=pending["scan_id"],
        coin=pending["coin"],
        timestamp=datetime.now(timezone.utc),
        setup_signal=None,
        decision="CONFIRM",
        confluence_score=None,
        analysis_text=None,
        parsed_decision=None,
        chart_paths=[],
        cost_usd=0.0,
        machine_block=mb,
        llm_confidence=mb.get("confidence"),
        detected_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Discord alerts
# ---------------------------------------------------------------------------

def _send_pending_alert(pending: dict, config: ExecutionConfig) -> bool:
    """Send Discord embed for new conditional entry setup."""
    webhook = config.discord_ops_webhook_url
    if not webhook:
        return False

    embed = {
        "title": f"\u23f3 PENDING — {pending['coin']} {pending['direction']}",
        "color": 0xFFA726,
        "fields": [
            {"name": "Trigger", "value": pending["entry_trigger"], "inline": True},
            {"name": "Trigger Price", "value": f"${pending['entry_trigger_price']:,.5f}", "inline": True},
            {"name": "Invalidation", "value": f"${pending['invalidation_price']:,.5f}", "inline": True},
            {"name": "Expiry", "value": f"{pending['expiry_candles']}h", "inline": True},
        ],
        "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
    }
    try:
        resp = requests.post(webhook, json={"embeds": [embed]}, timeout=10)
        return resp.status_code == 204
    except Exception as exc:
        log.error("Pending alert failed: %s", exc)
        return False


def _send_trigger_alert(pending: dict, config: ExecutionConfig) -> bool:
    """Send Discord embed when entry trigger fires."""
    webhook = config.discord_ops_webhook_url
    if not webhook:
        return False

    embed = {
        "title": f"\u26a1 TRIGGERED — {pending['coin']} {pending['direction']}",
        "color": 0x66BB6A,
        "fields": [
            {"name": "Trigger", "value": pending["entry_trigger"], "inline": True},
            {"name": "Trigger Price", "value": f"${pending['entry_trigger_price']:,.5f}", "inline": True},
            {"name": "Candles Waited", "value": str(pending.get("candles_checked", 0)), "inline": True},
        ],
        "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
    }
    try:
        resp = requests.post(webhook, json={"embeds": [embed]}, timeout=10)
        return resp.status_code == 204
    except Exception as exc:
        log.error("Trigger alert failed: %s", exc)
        return False


def _send_resolution_alert(
    pending: dict, reason: str, config: ExecutionConfig,
) -> bool:
    """Send Discord embed for invalidated/expired setups."""
    webhook = config.discord_ops_webhook_url
    if not webhook:
        return False

    embed = {
        "title": f"\u274c {reason.upper()} — {pending['coin']} {pending['direction']}",
        "color": 0xEF5350,
        "fields": [
            {"name": "Trigger Price", "value": f"${pending['entry_trigger_price']:,.5f}", "inline": True},
            {"name": "Candles Checked", "value": str(pending.get("candles_checked", 0)), "inline": True},
        ],
        "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
    }
    try:
        resp = requests.post(webhook, json={"embeds": [embed]}, timeout=10)
        return resp.status_code == 204
    except Exception as exc:
        log.error("Resolution alert failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Orchestrator — called from pipeline and monitor cycle
# ---------------------------------------------------------------------------

def save_conditional_entry(
    result: ScanResult,
    exec_config: ExecutionConfig,
    conn,
) -> bool:
    """Save a CONFIRM with trigger fields as a pending conditional entry.

    Called from pipeline.stage4_execute when conditional_entry_enabled
    and machine_block has trigger fields.
    Returns True if saved, False on error or duplicate.
    """
    mb = result.machine_block
    if not mb:
        return False

    coin = result.coin
    direction = mb.get("direction", "").upper()

    # Dedup — one active pending per coin
    if has_active_pending(coin, conn, exec_config.exchange, exec_config.mode):
        log.info("Pending entry already active for %s — skipping", coin)
        return False

    expiry = min(
        int(mb.get("expiry_candles", 12)),
        exec_config.conditional_entry_max_expiry,
    )

    entry = {
        "pending_id": new_order_id(),
        "scan_id": result.scan_id,
        "coin": coin,
        "direction": direction,
        "machine_block": json.dumps(mb),
        "entry_trigger": mb["entry_trigger"],
        "entry_trigger_price": float(mb["entry_trigger_price"]),
        "invalidation_price": float(mb["invalidation_price"]),
        "expiry_candles": expiry,
        "exchange": exec_config.exchange,
        "execution_mode": exec_config.mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        save_pending_entry(entry, conn)
        log.info("Conditional entry saved: %s %s — trigger %s @ %.5f, expiry %dh",
                 coin, direction, mb["entry_trigger"], entry["entry_trigger_price"], expiry)
        _send_pending_alert(entry, exec_config)
    except Exception as exc:
        log.error("Failed to save conditional entry for %s: %s", coin, exc)
        return False

    # Immediate check — if trigger already met at current price, execute now
    try:
        current_price = fetch_current_price(coin, exec_config)
        event, action = evaluate_entry_trigger(
            entry["entry_trigger"],
            entry["entry_trigger_price"],
            entry["invalidation_price"],
            direction,
            current_price,
        )
        if action == "execute":
            log.info("Trigger already met for %s %s at %.5f — executing immediately",
                     coin, direction, current_price)
            _execute_triggered_entry(entry, current_price, exec_config, conn)
        elif action == "kill":
            log.info("Invalidation already met for %s %s at %.5f — killing",
                     coin, direction, current_price)
            update_pending_status(entry["pending_id"], "INVALIDATED", conn)
            _send_resolution_alert(entry, "invalidated", exec_config)
    except Exception as exc:
        log.warning("Immediate trigger check failed for %s — will retry at 1H boundary: %s", coin, exc)

    return True


def check_pending_entries(exec_config: ExecutionConfig, conn) -> int:
    """Top-level pending entry check. Called from monitor cycle at 1H boundaries.

    Returns count of entries acted on (triggered + invalidated + expired).
    """
    if not exec_config.conditional_entry_enabled:
        return 0

    if not is_1h_boundary():
        return 0

    pending = get_active_pending_entries(conn, exec_config.exchange, exec_config.mode)
    if not pending:
        return 0

    acted = 0
    for entry in pending:
        try:
            acted += _process_pending_entry(entry, exec_config, conn)
        except Exception as exc:
            log.error("Pending entry check failed for %s %s: %s",
                      entry.get("coin"), entry.get("direction"), exc)

    return acted


def _process_pending_entry(
    entry: dict, config: ExecutionConfig, conn,
) -> int:
    """Process a single pending entry against the latest 1H candle.

    Returns 1 if acted (triggered/invalidated/expired), 0 otherwise.
    """
    coin = entry["coin"]
    direction = entry["direction"]
    pending_id = entry["pending_id"]

    # Check if a position already exists for this coin (e.g., manual trade)
    if has_open_position(coin, None, conn, config.exchange, config.mode):
        log.info("Position already exists for %s — invalidating pending", coin)
        update_pending_status(pending_id, "INVALIDATED", conn)
        _send_resolution_alert(entry, "invalidated (position exists)", config)
        return 1

    candle = _fetch_1h_candle(coin)
    if candle is None:
        log.warning("No 1H candle for %s — skipping this cycle", coin)
        return 0

    # Stale candle protection — candle timestamp should be within 2 hours
    try:
        candle_time = datetime.fromisoformat(candle["timestamp"])
        if candle_time.tzinfo is None:
            candle_time = candle_time.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - candle_time).total_seconds() / 3600
        if age_hours > 2.0:
            log.warning("Stale 1H candle for %s (%.1fh old) — skipping", coin, age_hours)
            return 0
    except (ValueError, TypeError):
        pass

    event, action = evaluate_entry_trigger(
        entry["entry_trigger"],
        float(entry["entry_trigger_price"]),
        float(entry["invalidation_price"]),
        direction,
        candle["close"],
    )

    if action == "execute":
        return _execute_triggered_entry(entry, candle["close"], config, conn)

    if action == "kill":
        log.info("Pending %s %s invalidated at %.5f", coin, direction, candle["close"])
        update_pending_status(pending_id, "INVALIDATED", conn)
        _send_resolution_alert(entry, "invalidated", config)
        return 1

    # Neither triggered nor invalidated — increment candle count, check expiry
    new_count = increment_candles_checked(pending_id, conn)
    if new_count >= entry["expiry_candles"]:
        log.info("Pending %s %s expired after %d candles", coin, direction, new_count)
        update_pending_status(pending_id, "EXPIRED", conn)
        _send_resolution_alert(entry, "expired", config)
        return 1

    return 0


def _execute_triggered_entry(
    entry: dict, trigger_close: float, config: ExecutionConfig, conn,
) -> int:
    """Execute a trade after the entry trigger fires.

    Reconstructs a ScanResult and delegates to execute_trade.
    Returns 1 (always acts — marks TRIGGERED regardless of execution outcome).
    """
    from .executor import execute_trade

    coin = entry["coin"]
    direction = entry["direction"]
    pending_id = entry["pending_id"]

    log.info("Trigger fired for %s %s at %.5f — executing", coin, direction, trigger_close)
    update_pending_status(pending_id, "TRIGGERED", conn)
    _send_trigger_alert(entry, config)

    scan_result = _reconstruct_scan_result(entry, trigger_close)
    scanner_config = ScannerConfig()

    try:
        er = execute_trade(scan_result, config, scanner_config, conn)
        if er.executed:
            log.info("Conditional entry executed: %s %s (entry=%s, sl=%s)",
                     coin, direction, er.entry_order_id, er.sl_order_id)
        elif er.skip_reason:
            log.warning("Conditional entry skipped after trigger: %s %s — %s",
                        coin, direction, er.skip_reason)
    except Exception as exc:
        log.error("execute_trade failed for triggered %s %s: %s", coin, direction, exc)

    return 1
