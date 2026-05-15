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
from scanner.models import (
    Direction,
    MarketContext,
    ScanResult,
    SetupSignal,
    SetupType,
    Structure,
)

from .config import ExecutionConfig
from .exchange import fetch_current_price, get_open_positions
from .sanity import check_price_sanity
from .storage import (
    claim_pending,
    get_active_pending_entries,
    has_active_pending,
    has_open_position,
    increment_candles_checked,
    new_order_id,
    save_pending_entry,
    update_last_checked_hour,
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


def evaluate_limit_at_trigger(
    entry_trigger_price: float,
    invalidation_price: float,
    direction: str,
    current_price: float,
) -> tuple[str | None, str]:
    """Spot-price touch evaluator for limit_at.

    Ruleset §4.2 Trigger 1 — limit-order semantics implemented as
    market-on-touch. Fires when spot reaches the buy/sell level:
    - LONG: current_price ≤ entry_trigger_price (buy level reached)
    - SHORT: current_price ≥ entry_trigger_price (sell level reached)

    Invalidation checked first. On a real limit the exchange fills
    deterministically at the level; we use market-on-touch so we still
    fire if price gaps through with no resting fill — polling resolution
    is 5 min, so tight hairpin sweeps may be missed. Acceptable for
    crypto-perp liquidity where gap risk > price-improvement upside.
    """
    if direction == "LONG" and current_price <= invalidation_price:
        return ("invalidated", "kill")
    if direction == "SHORT" and current_price >= invalidation_price:
        return ("invalidated", "kill")

    if direction == "LONG" and current_price <= entry_trigger_price:
        return ("triggered", "execute")
    if direction == "SHORT" and current_price >= entry_trigger_price:
        return ("triggered", "execute")

    return (None, "none")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_1h_candle(coin: str) -> dict | None:
    """Fetch the latest completed 1H candle for a coin.

    Returns dict with timestamp, open, high, low, close or None on failure.
    Requests 3 candles: [completed-2, completed-1, in-progress].
    Uses iloc[-2] — the most recent completed candle (iloc[-1] is in-progress).
    """
    try:
        from scanner.data import fetch_candles
        df = fetch_candles(coin, "1h", 3)
        if df is None or len(df) < 2:
            return None
        # iloc[-1] is the in-progress candle; iloc[-2] is the latest completed
        ts = df.index[-2]
        row = df.iloc[-2]
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


def _empty_setup_signal(coin: str, direction: Direction) -> SetupSignal:
    """Build a placeholder SetupSignal carrying only the screener direction.

    The reconstructed ScanResult from a triggered pending needs enough of a
    SetupSignal for executor.counter_trend detection. Context fields are
    defaulted — ATR sanity checks fall through when atr==0.
    """
    ctx = MarketContext(
        structure=Structure.RANGING,
        swings=[],
        sr_levels=[],
        msbs=[],
        range_bounds=None,
        atr=0.0,
        current_price=0.0,
        fib_levels=None,
    )
    return SetupSignal(
        coin=coin,
        setup_type=SetupType.A,
        direction=direction,
        confidence=0.0,
        key_level=0.0,
        context=ctx,
        description="conditional-entry trigger fired",
    )


def _reconstruct_scan_result(pending: dict, trigger_close: float) -> ScanResult:
    """Rebuild a ScanResult from a pending entry for execute_trade.

    Shifts SL/TP by (trigger_close - original_entry_mid) so the intended R is
    preserved when price has moved between scan time and trigger time, and
    brackets entry_low/high around the trigger close.

    Restores the screener direction on setup_signal so executor can still
    flag counter-trend entries (0.5x risk, min_rr=3.0, shorter timeout).
    """
    mb = json.loads(pending["machine_block"])

    # Shift SL/TP by the delta between trigger_close and original entry_mid.
    # Same logic as sizing.reanchor_exit_prices but applied to the saved mb.
    orig_entry_low = float(mb.get("entry_low", 0))
    orig_entry_high = float(mb.get("entry_high", 0))
    orig_entry_mid = (orig_entry_low + orig_entry_high) / 2 if orig_entry_low else trigger_close
    delta = trigger_close - orig_entry_mid
    if mb.get("stop_loss") is not None:
        mb["stop_loss"] = float(mb["stop_loss"]) + delta
    if mb.get("tp1") is not None:
        mb["tp1"] = float(mb["tp1"]) + delta
    if mb.get("tp2") is not None:
        mb["tp2"] = float(mb["tp2"]) + delta

    # Bracket the trigger close price for the entry zone
    buffer = trigger_close * 0.005
    mb["entry_low"] = trigger_close - buffer
    mb["entry_high"] = trigger_close + buffer

    setup_signal = None
    screener_dir = pending.get("screener_direction")
    if screener_dir:
        try:
            setup_signal = _empty_setup_signal(pending["coin"], Direction(screener_dir))
        except ValueError:
            log.warning("Unknown screener_direction %s for %s", screener_dir, pending["coin"])

    return ScanResult(
        scan_id=pending["scan_id"],
        coin=pending["coin"],
        timestamp=datetime.now(timezone.utc),
        setup_signal=setup_signal,
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

    # Param validation — same canonical validator the executor runs.
    # Without this backstop, blocks that violate LONG inv>=SL / SHORT inv<=SL
    # (or any other rule) become pending rows that deterministically
    # TRIGGER_FAILED at execute time (ZEC/NEAR 2026-04-29). Reject early.
    from scanner.llm import validate_execution_params
    valid, errors = validate_execution_params(mb)
    if not valid:
        log.warning("%s %s: invalid machine_block — %s",
                    coin, direction, "; ".join(errors))
        return False

    # Counter-trend R:R gate. Executor enforces counter_trend_min_rr at
    # execute time — mirroring it here prevents pendings that would
    # deterministically counter_trend_rr at trigger (DOGE/LTC/WLD
    # 2026-04-28..05-05; 3 of 7 V9.10 TRIGGER_FAILED cases).
    screener_direction = (
        result.setup_signal.direction.value if result.setup_signal else None
    )
    if screener_direction and screener_direction != direction:
        entry_mid = (float(mb["entry_low"]) + float(mb["entry_high"])) / 2
        risk = abs(entry_mid - float(mb["stop_loss"]))
        tp = float(mb.get("tp1") or 0) or float(mb.get("tp2") or 0)
        if risk > 0 and tp > 0:
            rr = abs(tp - entry_mid) / risk
            if rr < exec_config.counter_trend_min_rr:
                log.warning(
                    "%s %s: counter-trend R:R %.2f below min %.1f — rejecting pending",
                    coin, direction, rr, exec_config.counter_trend_min_rr,
                )
                return False

    # Dedup — one active pending per coin
    if has_active_pending(coin, conn, exec_config.exchange, exec_config.mode):
        log.info("Pending entry already active for %s — skipping", coin)
        return False

    # Position dedupe — block if coin already held. Prevents the trigger
    # firing into a duplicate_position skip (TRIGGER_FAILED noise + duplicate
    # Discord alerts). Match executor.py:168 pattern: prefer exchange truth,
    # fall back to DB on testnet exception.
    try:
        if any(p["coin"] == coin for p in get_open_positions(exec_config)):
            log.info("Open position exists for %s — skipping pending creation", coin)
            return False
    except Exception as exc:
        if exec_config.mode == "live":
            log.error("Position fetch failed in LIVE for %s: %s — blocking pending", coin, exc)
            return False
        log.warning("Position fetch failed: %s — falling back to DB (testnet)", exc)
        if has_open_position(coin, None, conn, exec_config.exchange, exec_config.mode):
            log.info("Open position exists for %s (DB) — skipping pending creation", coin)
            return False

    # Price sanity at save time. Catches LLM unit mismatches (BTC 2026-05-05
    # zone in $K vs spot in $) and out-of-tolerance entry deviations before
    # the pending is created. Best-effort: if price fetch fails, proceed and
    # let the executor re-check at trigger time.
    current_price = None
    try:
        current_price = fetch_current_price(coin, exec_config)
    except Exception as exc:
        log.warning("%s: price fetch failed at save — skipping sanity check (%s)", coin, exc)

    if current_price is not None:
        atr = result.setup_signal.context.atr if result.setup_signal else 0
        sanity = check_price_sanity(mb, current_price, atr, exec_config)
        if not sanity.passed:
            log.warning("%s %s: price sanity fail at save — %s",
                        coin, direction, "; ".join(sanity.errors))
            return False

    # Clamp expiry to [1, max]. LLM-emitted expiry_candles=0 otherwise
    # creates a DOA pending that expires on its first monitor tick.
    expiry = max(
        1,
        min(int(mb.get("expiry_candles", 12)), exec_config.conditional_entry_max_expiry),
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
        "screener_direction": screener_direction,
    }

    try:
        save_pending_entry(entry, conn)
        log.info("Conditional entry saved: %s %s — trigger %s @ %.5f, expiry %dh",
                 coin, direction, mb["entry_trigger"], entry["entry_trigger_price"], expiry)
        _send_pending_alert(entry, exec_config)
    except Exception as exc:
        log.error("Failed to save conditional entry for %s: %s", coin, exc)
        return False

    # Immediate check — fast path using the spot price already fetched for
    # save-time sanity. limit_at ONLY: its trigger semantics are spot-touch,
    # so spot IS the real signal and an already-met level should fill now
    # rather than wait up to 60min. candle_close_above/below is deliberately
    # excluded: its trigger is a *completed 1H close*, for which spot is not
    # a proxy — evaluating it off spot here fires an unconfirmed intrabar
    # market entry, defeating the close-confirmation the trigger exists for.
    # candle_close_* waits for the 1H-boundary monitor cycle.
    # If save-time fetch failed, skip — next monitor cycle picks it up.
    if current_price is not None:
        try:
            trigger = entry["entry_trigger"]
            if trigger == "limit_at":
                event, action = evaluate_limit_at_trigger(
                    entry["entry_trigger_price"],
                    entry["invalidation_price"],
                    direction,
                    current_price,
                )
            else:
                _event, action = (None, "none")

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
            log.warning("Immediate trigger check failed for %s — will retry at next cycle: %s", coin, exc)

    return True


def check_pending_entries(exec_config: ExecutionConfig, conn) -> int:
    """Top-level pending entry check.

    Candle-based triggers (candle_close_above/below) run
    only at 1H boundaries — their evaluators depend on completed 1H candles.
    Spot-based triggers (limit_at) run every monitor cycle so a fast price
    sweep through the buy/sell level isn't missed by hour-gated polling.

    Returns count of entries acted on (triggered + invalidated + expired).
    """
    if not exec_config.conditional_entry_enabled:
        return 0

    pending = get_active_pending_entries(conn, exec_config.exchange, exec_config.mode)
    if not pending:
        return 0

    hour_boundary = is_1h_boundary()
    acted = 0
    for entry in pending:
        try:
            trigger = entry.get("entry_trigger", "")
            if trigger == "limit_at":
                acted += _process_limit_at(entry, exec_config, conn)
            elif hour_boundary:
                acted += _process_pending_entry(entry, exec_config, conn)
        except Exception as exc:
            log.error("Pending entry check failed for %s %s: %s",
                      entry.get("coin"), entry.get("direction"), exc)

    return acted


def _process_limit_at(
    entry: dict, config: ExecutionConfig, conn,
) -> int:
    """Process a limit_at pending using spot-price polling (every cycle).

    Expiry is counted in 1H boundary hits, matching the expiry_candles
    semantic used by the other triggers. Dedup by hour tag so the counter
    doesn't advance 12x/hour.
    """
    coin = entry["coin"]
    direction = entry["direction"]
    pending_id = entry["pending_id"]

    if has_open_position(coin, None, conn, config.exchange, config.mode):
        log.info("Position already exists for %s — invalidating pending", coin)
        update_pending_status(pending_id, "INVALIDATED", conn)
        _send_resolution_alert(entry, "invalidated (position exists)", config)
        return 1

    try:
        current_price = fetch_current_price(coin, config)
    except Exception as exc:
        log.warning("limit_at price fetch failed for %s: %s", coin, exc)
        return 0

    event, action = evaluate_limit_at_trigger(
        float(entry["entry_trigger_price"]),
        float(entry["invalidation_price"]),
        direction,
        current_price,
    )

    if action == "execute":
        log.info("limit_at fired for %s %s at %.5f (trigger %.5f)",
                 coin, direction, current_price,
                 float(entry["entry_trigger_price"]))
        return _execute_triggered_entry(entry, current_price, config, conn)

    if action == "kill":
        log.info("limit_at invalidated for %s %s at %.5f",
                 coin, direction, current_price)
        update_pending_status(pending_id, "INVALIDATED", conn)
        _send_resolution_alert(entry, "invalidated", config)
        return 1

    # Expiry — count 1H boundaries, not monitor cycles
    hour_tag = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
    if entry.get("last_checked_hour") == hour_tag:
        return 0

    new_count = increment_candles_checked(pending_id, conn)
    update_last_checked_hour(pending_id, hour_tag, conn)
    if new_count >= entry["expiry_candles"]:
        log.info("limit_at expired for %s %s after %d hours",
                 coin, direction, new_count)
        update_pending_status(pending_id, "EXPIRED", conn)
        _send_resolution_alert(entry, "expired", config)
        return 1

    return 0


def _process_pending_entry(
    entry: dict, config: ExecutionConfig, conn,
) -> int:
    """Process a single pending entry against the latest 1H candle(s).

    Evaluates the trigger against the latest completed 1H candle.

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

    trigger = entry["entry_trigger"]

    single = _fetch_1h_candle(coin)
    if single is None:
        log.warning("No 1H candle for %s — skipping this cycle", coin)
        return 0
    latest = single
    latest_close = single["close"]
    event, action = evaluate_entry_trigger(
        trigger,
        float(entry["entry_trigger_price"]),
        float(entry["invalidation_price"]),
        direction,
        latest_close,
    )

    # Stale candle protection — latest candle timestamp should be within 2 hours
    try:
        candle_time = datetime.fromisoformat(latest["timestamp"])
        if candle_time.tzinfo is None:
            candle_time = candle_time.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - candle_time).total_seconds() / 3600
        if age_hours > 2.0:
            log.warning("Stale 1H candle for %s (%.1fh old) — skipping", coin, age_hours)
            return 0
    except (ValueError, TypeError):
        pass

    # Dedup: skip if we already evaluated this candle hour (monitor fires 3x per boundary)
    candle_hour = latest["timestamp"][:13]  # "2026-04-16T14"
    if entry.get("last_checked_hour") == candle_hour:
        return 0

    if action == "execute":
        update_last_checked_hour(pending_id, candle_hour, conn)
        return _execute_triggered_entry(entry, latest_close, config, conn)

    if action == "kill":
        log.info("Pending %s %s invalidated at %.5f (%s)",
                 coin, direction, latest_close, trigger)
        update_pending_status(pending_id, "INVALIDATED", conn)
        update_last_checked_hour(pending_id, candle_hour, conn)
        _send_resolution_alert(entry, "invalidated", config)
        return 1

    # Neither triggered nor invalidated — increment candle count, check expiry
    new_count = increment_candles_checked(pending_id, conn)
    update_last_checked_hour(pending_id, candle_hour, conn)
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

    Atomically claims the pending (ACTIVE -> TRIGGERED) before touching the
    exchange. If another process (scanner immediate-check vs monitor cycle)
    already claimed it, bail out to avoid double entry. On execute_trade
    failure or skip, flip status to TRIGGER_FAILED so the audit trail shows
    the trigger happened but the trade did not.
    """
    from .executor import execute_trade

    coin = entry["coin"]
    direction = entry["direction"]
    pending_id = entry["pending_id"]

    if not claim_pending(pending_id, "TRIGGERED", conn):
        log.info(
            "Pending %s %s already resolved (race lost) — skipping execute_trade",
            coin, direction,
        )
        return 0

    log.info("Trigger fired for %s %s at %.5f — executing", coin, direction, trigger_close)
    _send_trigger_alert(entry, config)

    scan_result = _reconstruct_scan_result(entry, trigger_close)
    scanner_config = ScannerConfig()

    try:
        er = execute_trade(scan_result, config, scanner_config, conn)
        if er.executed:
            log.info("Conditional entry executed: %s %s (entry=%s, sl=%s)",
                     coin, direction, er.entry_order_id, er.sl_order_id)
        else:
            log.warning("Conditional entry skipped after trigger: %s %s — %s",
                        coin, direction, er.skip_reason or "unknown")
            update_pending_status(pending_id, "TRIGGER_FAILED", conn)
    except Exception as exc:
        log.error("execute_trade failed for triggered %s %s: %s", coin, direction, exc)
        update_pending_status(pending_id, "TRIGGER_FAILED", conn)

    return 1
