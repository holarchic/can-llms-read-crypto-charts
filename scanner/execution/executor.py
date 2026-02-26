"""Trade execution orchestrator — takes CONFIRM results, places orders."""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

from scanner.config import get_exchange_meta
from scanner.models import ScanResult

from .config import ExecutionConfig
from .exchange import (
    get_auth_exchange,
    fetch_current_price,
    fetch_balance,
)
from .kill_switch import is_killed
from .sanity import check_price_sanity, is_price_in_entry_zone
from .sizing import calculate_position, PositionSize
from .storage import (
    new_order_id,
    save_order,
    count_open_positions,
    has_open_position,
    count_open_longs,
    sum_open_notional,
)

log = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    executed: bool = False
    scan_id: str = ""
    coin: str = ""
    direction: str = ""
    skip_reason: str = ""
    entry_order_id: str = ""
    sl_order_id: str = ""
    tp1_order_id: str = ""
    tp2_order_id: str = ""
    position_size: PositionSize | None = None
    error: str = ""


def validate_execution_params(machine_block: dict) -> list[str]:
    """Check that required machine_block fields exist and are valid."""
    errors = []
    required = ["direction", "entry_low", "entry_high", "stop_loss", "leverage"]
    for key in required:
        if key not in machine_block or machine_block[key] is None:
            errors.append(f"Missing required field: {key}")

    direction = machine_block.get("direction", "").upper()
    if direction not in ("LONG", "SHORT"):
        errors.append(f"Invalid direction: {direction}")

    leverage = machine_block.get("leverage")
    if leverage is not None and (not isinstance(leverage, (int, float)) or leverage < 1):
        errors.append(f"Invalid leverage: {leverage}")

    return errors


def execute_trade(
    result: ScanResult,
    exec_config: ExecutionConfig,
    scanner_config,
    conn,
) -> ExecutionResult:
    """Execute a trade for a CONFIRM result with a valid machine block.

    Steps 1-8 are pre-checks (fail = alert only, zero orders placed).
    Steps 9-11 place orders. Step 10 (SL) retries with URGENT alert on failure.
    """
    er = ExecutionResult(scan_id=result.scan_id, coin=result.coin)

    # 0. Staleness guard — reject signals older than 30 minutes
    age_seconds = (datetime.now(timezone.utc) - result.timestamp).total_seconds()
    if age_seconds > 1800:
        er.skip_reason = "stale_signal"
        er.error = f"Signal age {age_seconds / 60:.0f}m exceeds 30m limit"
        log.warning("%s: stale signal (%.0fm old) — skipping", result.coin, age_seconds / 60)
        return er

    # 1. Check machine_block exists
    mb = result.machine_block
    if not mb:
        er.skip_reason = "no_machine_block"
        return er

    er.direction = mb.get("direction", "").upper()

    # 2. Validate params
    param_errors = validate_execution_params(mb)
    if param_errors:
        er.skip_reason = "invalid_params"
        er.error = "; ".join(param_errors)
        log.warning("%s %s: invalid params — %s", result.coin, er.direction, er.error)
        return er

    # 3. Fetch current price + ATR
    try:
        current_price = fetch_current_price(result.coin, exec_config)
    except Exception as exc:
        er.skip_reason = "price_fetch_failed"
        er.error = str(exc)
        log.warning("%s: price fetch failed — %s", result.coin, exc)
        return er

    atr = result.setup_signal.context.atr if result.setup_signal else 0

    # 4. Price sanity
    sanity = check_price_sanity(mb, current_price, atr, exec_config)
    if not sanity.passed:
        er.skip_reason = "sanity_fail"
        er.error = "; ".join(sanity.errors)
        log.warning("%s %s: sanity fail — %s", result.coin, er.direction, er.error)
        return er

    # 5. Entry zone check
    entry_low = mb["entry_low"]
    entry_high = mb["entry_high"]
    if not is_price_in_entry_zone(current_price, entry_low, entry_high):
        er.skip_reason = "price_outside_zone"
        er.error = f"Price {current_price} outside [{entry_low}, {entry_high}]"
        log.info("%s %s: price outside entry zone", result.coin, er.direction)
        return er

    # 6. Kill switch
    if is_killed(exec_config):
        er.skip_reason = "kill_switch"
        log.warning("Kill switch active — skipping %s %s", result.coin, er.direction)
        return er

    # 7. Position limits
    exchange = exec_config.exchange

    if count_open_positions(conn, exchange) >= exec_config.max_concurrent_positions:
        er.skip_reason = "position_limit"
        log.info("Position limit reached — skipping %s", result.coin)
        return er

    if has_open_position(result.coin, er.direction, conn, exchange):
        er.skip_reason = "duplicate_position"
        log.info("Duplicate position %s %s — skipping", result.coin, er.direction)
        return er

    if er.direction == "LONG" and count_open_longs(conn, exchange) >= exec_config.max_open_longs:
        er.skip_reason = "long_limit"
        log.info("Long limit reached — skipping %s", result.coin)
        return er

    try:
        balance = fetch_balance(exec_config)
    except Exception as exc:
        er.skip_reason = "balance_fetch_failed"
        er.error = str(exc)
        log.warning("Balance fetch failed: %s", exc)
        _send_ops_alert(f"Balance fetch failed: {exc}", exec_config)
        return er

    if exec_config.max_portfolio_exposure_pct < 100:
        open_notional = sum_open_notional(conn, exchange)
        exposure_pct = (open_notional / balance * 100) if balance > 0 else 0
        if exposure_pct >= exec_config.max_portfolio_exposure_pct:
            er.skip_reason = "exposure_limit"
            log.info("Exposure limit %.1f%% — skipping %s", exposure_pct, result.coin)
            return er

    # 8. Calculate position size
    max_lev = result.setup_signal.max_leverage if result.setup_signal else None
    pos = calculate_position(mb, balance, exec_config.risk_pct, max_lev)
    if pos is None:
        er.skip_reason = "sizing_fail"
        log.warning("%s: position sizing failed", result.coin)
        return er

    er.position_size = pos

    # --- Order placement (past this point, errors are more serious) ---
    meta = get_exchange_meta(exec_config.exchange)
    symbol = meta["symbol_template"].format(coin=result.coin)
    side = "buy" if er.direction == "LONG" else "sell"
    now = datetime.now(timezone.utc).isoformat()

    # 9. Market entry
    try:
        entry_resp = _place_market_order(exec_config, symbol, side, pos.quantity, pos.leverage)
        entry_oid = new_order_id()
        er.entry_order_id = entry_oid
        save_order({
            "order_id": entry_oid,
            "scan_id": result.scan_id,
            "coin": result.coin,
            "direction": er.direction,
            "order_type": "entry",
            "exchange_order_id": entry_resp.get("id", ""),
            "status": "FILLED",
            "entry_price": mb.get("entry_low", 0) + (mb.get("entry_high", 0) - mb.get("entry_low", 0)) / 2,
            "fill_price": float(entry_resp.get("average", entry_resp.get("price", 0))),
            "quantity": pos.quantity,
            "leverage": pos.leverage,
            "placed_at": now,
            "filled_at": now,
            "exchange": exec_config.exchange,
            "execution_mode": exec_config.mode,
            "created_at": now,
        }, conn)
    except Exception as exc:
        er.skip_reason = "entry_failed"
        er.error = str(exc)
        log.error("%s: entry order failed — %s", result.coin, exc)
        _send_ops_alert(f"Entry order failed for {result.coin}: {exc}", exec_config)
        return er

    # 10. Stop loss (3 retries — URGENT if all fail)
    exit_side = "sell" if er.direction == "LONG" else "buy"
    sl_price = mb["stop_loss"]
    sl_resp = _place_stop_loss(exec_config, symbol, exit_side, pos.quantity, sl_price)
    if sl_resp:
        sl_oid = new_order_id()
        er.sl_order_id = sl_oid
        save_order({
            "order_id": sl_oid,
            "scan_id": result.scan_id,
            "coin": result.coin,
            "direction": er.direction,
            "order_type": "stop_loss",
            "exchange_order_id": sl_resp.get("id", ""),
            "status": "PLACED",
            "trigger_price": sl_price,
            "quantity": pos.quantity,
            "exchange": exec_config.exchange,
            "execution_mode": exec_config.mode,
            "placed_at": now,
            "created_at": now,
        }, conn)
    else:
        _send_urgent_alert(
            f"URGENT: SL placement FAILED for {result.coin} {er.direction} "
            f"after 3 retries. Position WITHOUT stop loss!",
            exec_config,
        )

    # 11. Take profits (non-fatal)
    if mb.get("tp1") and pos.tp1_quantity > 0:
        tp1_resp = _place_take_profit(exec_config, symbol, exit_side, pos.tp1_quantity, mb["tp1"])
        if tp1_resp:
            tp1_oid = new_order_id()
            er.tp1_order_id = tp1_oid
            save_order({
                "order_id": tp1_oid,
                "scan_id": result.scan_id,
                "coin": result.coin,
                "direction": er.direction,
                "order_type": "tp1",
                "exchange_order_id": tp1_resp.get("id", ""),
                "status": "PLACED",
                "trigger_price": mb["tp1"],
                "quantity": pos.tp1_quantity,
                "exchange": exec_config.exchange,
                "execution_mode": exec_config.mode,
                "placed_at": now,
                "created_at": now,
            }, conn)

    if mb.get("tp2") and pos.tp2_quantity > 0:
        tp2_resp = _place_take_profit(exec_config, symbol, exit_side, pos.tp2_quantity, mb["tp2"])
        if tp2_resp:
            tp2_oid = new_order_id()
            er.tp2_order_id = tp2_oid
            save_order({
                "order_id": tp2_oid,
                "scan_id": result.scan_id,
                "coin": result.coin,
                "direction": er.direction,
                "order_type": "tp2",
                "exchange_order_id": tp2_resp.get("id", ""),
                "status": "PLACED",
                "trigger_price": mb["tp2"],
                "quantity": pos.tp2_quantity,
                "exchange": exec_config.exchange,
                "execution_mode": exec_config.mode,
                "placed_at": now,
                "created_at": now,
            }, conn)

    er.executed = True
    _send_execution_alert(er, result, exec_config)

    return er


def _place_market_order(config, symbol: str, side: str, quantity: float, leverage: int) -> dict:
    """Set leverage + place market order."""
    exchange = get_auth_exchange(config)
    exchange.set_leverage(leverage, symbol, params={"marginMode": "isolated"})
    return exchange.create_order(symbol, "market", side, quantity)


def _place_stop_loss(
    config, symbol: str, side: str, quantity: float, trigger_price: float, max_retries: int = 3
) -> dict | None:
    """Place stop loss with retry + exponential backoff. Returns None if all retries fail."""
    exchange = get_auth_exchange(config)
    for attempt in range(max_retries):
        try:
            return exchange.create_order(
                symbol, "STOP_MARKET", side, quantity, trigger_price,
                {"stopLossPrice": trigger_price, "reduceOnly": True},
            )
        except Exception as exc:
            log.warning("SL attempt %d/%d failed for %s: %s", attempt + 1, max_retries, symbol, exc)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def _place_take_profit(
    config, symbol: str, side: str, quantity: float, trigger_price: float,
) -> dict | None:
    """Place take profit. Returns None on failure (non-critical)."""
    try:
        exchange = get_auth_exchange(config)
        return exchange.create_order(
            symbol, "TAKE_PROFIT_MARKET", side, quantity, trigger_price,
            {"takeProfitPrice": trigger_price, "reduceOnly": True},
        )
    except Exception as exc:
        log.warning("TP placement failed for %s: %s", symbol, exc)
        return None


def _send_execution_alert(exec_result: ExecutionResult, scan_result: ScanResult, config: ExecutionConfig) -> bool:
    """Send Discord notification for executed trade."""
    if not config.discord_execution_webhook_url:
        return False

    pos = exec_result.position_size
    embed = {
        "title": f"{'🟢' if exec_result.direction == 'LONG' else '🔴'} {exec_result.coin} {exec_result.direction} — EXECUTED",
        "color": 0x26A69A if exec_result.direction == "LONG" else 0xEF5350,
        "fields": [
            {"name": "Mode", "value": config.mode.upper(), "inline": True},
            {"name": "Leverage", "value": f"{pos.leverage}x" if pos else "?", "inline": True},
            {"name": "Quantity", "value": f"{pos.quantity:.6f}" if pos else "?", "inline": True},
            {"name": "Risk", "value": f"${pos.risk_amount:.2f} ({config.risk_pct}%)" if pos else "?", "inline": True},
            {"name": "R:R", "value": f"{pos.rr_ratio:.2f}" if pos else "?", "inline": True},
        ],
    }

    try:
        resp = requests.post(
            config.discord_execution_webhook_url,
            json={"embeds": [embed]},
            timeout=10,
        )
        return resp.status_code == 204
    except Exception as exc:
        log.warning("Discord execution alert failed: %s", exc)
        return False


def _send_urgent_alert(message: str, config: ExecutionConfig) -> bool:
    """Send URGENT Discord alert (e.g., SL placement failure)."""
    webhook = config.discord_execution_webhook_url
    if not webhook:
        return False

    try:
        resp = requests.post(
            webhook,
            json={"content": f"@everyone **URGENT**: {message}"},
            timeout=10,
        )
        return resp.status_code == 204
    except Exception as exc:
        log.error("URGENT alert send failed: %s", exc)
        return False


def _send_ops_alert(message: str, config: ExecutionConfig) -> bool:
    """Send operational Discord alert."""
    webhook = config.discord_execution_webhook_url
    if not webhook:
        return False

    try:
        resp = requests.post(webhook, json={"content": message}, timeout=10)
        return resp.status_code == 204
    except Exception as exc:
        log.warning("Ops alert failed: %s", exc)
        return False
