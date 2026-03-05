"""Trade execution orchestrator — takes CONFIRM results, places orders."""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
from ccxt.base.errors import DuplicateOrderId

from scanner.config import get_exchange_meta
from scanner.models import ScanResult

from .config import ExecutionConfig
from .exchange import (
    fetch_balance,
    fetch_current_price,
    get_auth_exchange,
)
from .kill_switch import is_killed
from .sanity import check_price_sanity, is_price_in_entry_zone
from .sizing import PositionSize, calculate_position
from .storage import (
    count_open_longs,
    count_open_positions,
    has_open_position,
    make_cloid,
    new_order_id,
    save_order,
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

    tp1_pct = machine_block.get("tp1_exit_pct")
    tp2_pct = machine_block.get("tp2_exit_pct", 0)
    if tp1_pct is not None and tp1_pct <= 0:
        errors.append(f"tp1_exit_pct must be positive: {tp1_pct}")
    if tp2_pct < 0:
        errors.append(f"tp2_exit_pct must be non-negative: {tp2_pct}")
    if tp1_pct is not None and tp1_pct + tp2_pct > 100:
        errors.append(f"exit pcts sum ({tp1_pct} + {tp2_pct}) > 100")

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

    # 0.5 BTC volatility gate — block execution during extreme BTC moves
    if exec_config.btc_vol_gate_enabled:
        try:
            from scanner.data import fetch_candles
            from scanner.indicators import compute_atr

            btc_df = fetch_candles("BTC", "4h", 50, scanner_config)
            if btc_df is not None and len(btc_df) >= 20:
                btc_atr = compute_atr(btc_df)
                last_range = float(btc_df["High"].iloc[-1] - btc_df["Low"].iloc[-1])
                if btc_atr > 0 and last_range / btc_atr > exec_config.btc_vol_gate_mult:
                    er.skip_reason = "btc_volatility_gate"
                    er.error = (f"BTC 4H range {last_range:.0f} "
                                f"> {exec_config.btc_vol_gate_mult}x ATR {btc_atr:.0f}")
                    log.warning("%s: %s", result.coin, er.error)
                    return er
        except Exception as exc:
            log.warning("BTC volatility gate fetch failed: %s — proceeding", exc)

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

    # 5. Entry zone check (direction-aware buffer for momentum entries)
    entry_low = mb["entry_low"]
    entry_high = mb["entry_high"]
    if not is_price_in_entry_zone(
        current_price, entry_low, entry_high,
        direction=er.direction, buffer_pct=exec_config.entry_zone_buffer_pct,
    ):
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
    mode = exec_config.mode

    if count_open_positions(conn, exchange, mode) >= exec_config.max_concurrent_positions:
        er.skip_reason = "position_limit"
        log.info("Position limit reached — skipping %s", result.coin)
        return er

    if has_open_position(result.coin, er.direction, conn, exchange, mode):
        er.skip_reason = "duplicate_position"
        log.info("Duplicate position %s %s — skipping", result.coin, er.direction)
        return er

    if er.direction == "LONG" and count_open_longs(conn, exchange, mode) >= exec_config.max_open_longs:
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
        open_notional = sum_open_notional(conn, exchange, mode)
        exposure_pct = (open_notional / balance * 100) if balance > 0 else 0
        if exposure_pct >= exec_config.max_portfolio_exposure_pct:
            er.skip_reason = "exposure_limit"
            log.info("Exposure limit %.1f%% — skipping %s", exposure_pct, result.coin)
            return er

    # 8. Calculate position size
    max_lev = result.setup_signal.max_leverage if result.setup_signal else None
    pos = calculate_position(mb, balance, exec_config.risk_pct, max_lev)
    if pos is None or pos.quantity <= 0:
        er.skip_reason = "sizing_fail"
        er.error = f"balance=${balance:.2f}" if balance <= 0 else "quantity=0"
        log.warning("%s: position sizing failed (balance=$%.2f, qty=%s)",
                    result.coin, balance, f"{pos.quantity}" if pos else "None")
        return er

    er.position_size = pos

    # --- Order placement (past this point, errors are more serious) ---
    meta = get_exchange_meta(exec_config.exchange)
    symbol = meta["symbol_template"].format(coin=result.coin)
    side = "buy" if er.direction == "LONG" else "sell"
    now = datetime.now(timezone.utc).isoformat()

    # 9. Market entry
    try:
        entry_oid = new_order_id()
        entry_resp = _place_market_order(
            exec_config, symbol, side, pos.quantity, pos.leverage,
            cloid=make_cloid(entry_oid),
        )
        er.entry_order_id = entry_oid

        # Use confirmed fill quantity from exchange response (fall back to
        # requested qty when field absent — defensive against non-ccxt mocks)
        raw_filled = entry_resp.get("filled")
        filled_qty = float(raw_filled) if raw_filled is not None else pos.quantity

        # Zero fill — IOC got nothing
        if filled_qty <= 0:
            save_order({
                "order_id": entry_oid,
                "scan_id": result.scan_id,
                "coin": result.coin,
                "direction": er.direction,
                "order_type": "entry",
                "exchange_order_id": entry_resp.get("id", ""),
                "status": "CANCELLED",
                "entry_price": 0,
                "fill_price": 0,
                "quantity": 0,
                "leverage": pos.leverage,
                "placed_at": now,
                "exchange": exec_config.exchange,
                "execution_mode": exec_config.mode,
                "created_at": now,
            }, conn)
            er.skip_reason = "entry_unfilled"
            log.warning("%s: entry order unfilled (IOC zero fill)", result.coin)
            return er

        # Partial fill — warn and alert
        if filled_qty < pos.quantity:
            log.warning(
                "%s: partial fill %.6f / %.6f requested",
                result.coin, filled_qty, pos.quantity,
            )
            _send_ops_alert(
                f"Partial fill {result.coin}: {filled_qty:.6f} / {pos.quantity:.6f}",
                exec_config,
            )

        # Rescale TP quantities proportionally to actual fill
        fill_ratio = filled_qty / pos.quantity
        actual_tp1_qty = pos.tp1_quantity * fill_ratio
        actual_tp2_qty = filled_qty - actual_tp1_qty

        save_order({
            "order_id": entry_oid,
            "scan_id": result.scan_id,
            "coin": result.coin,
            "direction": er.direction,
            "order_type": "entry",
            "exchange_order_id": entry_resp.get("id", ""),
            "status": "FILLED",
            "entry_price": float(entry_resp.get("average", entry_resp.get("price", 0))),
            "fill_price": float(entry_resp.get("average", entry_resp.get("price", 0))),
            "quantity": filled_qty,
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
    sl_oid = new_order_id()
    sl_resp = _place_stop_loss(
        exec_config, symbol, exit_side, filled_qty, sl_price,
        cloid=make_cloid(sl_oid),
    )
    if sl_resp:
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
            "quantity": filled_qty,
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
    if mb.get("tp1") and actual_tp1_qty > 0:
        tp1_oid = new_order_id()
        tp1_resp = _place_take_profit(
            exec_config, symbol, exit_side, actual_tp1_qty, mb["tp1"],
            cloid=make_cloid(tp1_oid),
        )
        if tp1_resp:
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
                "quantity": actual_tp1_qty,
                "exchange": exec_config.exchange,
                "execution_mode": exec_config.mode,
                "placed_at": now,
                "created_at": now,
            }, conn)

    if mb.get("tp2") and actual_tp2_qty > 0:
        tp2_oid = new_order_id()
        tp2_resp = _place_take_profit(
            exec_config, symbol, exit_side, actual_tp2_qty, mb["tp2"],
            cloid=make_cloid(tp2_oid),
        )
        if tp2_resp:
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
                "quantity": actual_tp2_qty,
                "exchange": exec_config.exchange,
                "execution_mode": exec_config.mode,
                "placed_at": now,
                "created_at": now,
            }, conn)

    er.executed = True
    _send_execution_alert(er, result, exec_config)

    return er


def _round_quantity(exchange, symbol: str, quantity: float) -> float:
    """Round quantity to exchange's required precision (szDecimals)."""
    return float(exchange.amount_to_precision(symbol, quantity))


def _place_market_order(
    config, symbol: str, side: str, quantity: float, leverage: int,
    cloid: str | None = None,
) -> dict:
    """Set leverage + place market order.

    HyperLiquid has no native market orders — ccxt simulates via limit IOC
    with 5% slippage. Requires current price as the base for slippage calc.
    """
    exchange = get_auth_exchange(config)
    exchange.set_leverage(leverage, symbol, params={"marginMode": "isolated"})
    quantity = _round_quantity(exchange, symbol, quantity)
    ticker = exchange.fetch_ticker(symbol)
    price = ticker["last"]
    params = {}
    if cloid:
        params["clientOrderId"] = cloid
    return exchange.create_order(symbol, "market", side, quantity, price, params)


def _place_stop_loss(
    config, symbol: str, side: str, quantity: float, trigger_price: float,
    max_retries: int = 3, cloid: str | None = None,
) -> dict | None:
    """Place stop loss with retry + exponential backoff. Returns None if all retries fail."""
    exchange = get_auth_exchange(config)
    quantity = _round_quantity(exchange, symbol, quantity)
    params = {"stopLossPrice": trigger_price, "reduceOnly": True}
    if cloid:
        params["clientOrderId"] = cloid
    for attempt in range(max_retries):
        try:
            return exchange.create_order(
                symbol, "STOP_MARKET", side, quantity, trigger_price,
                params,
            )
        except DuplicateOrderId:
            log.info("SL cloid already exists for %s — prior attempt succeeded", symbol)
            return {"id": "", "status": "duplicate_cloid"}
        except Exception as exc:
            log.warning("SL attempt %d/%d failed for %s: %s", attempt + 1, max_retries, symbol, exc)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def _place_take_profit(
    config, symbol: str, side: str, quantity: float, trigger_price: float,
    cloid: str | None = None,
) -> dict | None:
    """Place take profit. Returns None on failure (non-critical)."""
    try:
        exchange = get_auth_exchange(config)
        quantity = _round_quantity(exchange, symbol, quantity)
        params = {"takeProfitPrice": trigger_price, "reduceOnly": True}
        if cloid:
            params["clientOrderId"] = cloid
        return exchange.create_order(
            symbol, "TAKE_PROFIT_MARKET", side, quantity, trigger_price,
            params,
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
