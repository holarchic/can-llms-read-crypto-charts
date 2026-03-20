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
    get_exchange_open_orders,
    get_open_positions,
)
from .kill_switch import is_killed
from .sanity import check_price_sanity, is_price_in_entry_zone
from .sizing import PositionSize, calculate_position
from .storage import (
    count_open_longs,
    count_open_positions,
    get_open_position_coins,
    has_open_position,
    has_unprotected_positions,
    make_cloid,
    new_order_id,
    save_order,
)

log = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    executed: bool = False
    sl_failed: bool = False
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

    # Direction-safe price geometry
    entry_low = machine_block.get("entry_low", 0)
    entry_high = machine_block.get("entry_high", 0)
    stop = machine_block.get("stop_loss", 0)
    tp1 = machine_block.get("tp1")
    tp2 = machine_block.get("tp2")

    if entry_low and entry_high and entry_low > entry_high:
        errors.append(f"entry_low ({entry_low}) > entry_high ({entry_high})")

    if direction == "LONG":
        if stop and entry_low and stop >= entry_low:
            errors.append(f"LONG stop ({stop}) must be below entry_low ({entry_low})")
        if tp1 and entry_high and tp1 <= entry_high:
            errors.append(f"LONG tp1 ({tp1}) must be above entry_high ({entry_high})")
        if tp2 and tp1 and tp2 <= tp1:
            errors.append(f"LONG tp2 ({tp2}) must be above tp1 ({tp1})")
    elif direction == "SHORT":
        if stop and entry_high and stop <= entry_high:
            errors.append(f"SHORT stop ({stop}) must be above entry_high ({entry_high})")
        if tp1 and entry_low and tp1 >= entry_low:
            errors.append(f"SHORT tp1 ({tp1}) must be below entry_low ({entry_low})")
        if tp2 and tp1 and tp2 >= tp1:
            errors.append(f"SHORT tp2 ({tp2}) must be below tp1 ({tp1})")

    return errors


def current_open_notional(
    exec_config: ExecutionConfig,
    conn,
    exchange_positions: list[dict] | None = None,
) -> float:
    """Sum notional of open positions.

    When exchange_positions is provided, uses exchange truth (size * entry_price).
    Otherwise falls back to DB positions with live price lookups.
    """
    if exchange_positions is not None:
        return sum(
            abs(pos.get("size", 0)) * pos.get("entry_price", 0)
            for pos in exchange_positions
        )

    # Fallback to DB positions with live price lookups
    positions = get_open_position_coins(conn, exec_config.exchange, exec_config.mode)
    if not positions:
        return 0.0

    exchange = get_auth_exchange(exec_config)
    meta = get_exchange_meta(exec_config.exchange)
    total = 0.0
    for pos in positions:
        try:
            symbol = meta["symbol_template"].format(coin=pos["coin"])
            ticker = exchange.fetch_ticker(symbol)
            price = float(ticker["last"])
        except Exception:
            price = pos["entry_price"]  # fallback
        total += price * pos["remaining_qty"]
    return total


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
    er.direction = result.setup_signal.direction.value if result.setup_signal else ""

    # 0. Staleness guard — reject signals older than 30 minutes
    #    Use detected_at (screener time) when available, not analysis completion time
    staleness_ref = result.detected_at or result.timestamp
    age_seconds = (datetime.now(timezone.utc) - staleness_ref).total_seconds()
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
            if exec_config.mode == "live":
                er.skip_reason = "btc_vol_gate_error"
                er.error = f"BTC vol gate fetch failed: {exc}"
                log.error("BTC vol gate fetch failed in LIVE: %s — blocking trade", exc)
                return er
            log.warning("BTC volatility gate fetch failed: %s — proceeding (testnet)", exc)

    # 0.6 Exchange position reconciliation — ground truth from exchange
    exchange_positions = None
    try:
        exchange_positions = get_open_positions(exec_config)
        db_count = count_open_positions(conn, exec_config.exchange, exec_config.mode)
        exch_count = len(exchange_positions)
        if db_count != exch_count:
            er.skip_reason = "position_mismatch"
            er.error = f"DB positions={db_count} vs exchange={exch_count} — reconcile before trading"
            log.error("Position count mismatch: DB=%d, exchange=%d — blocking trade", db_count, exch_count)
            return er
    except Exception as exc:
        if exec_config.mode == "live":
            er.skip_reason = "reconciliation_error"
            er.error = f"exchange position fetch failed: {exc}"
            log.error("Reconciliation failed in LIVE: %s — blocking trade", exc)
            return er
        log.warning("Exchange position fetch failed: %s — falling back to DB (testnet)", exc)

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

    # 7. Position limits (prefer exchange truth when available)
    exchange = exec_config.exchange
    mode = exec_config.mode

    if exchange_positions is not None:
        exch_count = len(exchange_positions)
        if exch_count >= exec_config.max_concurrent_positions:
            er.skip_reason = "position_limit"
            log.info("Position limit reached (exchange: %d) — skipping %s", exch_count, result.coin)
            return er

        exch_has_dup = any(
            p["coin"] == result.coin
            for p in exchange_positions
        )
        if exch_has_dup:
            er.skip_reason = "duplicate_position"
            log.info("Duplicate position %s (exchange) — skipping", result.coin)
            return er

        exch_long_count = sum(1 for p in exchange_positions if p["direction"] == "LONG")
        if er.direction == "LONG" and exch_long_count >= exec_config.max_open_longs:
            er.skip_reason = "long_limit"
            log.info("Long limit reached (exchange: %d) — skipping %s", exch_long_count, result.coin)
            return er
    else:
        if count_open_positions(conn, exchange, mode) >= exec_config.max_concurrent_positions:
            er.skip_reason = "position_limit"
            log.info("Position limit reached — skipping %s", result.coin)
            return er

        if has_open_position(result.coin, None, conn, exchange, mode):
            er.skip_reason = "duplicate_position"
            log.info("Duplicate position %s (DB) — skipping", result.coin)
            return er

        if er.direction == "LONG" and count_open_longs(conn, exchange, mode) >= exec_config.max_open_longs:
            er.skip_reason = "long_limit"
            log.info("Long limit reached — skipping %s", result.coin)
            return er

    # 7.5 Unprotected position guard — block all entries if any position lacks SL
    if has_unprotected_positions(conn, exchange, mode):
        er.skip_reason = "unprotected_position"
        log.error("Unprotected position detected — blocking all new entries")
        return er

    # 7.6 Open order cap — prevent HL rejection at 1000 orders
    if exec_config.max_open_orders > 0:
        try:
            open_orders = get_exchange_open_orders(exec_config)
            if len(open_orders) >= exec_config.max_open_orders:
                er.skip_reason = "open_order_limit"
                er.error = f"open_orders={len(open_orders)} >= max={exec_config.max_open_orders}"
                log.warning("Open order cap reached (%d) — skipping %s", len(open_orders), result.coin)
                return er
        except Exception as exc:
            if exec_config.mode == "live":
                er.skip_reason = "order_count_error"
                er.error = f"open order count failed: {exc}"
                log.error("Open order count failed in LIVE: %s — blocking trade", exc)
                return er
            log.warning("Open order count failed: %s — proceeding (testnet)", exc)

    try:
        balance = fetch_balance(exec_config)
    except Exception as exc:
        er.skip_reason = "balance_fetch_failed"
        er.error = str(exc)
        log.warning("Balance fetch failed: %s", exc)
        _send_errors_alert(f"Balance fetch failed: {exc}", exec_config)
        return er

    # 7.6 Zero balance guard
    if balance <= 0:
        er.skip_reason = "zero_balance"
        er.error = f"balance=${balance:.2f}"
        log.warning("Zero or negative balance ($%.2f) — skipping %s", balance, result.coin)
        return er

    if exec_config.max_portfolio_exposure_pct < 100:
        open_notional = current_open_notional(
            exec_config, conn, exchange_positions=exchange_positions,
        )
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

    # 8b. Margin guard — reject if margin exceeds available balance
    if pos.margin_required > balance:
        er.skip_reason = "insufficient_margin"
        er.error = f"margin=${pos.margin_required:.2f} > balance=${balance:.2f}"
        log.warning("%s: margin $%.2f exceeds balance $%.2f",
                    result.coin, pos.margin_required, balance)
        return er

    # 8c. Max notional per trade
    if exec_config.max_notional_usd > 0 and pos.notional_value > exec_config.max_notional_usd:
        er.skip_reason = "notional_limit"
        er.error = (f"notional=${pos.notional_value:.2f} > "
                    f"max=${exec_config.max_notional_usd:.2f}")
        log.warning("%s: notional $%.2f exceeds max $%.2f",
                    result.coin, pos.notional_value, exec_config.max_notional_usd)
        return er

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

        fill_price = float(entry_resp.get("average", entry_resp.get("price", 0)))
        save_order({
            "order_id": entry_oid,
            "scan_id": result.scan_id,
            "coin": result.coin,
            "direction": er.direction,
            "order_type": "entry",
            "exchange_order_id": entry_resp.get("id", ""),
            "status": "FILLED",
            "entry_price": fill_price,
            "fill_price": fill_price,
            "quantity": filled_qty,
            "leverage": pos.leverage,
            "placed_at": now,
            "filled_at": now,
            "exchange": exec_config.exchange,
            "execution_mode": exec_config.mode,
            "created_at": now,
        }, conn)

        # Log risk delta: sizing uses entry_mid but fill is at market price
        intended_stop_dist = pos.stop_distance
        actual_stop_dist = abs(fill_price - mb["stop_loss"])
        if intended_stop_dist > 0:
            risk_delta_pct = (actual_stop_dist - intended_stop_dist) / intended_stop_dist * 100
            actual_risk = pos.risk_amount * (actual_stop_dist / intended_stop_dist)
            entry_mid = pos.notional_value / pos.quantity
            log.info(
                "%s risk delta: intended $%.2f, actual $%.2f (%+.1f%%) | "
                "entry_mid=%.4f fill=%.4f sl=%.4f",
                result.coin, pos.risk_amount, actual_risk, risk_delta_pct,
                entry_mid, fill_price, mb["stop_loss"],
            )
    except Exception as exc:
        er.skip_reason = "entry_failed"
        er.error = str(exc)
        log.error("%s: entry order failed — %s", result.coin, exc)
        _send_errors_alert(f"Entry order failed for {result.coin}: {exc}", exec_config)
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
        er.sl_failed = True
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
            log.warning("SL cloid duplicate for %s — querying exchange for actual order", symbol)
            try:
                open_orders = exchange.fetch_open_orders(symbol)
                cloid_hex = params.get("clientOrderId", "")
                for oo in open_orders:
                    if oo.get("clientOrderId") == cloid_hex:
                        log.info("Found existing SL order %s for %s", oo["id"], symbol)
                        return oo
            except Exception as fetch_exc:
                log.warning("Failed to fetch orders after DuplicateOrderId: %s", fetch_exc)
            return None  # Triggers urgent alert — better than phantom record
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
    if not config.discord_ops_webhook_url:
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

    if exec_result.sl_failed:
        embed["title"] += " [NO SL]"
        embed["fields"].append({"name": "Stop Loss", "value": "FAILED — position unprotected", "inline": False})

    try:
        resp = requests.post(
            config.discord_ops_webhook_url,
            json={"embeds": [embed]},
            timeout=10,
        )
        return resp.status_code == 204
    except Exception as exc:
        log.warning("Discord execution alert failed: %s", exc)
        return False


def _send_urgent_alert(message: str, config: ExecutionConfig) -> bool:
    """Send URGENT Discord alert (e.g., SL placement failure)."""
    webhook = config.discord_urgent_webhook_url
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
    webhook = config.discord_ops_webhook_url
    if not webhook:
        return False

    try:
        resp = requests.post(webhook, json={"content": message}, timeout=10)
        return resp.status_code == 204
    except Exception as exc:
        log.warning("Ops alert failed: %s", exc)
        return False


def _send_errors_alert(message: str, config: ExecutionConfig) -> bool:
    """Send alert to Discord errors channel."""
    webhook = config.discord_errors_webhook_url
    if not webhook:
        return False

    try:
        resp = requests.post(webhook, json={"content": message}, timeout=10)
        return resp.status_code == 204
    except Exception as exc:
        log.warning("Errors alert failed: %s", exc)
        return False


def preflight_check(exec_config: ExecutionConfig) -> list[str]:
    """Validate execution readiness. Returns list of failures (empty = OK)."""
    failures = []
    # 1. Exchange auth + balance
    try:
        balance = fetch_balance(exec_config)
        if balance <= 0:
            failures.append(f"Zero balance: ${balance:.2f}")
    except Exception as exc:
        failures.append(f"Exchange auth/balance failed: {exc}")
    # 2. Kill switch readable
    try:
        is_killed(exec_config)
    except Exception as exc:
        failures.append(f"Kill switch check failed: {exc}")
    return failures


_CRITICAL_SKIP_REASONS = frozenset({
    "position_mismatch", "reconciliation_error", "unprotected_position",
    "kill_switch", "btc_vol_gate_error", "order_count_error",
})

_ERROR_SKIP_REASONS = frozenset({
    "entry_failed", "balance_fetch_failed",
})


def send_skip_alert(er: ExecutionResult, config: ExecutionConfig) -> bool:
    """Send Discord notification when a CONFIRM is skipped (not executed).

    Critical skips get @everyone + BLOCKED prefix → urgent channel.
    Error skips (entry_failed, balance_fetch_failed) → errors channel.
    Routine skips get SKIPPED → ops channel.
    """
    if not er.skip_reason:
        return False

    detail = f" ({er.error})" if er.error else ""
    if er.skip_reason in _CRITICAL_SKIP_REASONS:
        message = f"@everyone **BLOCKED** {er.coin} {er.direction}: {er.skip_reason}{detail}"
        webhook = config.discord_urgent_webhook_url
        if not webhook:
            return False
        try:
            resp = requests.post(webhook, json={"content": message}, timeout=10)
            return resp.status_code == 204
        except Exception as exc:
            log.warning("Critical skip alert failed: %s", exc)
            return False
    elif er.skip_reason in _ERROR_SKIP_REASONS:
        message = f"**SKIPPED** {er.coin} {er.direction}: {er.skip_reason}{detail}"
        return _send_errors_alert(message, config)
    else:
        message = f"**SKIPPED** {er.coin} {er.direction}: {er.skip_reason}{detail}"
        return _send_ops_alert(message, config)
