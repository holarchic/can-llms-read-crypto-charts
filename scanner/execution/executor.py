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
    is_rate_limit_error,
)
from .kill_switch import is_killed
from .sanity import check_price_sanity, is_price_in_entry_zone
from .sizing import PositionSize, calculate_position, reanchor_exit_prices
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
    """Check that required machine_block fields exist and are valid.

    Delegates to the canonical validator in scanner.llm. Conditional-entry
    reconstruction bypasses parse_machine_block, so routing every mb through
    the same rules keeps that path from slipping malformed blocks past
    execution pre-checks.
    """
    from scanner.llm import validate_execution_params as _canonical_validate

    _, errors = _canonical_validate(machine_block)
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

    # 0. Staleness guard — reject signals older than the configured limit.
    #    Uses detected_at (screener time) when available so slow LLM stage2
    #    still gives the signal its full staleness budget — but budget itself
    #    (default 45min) has to be wide enough to absorb realistic stage2+
    #    stage3+stage4 latency, not just the price-moved-too-far concern.
    staleness_ref = result.detected_at or result.timestamp
    age_seconds = (datetime.now(timezone.utc) - staleness_ref).total_seconds()
    max_age = exec_config.signal_staleness_max_seconds
    if age_seconds > max_age:
        er.skip_reason = "stale_signal"
        er.error = f"Signal age {age_seconds / 60:.0f}m exceeds {max_age // 60}m limit"
        log.warning("%s: stale signal (%.0fm old) — skipping", result.coin, age_seconds / 60)
        return er

    # 0.5 BTC volatility gate — block execution during extreme BTC moves
    if exec_config.btc_vol_gate_enabled:
        try:
            from scanner.data import fetch_candles
            from scanner.indicators import compute_atr

            btc_df = fetch_candles("BTC", "4h", 50, scanner_config)
            # Drop the in-progress candle so the range check uses a fully
            # formed 4H bar (otherwise early-in-window ranges look small).
            if btc_df is not None and len(btc_df) >= 2:
                btc_df = btc_df.iloc[:-1]
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

    # 0.6 Exchange position reconciliation — ground truth from exchange.
    # Asymmetric: block only when DB undercounts (untracked on-exchange position
    # could run without SL/TP oversight). DB overcount is benign — a closed
    # position not yet synced — and the downstream position-limit check at
    # step 7 already prefers exchange truth, so the overcount can't inflate
    # concurrent-position totals. Regression: 2026-04-17 APT conditional
    # entry blocked by DB=2 / exch=1 during the DOT phantom_sl reconciliation.
    exchange_positions = None
    try:
        exchange_positions = get_open_positions(exec_config)
        db_count = count_open_positions(conn, exec_config.exchange, exec_config.mode)
        exch_count = len(exchange_positions)
        if db_count < exch_count:
            er.skip_reason = "position_mismatch"
            er.error = f"DB positions={db_count} vs exchange={exch_count} — reconcile before trading"
            log.error("Position count mismatch: DB=%d < exchange=%d — blocking trade "
                      "(untracked on-exchange position)", db_count, exch_count)
            return er
        if db_count > exch_count:
            log.warning("Position count drift: DB=%d > exchange=%d — proceeding "
                        "(stale DB open; next sync cycle will reconcile)",
                        db_count, exch_count)
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

    # 1b. Detect counter-trend (screener direction ≠ LLM direction)
    screener_dir = result.setup_signal.direction.value if result.setup_signal else ""
    counter_trend = bool(screener_dir and er.direction and screener_dir != er.direction)
    risk_mult = exec_config.counter_trend_risk_mult if counter_trend else 1.0
    if counter_trend:
        log.info("%s: counter-trend (%s screener vs %s LLM) — risk_mult=%.2f",
                 result.coin, screener_dir, er.direction, risk_mult)

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

    # 8. Calculate position size (reduced risk for counter-trend)
    max_lev = result.setup_signal.max_leverage if result.setup_signal else None
    effective_risk = exec_config.risk_pct * risk_mult
    pos = calculate_position(mb, balance, effective_risk, max_lev)
    if pos is None or pos.quantity <= 0:
        er.skip_reason = "sizing_fail"
        er.error = f"balance=${balance:.2f}" if balance <= 0 else "quantity=0"
        log.warning("%s: position sizing failed (balance=$%.2f, qty=%s)",
                    result.coin, balance, f"{pos.quantity}" if pos else "None")
        return er

    # 8a. Counter-trend adjustments: min R:R + single TP (FTA only)
    if counter_trend:
        if pos.rr_ratio < exec_config.counter_trend_min_rr:
            er.skip_reason = "counter_trend_rr"
            er.error = f"R:R {pos.rr_ratio:.2f} < {exec_config.counter_trend_min_rr} (counter-trend min)"
            log.info("%s %s: counter-trend R:R too low — skipping", result.coin, er.direction)
            return er
        pos.tp1_quantity = pos.quantity
        pos.tp2_quantity = 0
        log.info("%s %s: counter-trend — %.0f%% risk, single TP, min R:R %.1f",
                 result.coin, er.direction, risk_mult * 100, exec_config.counter_trend_min_rr)

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
            cloid=make_cloid(entry_oid), price=current_price,
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

        # Rescale TP quantities proportionally to actual fill.
        # Scale both TP1 and TP2 by the same ratio so a partial entry doesn't
        # asymmetrically oversize TP2 (old code used filled-tp1 as the TP2
        # quantity, which absorbed all the fill slack into TP2).
        fill_ratio = filled_qty / pos.quantity
        actual_tp1_qty = pos.tp1_quantity * fill_ratio
        actual_tp2_qty = pos.tp2_quantity * fill_ratio

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
            "risk_mult": risk_mult,
        }, conn)

        # Reanchor SL/TP to fill price so actual R matches intended R
        reanchored = reanchor_exit_prices(mb, fill_price)
        if reanchored["delta"] != 0:
            log.info(
                "%s reanchored exits: delta=%+.4f fill=%.4f | "
                "sl %.4f→%.4f tp1 %s→%s tp2 %s→%s",
                result.coin, reanchored["delta"], fill_price,
                mb["stop_loss"], reanchored["stop_loss"],
                mb.get("tp1"), reanchored.get("tp1"),
                mb.get("tp2"), reanchored.get("tp2"),
            )
    except Exception as exc:
        er.skip_reason = "entry_failed"
        er.error = str(exc)
        log.error("%s: entry order failed — %s", result.coin, exc)
        _send_errors_alert(f"Entry order failed for {result.coin}: {exc}", exec_config)
        return er

    # --- Exit orders: batch SL + TP1 + TP2 (single API call, ~1s faster) ---
    exit_side = "sell" if er.direction == "LONG" else "buy"
    sl_price = reanchored["stop_loss"]
    sl_oid = new_order_id()
    sl_cloid = make_cloid(sl_oid)

    tp1_price = reanchored.get("tp1", mb.get("tp1"))
    tp1_oid = new_order_id() if tp1_price and actual_tp1_qty > 0 else None
    tp1_cloid = make_cloid(tp1_oid) if tp1_oid else None

    tp2_price = reanchored.get("tp2", mb.get("tp2"))
    tp2_oid = new_order_id() if tp2_price and actual_tp2_qty > 0 else None
    tp2_cloid = make_cloid(tp2_oid) if tp2_oid else None

    # Try batch first — single API call for all exit orders
    sl_resp = tp1_resp = tp2_resp = None
    try:
        batch = _place_exit_orders_batch(
            exec_config, symbol, exit_side,
            filled_qty, sl_price, sl_cloid,
            tp1_price, actual_tp1_qty, tp1_cloid,
            tp2_price, actual_tp2_qty, tp2_cloid,
        )
        sl_resp = batch["sl"]
        tp1_resp = batch["tp1"]
        tp2_resp = batch["tp2"]
        if sl_resp:
            log.info("Batch exit placement OK for %s: SL + %d TPs",
                     result.coin, sum(1 for r in [tp1_resp, tp2_resp] if r))
    except Exception as exc:
        log.warning("Batch exit placement failed for %s: %s — sequential fallback",
                    result.coin, exc)

    # 10. SL: fall back to sequential with retries if batch didn't produce SL
    if sl_resp is None:
        sl_resp = _place_stop_loss(
            exec_config, symbol, exit_side, filled_qty, sl_price,
            cloid=sl_cloid,
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
        # Save FAILED SL so monitor can recover trigger_price later
        save_order({
            "order_id": sl_oid,
            "scan_id": result.scan_id,
            "coin": result.coin,
            "direction": er.direction,
            "order_type": "stop_loss",
            "status": "FAILED",
            "trigger_price": sl_price,
            "quantity": filled_qty,
            "exchange": exec_config.exchange,
            "execution_mode": exec_config.mode,
            "created_at": now,
        }, conn)

        # Emergency close — naked positions are unacceptable
        close_price = _emergency_close_position(
            exec_config, symbol, exit_side, filled_qty, result, er, conn, now,
        )
        if close_price:
            _send_urgent_alert(
                f"SL FAILED for {result.coin} {er.direction} — "
                f"emergency closed at {close_price}",
                exec_config,
            )
            er.executed = True
            _send_execution_alert(er, result, exec_config)
            return er
        else:
            _send_urgent_alert(
                f"URGENT: SL placement FAILED for {result.coin} {er.direction} "
                f"after 3 retries. Emergency close also FAILED. "
                f"Position WITHOUT stop loss!",
                exec_config,
            )
            return er

    # 11. Take profits (non-fatal) — use batch response or fall back to sequential
    if tp1_price and actual_tp1_qty > 0:
        if tp1_resp is None:
            tp1_resp = _place_take_profit(
                exec_config, symbol, exit_side, actual_tp1_qty, tp1_price,
                cloid=tp1_cloid,
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
                "trigger_price": tp1_price,
                "quantity": actual_tp1_qty,
                "exchange": exec_config.exchange,
                "execution_mode": exec_config.mode,
                "placed_at": now,
                "created_at": now,
            }, conn)

    if tp2_price and actual_tp2_qty > 0:
        if tp2_resp is None:
            tp2_resp = _place_take_profit(
                exec_config, symbol, exit_side, actual_tp2_qty, tp2_price,
                cloid=tp2_cloid,
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
                "trigger_price": tp2_price,
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
    cloid: str | None = None, price: float | None = None,
    max_retries: int = 3,
) -> dict:
    """Set leverage + place market order with retry.

    HyperLiquid has no native market orders — ccxt simulates via limit IOC
    with 5% slippage. Requires current price as the base for slippage calc.
    Pass price to skip the redundant fetch_ticker() call. Retries on
    transient network errors; DuplicateOrderId (cloid reuse) returns the
    existing order. Aborts immediately on rate-limit to avoid escalation.
    """
    exchange = get_auth_exchange(config)
    exchange.set_leverage(leverage, symbol, params={"marginMode": "isolated"})
    quantity = _round_quantity(exchange, symbol, quantity)
    if price is None:
        ticker = exchange.fetch_ticker(symbol)
        price = ticker["last"]
    params = {}
    if cloid:
        params["clientOrderId"] = cloid

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return exchange.create_order(symbol, "market", side, quantity, price, params)
        except DuplicateOrderId:
            log.warning("Entry cloid duplicate for %s — querying exchange", symbol)
            try:
                open_orders = exchange.fetch_open_orders(symbol)
                for oo in open_orders:
                    if oo.get("clientOrderId") == cloid:
                        return oo
            except Exception as fetch_exc:
                log.warning("Failed to fetch orders after DuplicateOrderId: %s", fetch_exc)
            raise
        except Exception as exc:
            last_exc = exc
            log.warning(
                "Entry attempt %d/%d failed for %s: %s",
                attempt + 1, max_retries, symbol, exc,
            )
            if is_rate_limit_error(exc):
                log.error("Rate limited — aborting entry retries for %s", symbol)
                raise
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise last_exc if last_exc else RuntimeError("Entry placement failed")


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
                symbol, "market", side, quantity, trigger_price,
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
            if is_rate_limit_error(exc):
                log.error("Rate limited — aborting SL retries for %s", symbol)
                break
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def _place_take_profit(
    config, symbol: str, side: str, quantity: float, trigger_price: float,
    cloid: str | None = None, max_retries: int = 3,
) -> dict | None:
    """Place take profit with retry. Returns None on final failure.

    TP missing is silent expectancy degradation — no SL alert, no visible
    error. Retry on transient, abort on rate-limit. Caller receives None
    and routes to the errors channel alert.
    """
    exchange = get_auth_exchange(config)
    quantity = _round_quantity(exchange, symbol, quantity)
    params = {"takeProfitPrice": trigger_price, "reduceOnly": True}
    if cloid:
        params["clientOrderId"] = cloid

    for attempt in range(max_retries):
        try:
            return exchange.create_order(
                symbol, "market", side, quantity, trigger_price, params,
            )
        except Exception as exc:
            log.warning(
                "TP attempt %d/%d failed for %s: %s",
                attempt + 1, max_retries, symbol, exc,
            )
            if is_rate_limit_error(exc):
                log.error("Rate limited — aborting TP retries for %s", symbol)
                return None
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def _place_exit_orders_batch(
    config, symbol: str, exit_side: str,
    sl_qty: float, sl_price: float, sl_cloid: str,
    tp1_price: float | None, tp1_qty: float, tp1_cloid: str | None,
    tp2_price: float | None, tp2_qty: float, tp2_cloid: str | None,
) -> dict[str, dict | None]:
    """Batch SL + TP1 + TP2 in one create_orders() call (~1s faster).

    Returns {"sl": resp|None, "tp1": resp|None, "tp2": resp|None}.
    Raises on total batch failure (caller falls back to sequential).
    """
    exchange = get_auth_exchange(config)
    orders = []
    labels = []

    # SL (always included)
    orders.append({
        "symbol": symbol, "type": "market", "side": exit_side,
        "amount": _round_quantity(exchange, symbol, sl_qty),
        "price": sl_price,
        "params": {
            "stopLossPrice": sl_price, "reduceOnly": True,
            "clientOrderId": sl_cloid,
        },
    })
    labels.append("sl")

    # TP1 (optional)
    if tp1_price and tp1_qty > 0 and tp1_cloid:
        orders.append({
            "symbol": symbol, "type": "market", "side": exit_side,
            "amount": _round_quantity(exchange, symbol, tp1_qty),
            "price": tp1_price,
            "params": {
                "takeProfitPrice": tp1_price, "reduceOnly": True,
                "clientOrderId": tp1_cloid,
            },
        })
        labels.append("tp1")

    # TP2 (optional)
    if tp2_price and tp2_qty > 0 and tp2_cloid:
        orders.append({
            "symbol": symbol, "type": "market", "side": exit_side,
            "amount": _round_quantity(exchange, symbol, tp2_qty),
            "price": tp2_price,
            "params": {
                "takeProfitPrice": tp2_price, "reduceOnly": True,
                "clientOrderId": tp2_cloid,
            },
        })
        labels.append("tp2")

    responses = exchange.create_orders(orders)

    result = {"sl": None, "tp1": None, "tp2": None}
    cloids: dict[str, str] = {}  # label → clientOrderId for recovery
    for i, label in enumerate(labels):
        cloids[label] = orders[i].get("params", {}).get("clientOrderId", "")
        if i < len(responses) and responses[i] and responses[i].get("id"):
            result[label] = responses[i]

    # HL trigger orders return "waitingForTrigger" with no OID in batch responses.
    # Recover by matching clientOrderId in open orders.
    missing = [label for label in labels if result[label] is None and cloids.get(label)]
    if missing:
        try:
            open_orders = exchange.fetch_open_orders(symbol)
            by_cloid = {o.get("clientOrderId"): o for o in open_orders if o.get("clientOrderId")}
            for label in missing:
                matched = by_cloid.get(cloids[label])
                if matched and matched.get("id"):
                    result[label] = matched
                    log.info("OID recovery: %s %s matched by cloid → %s", symbol, label, matched["id"])
        except Exception as exc:
            log.warning("OID recovery: failed to fetch open orders for %s: %s", symbol, exc)

    return result


def _emergency_close_position(
    config: ExecutionConfig, symbol: str, exit_side: str, quantity: float,
    scan_result: ScanResult, er: ExecutionResult, conn, now: str,
    max_retries: int = 3,
) -> float | None:
    """Market-close a position after SL placement fails. Returns fill price or None."""
    exchange = get_auth_exchange(config)
    quantity = _round_quantity(exchange, symbol, quantity)
    order_id = new_order_id()
    cloid = make_cloid(order_id)

    for attempt in range(1, max_retries + 1):
        try:
            ticker = exchange.fetch_ticker(symbol)
            price = ticker["last"]
            resp = exchange.create_order(
                symbol, "market", exit_side, quantity, price,
                {"reduceOnly": True, "clientOrderId": cloid},
            )
            fill_price = float(
                resp.get("average") or resp.get("price") or 0
            )
            save_order({
                "order_id": order_id,
                "scan_id": scan_result.scan_id,
                "coin": scan_result.coin,
                "direction": er.direction,
                "order_type": "emergency_close",
                "exchange_order_id": resp.get("id", ""),
                "status": "FILLED",
                "fill_price": fill_price,
                "quantity": quantity,
                "exchange": config.exchange,
                "execution_mode": config.mode,
                "filled_at": now,
                "created_at": now,
            }, conn)
            log.info(
                "Emergency close %s %s: qty %s at %s",
                scan_result.coin, er.direction, quantity, fill_price,
            )
            return fill_price or 0.01
        except DuplicateOrderId:
            log.warning(
                "Emergency close DuplicateOrderId for %s — treating as filled",
                scan_result.coin,
            )
            save_order({
                "order_id": order_id,
                "scan_id": scan_result.scan_id,
                "coin": scan_result.coin,
                "direction": er.direction,
                "order_type": "emergency_close",
                "exchange_order_id": "",
                "status": "FILLED",
                "fill_price": float(price),
                "quantity": quantity,
                "exchange": config.exchange,
                "execution_mode": config.mode,
                "filled_at": now,
                "created_at": now,
            }, conn)
            return float(price) or 0.01
        except Exception as exc:
            log.error(
                "Emergency close attempt %d/%d for %s failed: %s",
                attempt, max_retries, scan_result.coin, exc,
            )
            if attempt < max_retries:
                delay = 20 if is_rate_limit_error(exc) else 5
                time.sleep(delay)

    log.error("All emergency close attempts exhausted for %s", scan_result.coin)
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
    """Send URGENT Discord alert (e.g., SL placement failure).

    Retry with backoff; always mirror to data/logs/urgent-alerts.log so a
    Discord outage never silences the paper trail.
    """
    from .alerts import send_urgent
    return send_urgent(
        config.discord_urgent_webhook_url,
        f"@everyone **URGENT**: {message}",
    )


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
        webhook = config.discord_ops_webhook_url
        if not webhook:
            return False
        fields = [{"name": "Reason", "value": er.skip_reason, "inline": True}]
        if er.error:
            fields.append({"name": "Detail", "value": er.error, "inline": True})
        embed = {
            "title": f"\u23ed\ufe0f SKIPPED — {er.coin} {er.direction}",
            "color": 0x999999,
            "fields": fields,
            "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
        }
        try:
            resp = requests.post(webhook, json={"embeds": [embed]}, timeout=10)
            return resp.status_code == 204
        except Exception as exc:
            log.warning("Skip alert failed: %s", exc)
            return False
