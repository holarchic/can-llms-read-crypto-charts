"""Monitor alert composition — Discord embed/message builders + R math.

Extracted from `monitor.py` (T-016 Phase 4a, SPEC #27). These helpers own
context-specific embed/message *composition* + webhook selection + the
return-value contract; delivery + the 204-interpretation is the Phase-3
single owner `scanner.execution.alerts.post_embed/post_content/send_urgent`.

Logger provenance is byte-identical: this module logs through the
`"scanner.execution.monitor"` logger (the literal name, NOT `__name__`) so
`LogRecord.name` is unchanged in CLI/file logs and in the buffered Discord
monitor-error embeds keyed on `record.name` (`discord_logging.py`). Same
discipline as the Phase-3 `failure_logger`/`LogRecord.name` preservation.
"""

import logging
from datetime import datetime, timezone

from .config import ExecutionConfig
from .storage import original_stop_price

log = logging.getLogger("scanner.execution.monitor")


def _calc_r(entry_price: float, exit_price: float, sl_price: float, direction: str) -> float | None:
    """Calculate R (risk multiple) for a closed position."""
    risk = abs(entry_price - sl_price)
    if risk <= 0:
        return None
    if direction == "LONG":
        return (exit_price - entry_price) / risk
    else:
        return (entry_price - exit_price) / risk


def _send_fill_alert(
    entry: dict, order: dict, fill_price: float | None, config: ExecutionConfig,
    conn,
) -> bool:
    """Send Discord embed when an exit order (SL/TP) fills."""
    webhook = config.discord_ops_webhook_url
    if not webhook:
        return False

    type_labels = {"stop_loss": "SL", "tp1": "TP1", "tp2": "TP2"}
    label = type_labels.get(order["order_type"], order["order_type"])
    is_win = order["order_type"] in ("tp1", "tp2")
    emoji = "\U0001f7e2" if is_win else "\U0001f534"
    color = 0x26A69A if is_win else 0xEF5350

    # Calculate R value — scaled by fraction of original position this fill covers
    r_val = None
    if fill_price and entry.get("entry_price"):
        # Original (first-placed) SL is the risk anchor — drift sl_move and
        # post-TP1 trail mutate the current SL, so using the order that just
        # filled collapses risk to 0 on drift-flatten. Single owner (KL#2(iv),
        # SPEC #27): the Phase-2a storage.original_stop_price (ORDER BY
        # created_at ASC LIMIT 1, mode+exchange-scoped). `or None` preserves
        # the prior falsy-trigger→no-anchor handling byte-identically; no
        # scan_id (can't look up) → None, same as the old empty-scan_orders
        # path.
        scan_id = entry.get("scan_id")
        sl_price = (
            original_stop_price(scan_id, conn, config.mode, config.exchange)
            if scan_id else None
        ) or None
        # Fallback for SL fills with no original-SL row in the DB. For TP
        # fills there's no fallback — no SL reference, so R stays None.
        if sl_price is None and order["order_type"] == "stop_loss":
            sl_price = order.get("trigger_price")
        if sl_price:
            r_val = _calc_r(float(entry["entry_price"]), fill_price, float(sl_price), entry["direction"])
            # Scale by position fraction (e.g. SL on half position after TP1 = -0.5R not -1R)
            if r_val is not None and order.get("quantity") and entry.get("quantity"):
                frac = float(order["quantity"]) / float(entry["quantity"])
                if frac < 1.0 - 1e-9:
                    r_val *= frac
            # Scale by risk_mult (counter-trend 0.5x = -0.5R not -1R)
            if r_val is not None:
                r_mult = float(entry.get("risk_mult") or 1.0)
                if r_mult < 1.0 - 1e-9:
                    r_val *= r_mult

    fields = []
    if fill_price:
        fields.append({"name": "Fill Price", "value": f"${fill_price:,.5f}", "inline": True})
    if r_val is not None:
        fields.append({"name": "R Value", "value": f"{r_val:+.2f}R", "inline": True})

    embed = {
        "title": f"{emoji} {label} FILLED — {entry['coin']} {entry['direction']}",
        "color": color,
        "fields": fields,
        "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
    }

    from .alerts import post_embed
    return post_embed(webhook, embed, "Fill alert", failure_logger=log)


def _send_timeout_alert(
    entry: dict, age_hours: float, qty: float, config: ExecutionConfig,
    *, fill_price: float | None = None, sl_price: float | None = None,
) -> bool:
    """Send Discord embed for a timed-out position."""
    webhook = config.discord_ops_webhook_url
    if not webhook:
        return False

    fields = [
        {"name": "Duration", "value": f"{age_hours:.1f}h / {config.position_timeout_hours}h limit", "inline": True},
        {"name": "Quantity", "value": str(qty), "inline": True},
    ]

    if fill_price and sl_price and entry.get("entry_price"):
        r_val = _calc_r(float(entry["entry_price"]), fill_price, sl_price, entry["direction"])
        if r_val is not None:
            r_mult = float(entry.get("risk_mult") or 1.0)
            if r_mult < 1.0 - 1e-9:
                r_val *= r_mult
            fields.append({"name": "R Value", "value": f"{r_val:+.2f}R", "inline": True})

    embed = {
        "title": f"\u23f0 TIMEOUT — {entry['coin']} {entry['direction']}",
        "color": 0xFFA726,
        "fields": fields,
        "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
    }

    from .alerts import post_embed
    return post_embed(webhook, embed, "Timeout alert send",
                      failure_logger=log, failure_level=logging.ERROR)


def _send_urgent_sl_alert(coin: str, direction: str, config: ExecutionConfig) -> bool:
    """Send URGENT Discord alert when SL replacement exhausts retries.

    Retry-with-durable-log path so a Discord blip never silences the page.
    """
    from .alerts import send_urgent
    message = (f"@everyone **URGENT**: SL replacement FAILED for {coin} {direction} "
               f"after 3 attempts — position running WITHOUT stop loss")
    return send_urgent(config.discord_urgent_webhook_url, message)


def _send_urgent_close_alert(coin: str, direction: str, config: ExecutionConfig) -> bool:
    """URGENT when a timeout market-close did NOT confirm (zero/partial/
    unpriceable/retry-exhausted → `CloseResult.kind` in {partial,none}).

    The protective SL is RETAINED — counterpart cancels run only on a
    confirmed full close (`kind=="full"`), so the position is NOT naked
    (distinct from `_send_urgent_sl_alert`'s "running WITHOUT stop loss"
    condition). The position/remainder is re-attempted next monitor cycle;
    reconcile from exchange truth.
    """
    from .alerts import send_urgent
    message = (f"@everyone **URGENT**: TIMEOUT close did NOT confirm for {coin} "
               f"{direction} — position retained WITH its stop loss; remainder "
               f"re-attempted next monitor cycle. Reconcile from exchange.")
    return send_urgent(config.discord_urgent_webhook_url, message)


def _send_errors_tp_alert(
    coin: str, direction: str, order_type: str, config: ExecutionConfig,
) -> bool:
    """Send Discord errors-channel alert when TP replacement exhausts retries."""
    webhook = config.discord_errors_webhook_url
    if not webhook:
        return False
    label = order_type.upper()
    message = (f"**{label} rejected by exchange** for {coin} {direction} "
               f"— profit target not placed")
    from .alerts import post_content
    return post_content(webhook, message, "TP errors alert send",
                        failure_logger=log, failure_level=logging.ERROR)
