"""Market-close execution + the typed close-result single owner.

Extracted from `monitor.py` (T-016 Phase 4a, SPEC #27). `market_close_position`
is the sole producer of `CloseResult` (renamed from `MarketCloseResult` — C4#6
already typed the market-close slice; Phase 4a is the rename-and-rehome
single-owner consolidation of KL#9(a)). The two consumers — timeout
`monitor.enforce_position_timeouts` and `drift._execute_force_close` — import
this one shared class; the sync exit-fill terminal is a structurally distinct
terminal that owns its own FILLED write and is NOT a `CloseResult` consumer.

Logger provenance is byte-identical: this module logs through the
`"scanner.execution.monitor"` logger (the literal name, NOT `__name__`) so
`LogRecord.name` is unchanged (CLI/file logs + the `discord_logging`
ERROR-embed keyed on `record.name`). One-directional: this module imports
only `.config` + `.storage`; nothing imports back into it except via the
`monitor` re-export.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import ExecutionConfig
from .storage import OPEN_QTY_EPSILON, make_cloid, new_order_id

log = logging.getLogger("scanner.execution.monitor")


def _extract_fill_price(exch_order: dict) -> float | None:
    """Extract fill price from exchange order. Returns None if unavailable.

    Uses 'average' (avg fill price) only. The 'price' field is unreliable for
    trigger orders (TP/SL) on HyperLiquid — it may return the mark price
    instead of the actual fill price.
    """
    raw = exch_order.get("average")
    if raw is not None and float(raw) > 0:
        return float(raw)
    return None


def _resolve_close_fill(src: dict) -> tuple[float | None, float | None, str]:
    """Resolve (fill_price, filled_qty, status) from a ccxt order / fetch_order dict.

    fill_price: `average` first, then `price` when > 0 — a market close is NOT a
    trigger order, so `price` is a usable fill reference here (unlike
    `_extract_fill_price`'s average-only rule, which exists because TP/SL trigger
    orders report mark price in `price`). filled_qty: `None` when the field is
    absent — distinct from an explicit `0`: `status=="closed"` with filled
    absent is a full fill, with explicit `0` an exchange auto-cancel (the
    existing monitor convention at the `sync_order_status` "closed"/"canceled"
    branches). status: lowercased ("" when absent).
    """
    fill_price = _extract_fill_price(src)
    if fill_price is None:
        raw_p = src.get("price")
        if raw_p is not None and float(raw_p) > 0:
            fill_price = float(raw_p)
    filled_raw = src.get("filled")
    filled = float(filled_raw) if filled_raw is not None else None
    status = (src.get("status") or "").lower()
    return fill_price, filled, status


@dataclass(frozen=True)
class CloseResult:
    """Typed outcome of `market_close_position` (KL#9(a) — the cross-path
    market-close-result single owner; renamed from `MarketCloseResult` in
    T-016 Phase 4a, rehomed here with its sole producer).

    `market_close_position` re-derives fill state from the exchange but no
    longer self-persists; the caller owns persistence so the close-state write
    can commit atomically with its counterpart-CANCELLED marks (the atomic
    full-close boundary — load-bearing invariant: no DB txn spans an exchange
    REST call). The three-way `kind` preserves Fix-3's safety exactly — a flat
    `payload | None` could not (a truthy partial would make callers cancel the
    SL + fire a false full-close alert; a None partial would lose the truthful
    partial row):

      - "full"    → `payload` is the close `save_order` dict; caller composes
                    the atomic close+cancel transaction. `fill_price` set.
      - "partial" → `payload` is the truthful partial leg (actual filled qty);
                    caller persists ONLY that row and RETAINS the SL/counterparts
                    (remainder retried next cycle). `fill_price` set.
      - "none"    → no row, no cancel (zero/unconfirmed/unpriceable/exhausted);
                    `payload`/`fill_price` are None.
    """

    kind: str
    payload: dict | None
    fill_price: float | None


# Shared sentinel — every non-confirmed/exception/exhausted path returns this.
_CLOSE_NONE = CloseResult("none", None, None)


def market_close_position(
    entry: dict, symbol: str, exit_side: str, qty: float,
    exchange, config: ExecutionConfig,
    order_type: str = "timeout_close",
) -> CloseResult:
    """Place a reduceOnly market order to close a position. 3 retries.

    Returns a `CloseResult` — persistence is the CALLER's job (so the
    close row commits atomically with counterpart cancels; see the type's
    docstring). `kind="full"` only on a confirmed FULL priced close;
    `kind="partial"` carries the truthful partial leg (caller retains the SL,
    remainder retried next cycle); `kind="none"` on a zero/unconfirmed/
    unpriceable result (NO row — a `FILLED`/NULL-price row would fabricate a
    close, the 2026-04-17 phantom_sl class) and on retry exhaustion.
    `create_order` succeeding is not proof of a fill — HL has no native market
    order; ccxt simulates via limit-IOC which can under/not-fill, so fill state
    is re-derived from `status`/`filled` (T-014(a), SPEC #24). The confirmed-
    fill GATING is preserved verbatim from Fix-3 — only persistence moved out.

    HyperLiquid has no native market orders — ccxt simulates via limit IOC
    with slippage. Requires current price as the base for slippage calc.
    """
    order_id = new_order_id()
    cloid = make_cloid(order_id)
    rounded_qty = float(exchange.amount_to_precision(symbol, qty))

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            ticker = exchange.fetch_ticker(symbol)
            price = ticker["last"]
            resp = exchange.create_order(
                symbol, "market", exit_side, rounded_qty, price,
                {"reduceOnly": True, "clientOrderId": cloid},
            )
            exch_id = resp.get("id", "")
            fill_price, filled, status = _resolve_close_fill(resp)
            # `price` may merely echo the submitted IOC limit, so it is NOT
            # fill evidence unless the order is `closed`. Re-derive from the
            # exchange when the create resp is not a confirmed priced close.
            if (status != "closed" or fill_price is None) and exch_id:
                try:
                    fill_price, filled, status = _resolve_close_fill(
                        exchange.fetch_order(exch_id, symbol)
                    )
                except Exception as fexc:
                    log.warning(
                        "%s fetch_order(%s) for %s failed: %s",
                        order_type, exch_id, entry["coin"], fexc,
                    )

            # Confirmed priced close requires status closed AND a price AND a
            # non-zero fill (filled absent ≠ explicit 0 — see _resolve_close_fill).
            confirmed = status == "closed" and fill_price is not None and filled != 0
            if confirmed:
                full = filled is None or filled + OPEN_QTY_EPSILON >= rounded_qty
                recorded_qty = rounded_qty if full else filled
                # Two separate now() calls — byte-identical to the pre-recontract
                # save_order payload (only the persistence boundary moved).
                payload = {
                    "order_id": order_id,
                    "scan_id": entry["scan_id"],
                    "coin": entry["coin"],
                    "direction": entry["direction"],
                    "order_type": order_type,
                    "exchange_order_id": exch_id,
                    "status": "FILLED",
                    "fill_price": fill_price,
                    "quantity": recorded_qty,
                    "exchange": config.exchange,
                    "execution_mode": config.mode,
                    "filled_at": datetime.now(timezone.utc).isoformat(),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                if full:
                    log.info(
                        "%s %s %s: qty %s at %s",
                        order_type, entry["coin"], entry["direction"], recorded_qty, fill_price,
                    )
                    return CloseResult("full", payload, fill_price)
                log.warning(
                    "%s %s %s: PARTIAL close %s/%s at %s — remainder open, "
                    "retried next monitor cycle",
                    order_type, entry["coin"], entry["direction"],
                    recorded_qty, rounded_qty, fill_price,
                )
                return CloseResult("partial", payload, fill_price)

            # Not a confirmed priced close (status not closed, zero fill, or
            # unpriceable). Write NO row — a FILLED/NULL-price row fabricates a
            # close (phantom_sl class). No row to reconcile against, so the
            # durable exchange id is preserved in the log.
            log.error(
                "%s for %s NOT confirmed (status=%s filled=%s price=%s) — no row "
                "written; order_id=%s cloid=%s exchange_id=%s; reconcile from exchange",
                order_type, entry["coin"], status, filled, fill_price,
                order_id, cloid, exch_id,
            )
            return _CLOSE_NONE
        except Exception as exc:
            log.error(
                "%s attempt %d/%d for %s failed: %s",
                order_type, attempt, max_retries, entry["coin"], exc,
            )
            if attempt < max_retries:
                time.sleep(attempt)

    log.error("All %s attempts exhausted for %s — position still open", order_type, entry["coin"])
    return _CLOSE_NONE
