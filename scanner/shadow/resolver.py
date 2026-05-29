"""Pure forward resolver for shadow signals (T-020.1, SPEC #41).

`resolve_signal` walks ONE captured CONFIRM forward against an OHLCV frame and returns
a `ShadowOutcome` with a terminal status (or `OPEN`) and, when it became a trade, a
realized R **net of taker fees + slippage**. No DB, no network — the frame is supplied
by the caller (`shadow-resolve` fetches it for the signal's venue).

Fidelity (faithful to the live entry/exit path, deliberately pessimistic for a go-live
gate):
- **Entry, `entry_policy=v1`:** IMMEDIATE fills at the next bar's open; `candle_close_*`
  on the confirming close; `limit_at` when a bar's range spans the limit (flagged
  `low_fidelity` — a 1H frame can't see the true intrabar fill). A fill outside
  `[entry_low, entry_high] ± entry_zone_buffer` is `NO_FILL` (live `price_outside_zone`).
  For conditional entries the live `invalidation_price` is checked **before** the
  trigger each bar (`entry_monitor` ordering): a pre-trigger hit is `INVALIDATED`. No
  trigger within `expiry_candles` bars → `TRIGGER_EXPIRED`.
- **Exit, `exit_policy=v1`:** SL / TP1 partial / TP2 / time-invalidation (`max_hold_bars`).
  **Pessimistic intrabar rule:** a bar whose range spans both the stop and a TP counts
  as SL-first. Drift-to-BE rescue + the post-TP1 trail are `v2` — v1 takes NO BE rescue,
  so every non-TP loser is a full −1R.
- Exits reanchor to the actual fill (`reanchor_exit_prices`) so realized R uses the
  intended stop distance; slippage is adverse on every leg; fees are taker per leg.

Non-trade statuses (`TRIGGER_EXPIRED`/`NO_FILL`/`INVALIDATED`/`DATA_INCOMPLETE`) and
`OPEN` carry `realized_r is None` and are excluded from the expectancy sample.
"""

from datetime import datetime

import pandas as pd

from scanner.execution.sizing import reanchor_exit_prices
from scanner.models import MachineBlock, ShadowOutcome

__all__ = [
    "resolve_signal",
    "ENTRY_POLICY_VERSION", "EXIT_POLICY_VERSION",
    "RESOLVED", "OPEN", "TRIGGER_EXPIRED", "NO_FILL", "INVALIDATED", "DATA_INCOMPLETE",
]

ENTRY_POLICY_VERSION = "v1"
EXIT_POLICY_VERSION = "v1"

RESOLVED = "RESOLVED"
OPEN = "OPEN"
TRIGGER_EXPIRED = "TRIGGER_EXPIRED"
NO_FILL = "NO_FILL"
INVALIDATED = "INVALIDATED"
DATA_INCOMPLETE = "DATA_INCOMPLETE"

# HyperLiquid perp taker fee (~3.5 bps) and a conservative per-side slippage assumption.
# These are venue COST constants, not parameters fitted to outcomes.
_TAKER_FEE = 0.00035
_SLIPPAGE = 0.0005


def _slip(px: float, *, buy: bool, slippage: float) -> float:
    """Adverse slippage: a buy fills higher, a sell fills lower."""
    return px * (1 + slippage) if buy else px * (1 - slippage)


def _resolve_entry(mb, fwd, is_long, slippage, buffer):
    """Returns ('FILL', fill_price, bar_pos, low_fidelity) or (status, None, None, False)."""
    lo_z = mb.entry_low * (1 - buffer)
    hi_z = mb.entry_high * (1 + buffer)
    trig = mb.entry_trigger

    if trig is None:  # IMMEDIATE — fill at the first forward bar's open
        px = float(fwd.iloc[0]["Open"])
        if not (lo_z <= px <= hi_z):
            return (NO_FILL, None, None, False)
        return ("FILL", _slip(px, buy=is_long, slippage=slippage), 0, False)

    inv = mb.invalidation_price
    trigger_px = mb.entry_trigger_price
    expiry = mb.expiry_candles or len(fwd)
    for i in range(min(expiry, len(fwd))):
        hi = float(fwd.iloc[i]["High"])
        lo = float(fwd.iloc[i]["Low"])
        close = float(fwd.iloc[i]["Close"])
        # Invalidation is checked BEFORE the trigger (live entry_monitor ordering): a
        # pre-trigger kill wins over a same-bar trigger.
        if inv is not None and ((is_long and lo <= inv) or (not is_long and hi >= inv)):
            return (INVALIDATED, None, None, False)
        if trig == "candle_close_above" and close >= trigger_px:
            return _conditional_fill(close, lo_z, hi_z, is_long, slippage, i)
        if trig == "candle_close_below" and close <= trigger_px:
            return _conditional_fill(close, lo_z, hi_z, is_long, slippage, i)
        if trig == "limit_at" and lo <= trigger_px <= hi:
            return ("FILL", _slip(trigger_px, buy=is_long, slippage=slippage), i, True)
    return (TRIGGER_EXPIRED, None, None, False)


def _conditional_fill(close, lo_z, hi_z, is_long, slippage, i):
    if not (lo_z <= close <= hi_z):  # momentum carried price past the zone — live rejects
        return (NO_FILL, None, None, False)
    return ("FILL", _slip(close, buy=is_long, slippage=slippage), i, False)


def _exit_leg(entry_fill, target, frac, is_long, slippage, taker_fee, risk):
    """Per-leg (pnl_r, fee_r, slip_r) at `target`, adverse-slipped and taker-feed."""
    eff = _slip(target, buy=not is_long, slippage=slippage)  # exiting LONG is a sell
    raw_r = ((eff - entry_fill) if is_long else (entry_fill - eff)) / risk
    return raw_r * frac, taker_fee * eff / risk * frac, abs(target - eff) / risk * frac


def resolve_signal(
    mb: MachineBlock,
    anchor_ts: datetime,
    frame: pd.DataFrame,
    *,
    taker_fee: float = _TAKER_FEE,
    slippage: float = _SLIPPAGE,
    entry_zone_buffer: float = 0.02,
    max_hold_bars: int = 240,
) -> ShadowOutcome:
    """Walk one captured signal forward. `frame` is the venue's OHLCV (1H bars assumed —
    `expiry_candles` is in 1H units), index = bar-close datetime, cols Open/High/Low/Close."""
    if len(frame) == 0 or frame.index[0] > anchor_ts:
        # The bars right after the signal are not in the fetched window — can't resolve.
        return ShadowOutcome(status=DATA_INCOMPLETE)
    fwd = frame[frame.index > anchor_ts]
    if len(fwd) == 0:
        return ShadowOutcome(status=OPEN)  # too recent — no forward bars yet

    is_long = mb.direction.upper() == "LONG"
    raw_fill_marker, entry_fill, entry_pos, low_fid = _resolve_entry(
        mb, fwd, is_long, slippage, entry_zone_buffer,
    )
    if raw_fill_marker != "FILL":
        return ShadowOutcome(status=raw_fill_marker, low_fidelity=False)

    exits = reanchor_exit_prices(mb, entry_fill)
    stop = exits["stop_loss"]
    tp1 = exits.get("tp1")
    tp2 = exits.get("tp2")
    risk = abs(entry_fill - stop)
    if risk <= 0:  # degenerate geometry (validate_execution_params should prevent) — exclude
        return ShadowOutcome(status=NO_FILL, low_fidelity=low_fid)

    entry_fee_r = taker_fee * entry_fill / risk
    entry_slip_r = abs(entry_fill - entry_fill / (1 + slippage if is_long else 1 - slippage)) / risk
    tp1_frac = (mb.tp1_exit_pct or 0) / 100 if tp1 else 0.0

    mgmt = fwd.iloc[entry_pos + 1:]
    pnl_r = 0.0
    leg_fees_r = 0.0
    leg_slip_r = 0.0
    remaining = 1.0
    tp1_done = False
    exit_reason = None
    exit_price = None

    def _terminal(reason, price, bars):
        return ShadowOutcome(
            status=RESOLVED, entry_fill=entry_fill, exit_price=price, exit_reason=reason,
            realized_r=pnl_r - entry_fee_r - leg_fees_r,
            fees_r=entry_fee_r + leg_fees_r, slippage_r=entry_slip_r + leg_slip_r,
            bars_held=bars, low_fidelity=low_fid,
        )

    for j in range(len(mgmt)):
        if j >= max_hold_bars:  # time-invalidation: close the remainder at this bar's close
            close = float(mgmt.iloc[j]["Close"])
            p, f, s = _exit_leg(entry_fill, close, remaining, is_long, slippage, taker_fee, risk)
            pnl_r += p
            leg_fees_r += f
            leg_slip_r += s
            return _terminal("TIME_EXPIRY" if not tp1_done else "TP1_TIME", close, j + 1)
        hi = float(mgmt.iloc[j]["High"])
        lo = float(mgmt.iloc[j]["Low"])
        # Pessimistic: stop first when a bar spans both stop and a TP.
        if (is_long and lo <= stop) or (not is_long and hi >= stop):
            p, f, s = _exit_leg(entry_fill, stop, remaining, is_long, slippage, taker_fee, risk)
            pnl_r += p
            leg_fees_r += f
            leg_slip_r += s
            return _terminal("TP1_SL" if tp1_done else "SL", stop, j + 1)
        if tp1 and not tp1_done and ((is_long and hi >= tp1) or (not is_long and lo <= tp1)):
            p, f, s = _exit_leg(entry_fill, tp1, tp1_frac, is_long, slippage, taker_fee, risk)
            pnl_r += p
            leg_fees_r += f
            leg_slip_r += s
            remaining -= tp1_frac
            tp1_done = True
            exit_reason, exit_price = "TP1", tp1
        if tp2 and remaining > 0 and ((is_long and hi >= tp2) or (not is_long and lo <= tp2)):
            p, f, s = _exit_leg(entry_fill, tp2, remaining, is_long, slippage, taker_fee, risk)
            pnl_r += p
            leg_fees_r += f
            leg_slip_r += s
            return _terminal("TP1_TP2" if tp1_done else "TP2", tp2, j + 1)

    # Frame ended with size still open → not yet terminal; re-resolve on a later run.
    return ShadowOutcome(
        status=OPEN, entry_fill=entry_fill, exit_reason=exit_reason, exit_price=exit_price,
        bars_held=len(mgmt), low_fidelity=low_fid,
    )
