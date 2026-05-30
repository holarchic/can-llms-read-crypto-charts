"""Pure forward resolver for shadow signals (T-020.1, SPEC #41).

`resolve_signal` walks ONE captured CONFIRM forward against an OHLCV frame and returns
a `ShadowOutcome` with a terminal status (or `OPEN`) and, when it became a trade, a
realized R **net of taker fees + slippage**. No DB, no network — the frame is supplied
by the caller (`shadow-resolve` fetches it for the signal's venue).

Fidelity (faithful to the live entry/exit path, deliberately pessimistic for a go-live
gate):
- **Entry, `entry_policy=v1`:** IMMEDIATE fills at the signal bar's open; `candle_close_*`
  on the confirming close; `limit_at` when a bar's range spans the limit (flagged
  `low_fidelity` — a 1H frame can't see the true intrabar fill). A fill outside
  `[entry_low, entry_high] ± entry_zone_buffer` is `NO_FILL` (live `price_outside_zone`).
  For conditional entries the live `invalidation_price` is checked **before** the
  trigger each bar (`entry_monitor` ordering): a pre-trigger hit is `INVALIDATED`. No
  trigger within `expiry_candles` bars → `TRIGGER_EXPIRED`.
- **Exit, `exit_policy=v1`:** SL / TP1 partial / TP2 / time-invalidation (`max_hold_bars`).
  **Pessimistic intrabar rule:** a bar whose range spans both the stop and a TP counts
  as SL-first. v1 takes NO BE rescue, so every non-TP loser is a full −1R.
- **Exit, `exit_policy=v2` (T-020.11):** as v1, plus a threshold-free BE rescue — once the
  TP1 leg fills, the remaining size's stop ratchets to `entry_fill` (breakeven, ruleset
  §6.4), so a post-TP1 reversal exits at BE (`TP1_BE`, ≈ −costs) instead of the original
  stop (`TP1_SL`, ≈ −1R). BE applies from the bar AFTER the TP1 fill (intrabar SL-first
  preserved). The post-TP1 setup-TF trail + runner-size floor are the sibling T2 chunks.
- Exits reanchor to the actual fill (`reanchor_exit_prices`) so realized R uses the
  intended stop distance; slippage is adverse on every leg; fees are taker per leg.

Non-trade statuses (`TRIGGER_EXPIRED`/`NO_FILL`/`INVALIDATED`/`DATA_INCOMPLETE`) and
`OPEN` carry `realized_r is None` and are excluded from the expectancy sample.
"""

from datetime import datetime, timedelta

import pandas as pd

from scanner.data_plane_v2 import SETUP_TFS
from scanner.execution.sizing import reanchor_exit_prices
from scanner.indicators import find_swing_points
from scanner.models import MachineBlock, ShadowOutcome

__all__ = [
    "resolve_signal",
    "ENTRY_POLICY_VERSION", "EXIT_POLICY_VERSION", "EXIT_POLICY_VERSIONS", "RETRYABLE",
    "RESOLVED", "OPEN", "TRIGGER_EXPIRED", "NO_FILL", "INVALIDATED", "DATA_INCOMPLETE",
]

ENTRY_POLICY_VERSION = "v1"
EXIT_POLICY_VERSION = "v1"           # default exit policy
EXIT_POLICY_VERSIONS = frozenset({"v1", "v2"})  # the resolvable exit policies (T-020.11)

RESOLVED = "RESOLVED"
OPEN = "OPEN"
TRIGGER_EXPIRED = "TRIGGER_EXPIRED"
NO_FILL = "NO_FILL"
INVALIDATED = "INVALIDATED"
DATA_INCOMPLETE = "DATA_INCOMPLETE"

# Statuses worth re-resolving on a later run: still in-flight (OPEN), or awaiting a fetch
# window that reaches the signal (DATA_INCOMPLETE). The store reloads exactly these so the
# tape stays re-runnable; the rest are terminal non-trades / trades.
RETRYABLE = frozenset({OPEN, DATA_INCOMPLETE})

# T-020.12 (#51): exit_policy=v2 post-TP1 runner trail, keyed to the SETUP TF. The resolver
# walks a 1H frame, so setup-TF swings come from resampling the frame up to `setup_tf` (1d/4h);
# `1h`/`15m`/unknown/None use the 1H frame as-is — the resolution floor, can't resample finer
# than the fetched bars. Params mirror live `drift` (the resolver is config-free by design #41).
_TRAIL_LOOKBACK = 3            # find_swing_points lookback (live trailing_sl_lookback)
_TRAIL_MIN_DIST_PCT = 0.2      # keep the trailed stop ≥ this % from price (no instant trigger)
# Resample rule per setup TF: 1d/4h resample the 1H frame UP; 1h/15m use the frame as-is (the
# 1H resolution floor — can't resample finer than the fetched bars). A setup_tf NOT in this map
# (None / unknown) gets NO trail — the .11 BE floor stands (conservative: never guess a TF).
_TRAIL_RULE = {"1d": "1D", "4h": "4h", "1h": None, "15m": None}
_TRAIL_TF_DELTA = {"1d": timedelta(days=1), "4h": timedelta(hours=4)}
_FRAME_DELTA = timedelta(hours=1)                       # the resolver frame is 1H
# The trail-rule key set IS the canonical setup-TF set (owner: data_plane_v2.SETUP_TFS);
# parity-guarded at import so it can't drift (cf. pipeline `_TF_SECONDS`). A real `setup_tf` is
# always one of these — the store fails loud on any other non-null value, so the "not in" branch
# of `_trail_swings` is a defense-in-depth no-trail for None.
if set(_TRAIL_RULE) != set(SETUP_TFS):
    raise RuntimeError(f"_TRAIL_RULE keys {sorted(_TRAIL_RULE)} != SETUP_TFS {sorted(SETUP_TFS)}")


def _trail_swings(frame, setup_tf):
    """Setup-TF swing points + the TF bar duration, for the post-TP1 trail.

    `1d`/`4h` resample the 1H `frame` up; `1h`/`15m` use it as-is (the resolution floor); a
    `setup_tf` not in `_TRAIL_RULE` (None/unknown) returns ([], …) so the trail never arms (BE
    floor stands). Also ([], …) when there are too few bars for `find_swing_points`."""
    if setup_tf not in _TRAIL_RULE:        # None / unknown TF → no trail
        return [], _FRAME_DELTA
    rule = _TRAIL_RULE[setup_tf]
    if rule is None:                       # 1h / 15m → the 1H frame is the floor
        tf_df, tf_delta = frame, _FRAME_DELTA
    else:                                  # 1d / 4h → resample up
        tf_df = frame.resample(rule).agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        ).dropna()
        # Drop a partial LEADING bucket: if `frame` starts mid-HTF-bar, the first resampled
        # bucket is missing its earlier 1H bars and would mis-form swings vs a true HTF series
        # (left-edge truncation — the confirmation gate only guards the right edge; a trailing
        # partial bucket is harmless since a swing there never confirms within the frame).
        if len(tf_df) and tf_df.index[0] < frame.index[0]:
            tf_df = tf_df.iloc[1:]
        tf_delta = _TRAIL_TF_DELTA[setup_tf]
    if len(tf_df) < 2 * _TRAIL_LOOKBACK + 1:
        return [], tf_delta
    return find_swing_points(tf_df, _TRAIL_LOOKBACK), tf_delta


def _trail_stop(swings, tf_delta, current_ts, bar_open, is_long, effective_stop):
    """The trailed stop at `current_ts` (the bar OPEN), or None if no qualifying swing.

    A swing is eligible only once CONFIRMED — `swing.timestamp` is the (resampled) bar OPEN and
    find_swing_points needs `lookback` bars after it, so it is visible at
    `swing.timestamp + (lookback+1)·tf_delta` (≡ swing_close + lookback·tf_delta). Among confirmed
    swings of the trailing type (low for LONG / high for SHORT) that TIGHTEN the stop and sit
    ≥ `_TRAIL_MIN_DIST_PCT` from the bar's OPEN price (no instant-trigger), return the most recent —
    mirroring live `find_trailing_sl` + its tighten-only guard."""
    want = "low" if is_long else "high"
    confirm_delay = (_TRAIL_LOOKBACK + 1) * tf_delta
    if is_long:
        ceiling = bar_open * (1 - _TRAIL_MIN_DIST_PCT / 100)
        ok = [s for s in swings if s.type == want and s.timestamp + confirm_delay <= current_ts
              and effective_stop < s.price < ceiling]
    else:
        floor = bar_open * (1 + _TRAIL_MIN_DIST_PCT / 100)
        ok = [s for s in swings if s.type == want and s.timestamp + confirm_delay <= current_ts
              and floor < s.price < effective_stop]
    return ok[-1].price if ok else None


def _slip(px: float, *, buy: bool, slippage: float) -> float:
    """Adverse slippage: a buy fills higher, a sell fills lower."""
    return px * (1 + slippage) if buy else px * (1 - slippage)


def _resolve_entry(mb, fwd, is_long, slippage, buffer, max_expiry):
    """Returns ('FILL', fill_price, bar_pos, low_fidelity) or (status, None, None, False)."""
    lo_z = mb.entry_low * (1 - buffer)
    hi_z = mb.entry_high * (1 + buffer)
    trig = mb.entry_trigger

    if trig is None:  # IMMEDIATE — fill at the signal bar's open
        px = float(fwd.iloc[0]["Open"])
        if not (lo_z <= px <= hi_z):
            return (NO_FILL, None, None, False)
        return ("FILL", _slip(px, buy=is_long, slippage=slippage), 0, False)

    inv = mb.invalidation_price
    trigger_px = mb.entry_trigger_price
    # Mirror live entry_monitor's clamp: default 12 when unset, then clamp to [1, max_expiry].
    raw_expiry = mb.expiry_candles if mb.expiry_candles is not None else 12
    expiry = max(1, min(int(raw_expiry), max_expiry))
    for i in range(min(expiry, len(fwd))):
        bar = fwd.iloc[i]
        close = float(bar["Close"])
        if trig == "limit_at":
            # A resting limit: live polls spot, so invalidation trips on an intrabar wick
            # touch and the fill is touch-based (low fidelity on a 1H frame). Invalidation
            # is checked first so a pre-fill kill wins.
            hi = float(bar["High"])
            lo = float(bar["Low"])
            if inv is not None and ((is_long and lo <= inv) or (not is_long and hi >= inv)):
                return (INVALIDATED, None, None, False)
            if lo <= trigger_px <= hi:
                return ("FILL", _slip(trigger_px, buy=is_long, slippage=slippage), i, True)
            continue
        # candle_close_*: live evaluates BOTH invalidation and the trigger on the CLOSE,
        # invalidation-first (a wick through inv that closes back is NOT a kill — mirrors
        # entry_monitor.evaluate_entry_trigger).
        if inv is not None and ((is_long and close <= inv) or (not is_long and close >= inv)):
            return (INVALIDATED, None, None, False)
        if trig == "candle_close_above" and close >= trigger_px:
            return _conditional_fill(close, lo_z, hi_z, is_long, slippage, i)
        if trig == "candle_close_below" and close <= trigger_px:
            return _conditional_fill(close, lo_z, hi_z, is_long, slippage, i)
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
    exit_policy: str = EXIT_POLICY_VERSION,
    setup_tf: str | None = None,  # v2 post-TP1 trail TF (#51); ignored under v1
    # Production sources the per-venue taker fee from EXCHANGE_META[exchange]["fee_rate"]
    # (the resolve CLI passes it); these defaults are a fallback for direct / test calls.
    taker_fee: float = 0.00035,
    slippage: float = 0.0005,
    entry_zone_buffer: float = 0.02,
    max_expiry: int = 24,  # mirrors execution conditional_entry_max_expiry
    max_hold_bars: int = 240,
) -> ShadowOutcome:
    """Walk one captured signal forward. `frame` is the venue's OHLCV (1H bars assumed —
    `expiry_candles` is in 1H units), index = candle-OPEN datetime (ccxt), cols Open/High/Low/Close."""
    if exit_policy not in EXIT_POLICY_VERSIONS:
        raise ValueError(
            f"unknown exit_policy {exit_policy!r} (expected one of {sorted(EXIT_POLICY_VERSIONS)})"
        )
    be_after_tp1 = exit_policy == "v2"  # ratchet the remainder's stop to BE once TP1 fills
    # `fetch_candles` indexes by candle-OPEN time, so the bar CONTAINING the signal is the
    # last one whose open <= anchor_ts. Resolve from that bar (not the next), or an IMMEDIATE
    # fill lands a full bar late and the OPEN/DATA_INCOMPLETE boundary shifts. If no bar opens
    # at/before the signal, its bar is outside the fetched window — can't resolve faithfully.
    prior = frame.index[frame.index <= anchor_ts]
    if len(frame) == 0 or len(prior) == 0:
        return ShadowOutcome(status=DATA_INCOMPLETE)
    fwd = frame[frame.index >= prior[-1]]

    is_long = mb.direction.upper() == "LONG"
    raw_fill_marker, entry_fill, entry_pos, low_fid = _resolve_entry(
        mb, fwd, is_long, slippage, entry_zone_buffer, max_expiry,
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
    # Mirror the live size split (sizing.calculate_position): with both targets the TP1
    # fraction normalizes against the pct sum; with TP1 only, full size exits at TP1.
    if tp1 and tp2:
        _t1 = mb.tp1_exit_pct or 0
        _t2 = mb.tp2_exit_pct if mb.tp2_exit_pct is not None else 50
        tp1_frac = _t1 / (_t1 + _t2) if (_t1 + _t2) > 0 else 0.5
    else:
        tp1_frac = 1.0 if tp1 else 0.0

    # IMMEDIATE enters at the signal bar's OPEN, so that bar's post-open range can hit
    # SL/TP — include it. Conditional/limit enter at/after a bar close, so manage from the
    # next bar (avoids counting the entry bar's pre-fill range).
    mgmt_start = entry_pos if mb.entry_trigger is None else entry_pos + 1
    mgmt = fwd.iloc[mgmt_start:]
    pnl_r = 0.0
    leg_fees_r = 0.0
    leg_slip_r = 0.0
    remaining = 1.0
    tp1_done = False
    exit_reason = None
    exit_price = None
    # v2: once TP1 fills, the remaining size's stop ratchets to BE (entry_fill) — ruleset
    # §6.4. v1 leaves it at the original stop (no rescue, #41), so `effective_stop` never moves.
    effective_stop = stop
    # The label a post-TP1 stop-out carries, derived from the stop's STATE (not the policy
    # flag): "TP1_SL" at the original stop, "TP1_BE" once ratcheted to breakeven. .12's trail
    # sets its own label here when it moves the stop above BE — so the label never lies.
    post_tp1_stop_label = "TP1_SL"
    # v2 trail substrate (T-020.12): setup-TF swings + bar duration, precomputed once on the
    # post-entry frame. Empty for v1 (be_after_tp1 False) — the trail never arms.
    trail_swings, trail_tf_delta = _trail_swings(fwd, setup_tf) if be_after_tp1 else ([], _FRAME_DELTA)

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
        # v2 trail (T-020.12): post-TP1, ratchet the stop UP to confirmed setup-TF swings using
        # ONLY info available at this bar's OPEN (swings confirmed before the open; min-distance
        # vs the open price) — armed BEFORE the stop-first check below tests the bar's range
        # against it. No same-bar look-ahead. Tighten-only; stays ≥ the BE floor.
        if be_after_tp1 and tp1_done:
            trailed = _trail_stop(trail_swings, trail_tf_delta, mgmt.index[j],
                                  float(mgmt.iloc[j]["Open"]), is_long, effective_stop)
            if trailed is not None:
                effective_stop = trailed
                post_tp1_stop_label = "TP1_TRAIL"
        # Pessimistic: stop first when a bar spans both stop and a TP. Post-TP1 under v2,
        # `effective_stop` is BE or a setup-TF trail (set from the bar AFTER the TP1 fill — this
        # check already ran against the pre-TP1 stop on the fill bar, so it is never retroactive).
        if (is_long and lo <= effective_stop) or (not is_long and hi >= effective_stop):
            p, f, s = _exit_leg(entry_fill, effective_stop, remaining, is_long, slippage, taker_fee, risk)
            pnl_r += p
            leg_fees_r += f
            leg_slip_r += s
            reason = post_tp1_stop_label if tp1_done else "SL"
            return _terminal(reason, effective_stop, j + 1)
        if tp1 and not tp1_done and ((is_long and hi >= tp1) or (not is_long and lo <= tp1)):
            p, f, s = _exit_leg(entry_fill, tp1, tp1_frac, is_long, slippage, taker_fee, risk)
            pnl_r += p
            leg_fees_r += f
            leg_slip_r += s
            remaining -= tp1_frac
            tp1_done = True
            exit_reason, exit_price = "TP1", tp1
            if be_after_tp1:  # v2: ratchet the remainder's stop to breakeven
                effective_stop = entry_fill
                post_tp1_stop_label = "TP1_BE"
            if remaining <= 1e-9:  # full size exits at TP1 (single-target / 100-0 split)
                return _terminal("TP1", tp1, j + 1)
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
