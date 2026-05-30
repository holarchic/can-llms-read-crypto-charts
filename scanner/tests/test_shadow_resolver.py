"""T-020.1 forward resolver (SPEC #41) — `shadow.resolver.resolve_signal`.

Synthetic 1H OHLCV frames; bar 0 is the anchor bar (the signal's scan bar — entry
fills from the FIRST forward bar). Covers entry semantics (IMMEDIATE / conditional /
limit_at / invalidation / expiry / zone-reject), exit semantics (SL / TP1→TP2 / TP1→SL
/ pessimistic intrabar / OPEN), and the data-window guard.
"""

import pandas as pd
import pytest

from scanner.models import MachineBlock
from scanner.shadow import resolver as R
from scanner.shadow.resolver import resolve_signal

_BASE = {
    "entry_low": 100.0, "entry_high": 101.0, "stop_loss": 95.0,
    "tp1": 110.0, "tp1_exit_pct": 50, "tp2": 120.0, "tp2_exit_pct": 50,
    "direction": "LONG", "setup_type": "trend_continuation",
    "setup_quality": "textbook", "confidence": 7, "leverage": 5,
}


def _mb(**over) -> MachineBlock:
    return MachineBlock.from_raw({**_BASE, **over})


def _frame(bars):
    """bars: list of (open, high, low, close). Index is 1H bars from a fixed UTC start."""
    idx = pd.date_range("2026-05-29T12:00:00Z", periods=len(bars), freq="1h")
    return pd.DataFrame(bars, index=idx, columns=["Open", "High", "Low", "Close"])


def _anchor(frame, bar=0):
    return frame.index[bar]


def test_long_immediate_runs_to_tp2():
    f = _frame([
        (100, 101, 99, 100.5),    # anchor bar
        (100.5, 101, 100, 100.5),  # entry fill ~100.55
        (105, 111, 104, 108),      # TP1 (high>=110)
        (108, 121, 107, 119),      # TP2 (high>=120)
    ])
    out = resolve_signal(_mb(), _anchor(f), f)
    assert out.status == R.RESOLVED
    assert out.exit_reason == "TP1_TP2"
    assert out.realized_r > 1.5          # big winner, net of costs
    assert out.fees_r > 0 and out.slippage_r > 0


def test_long_immediate_full_stop_is_worse_than_minus_one():
    f = _frame([
        (100, 101, 99, 100.5),
        (100.5, 101, 100, 100.5),  # fill
        (100, 101, 94, 96),        # low 94 <= stop -> SL
    ])
    out = resolve_signal(_mb(), _anchor(f), f)
    assert out.status == R.RESOLVED and out.exit_reason == "SL"
    assert out.realized_r < -1.0          # slippage + fees push it below -1R


def test_pessimistic_intrabar_stop_wins_over_tp():
    f = _frame([
        (100, 101, 99, 100.5),
        (100.5, 101, 100, 100.5),  # fill
        (100, 121, 94, 100),       # spans BOTH tp2 and stop -> SL first
    ])
    out = resolve_signal(_mb(), _anchor(f), f)
    assert out.status == R.RESOLVED and out.exit_reason == "SL"
    assert out.realized_r < 0


def test_tp1_then_stop_on_remainder():
    f = _frame([
        (100, 101, 99, 100.5),
        (100.5, 101, 100, 100.5),  # fill
        (105, 111, 104, 108),      # TP1 (no stop touch)
        (100, 101, 94, 96),        # remainder stopped
    ])
    out = resolve_signal(_mb(), _anchor(f), f)
    assert out.status == R.RESOLVED and out.exit_reason == "TP1_SL"
    assert -0.5 < out.realized_r < 1.5    # half banked at TP1, half lost


def test_conditional_close_above_triggers_then_resolves():
    mb = _mb(entry_trigger="candle_close_above", entry_trigger_price=102.0,
             invalidation_price=98.0, expiry_candles=5)
    f = _frame([
        (100, 101, 99, 100.5),     # anchor
        (100, 101, 99, 101),       # no trigger (close 101<102), no inval (low 99>98)
        (101, 103, 100, 102.5),    # close 102.5>=102 -> trigger+fill (in zone+buffer)
        (103, 123, 102, 121),      # TP1+TP2
    ])
    out = resolve_signal(mb, _anchor(f), f)
    assert out.status == R.RESOLVED
    assert out.entry_fill is not None


def test_conditional_invalidation_wins_before_trigger():
    mb = _mb(entry_trigger="candle_close_above", entry_trigger_price=102.0,
             invalidation_price=98.0, expiry_candles=5)
    f = _frame([
        (100, 101, 99, 100.5),     # signal bar (no trigger, no inval close)
        (100, 101, 96, 97),        # CLOSE 97 <= inval 98 -> INVALIDATED (close-based, not wick)
    ])
    out = resolve_signal(mb, _anchor(f), f)
    assert out.status == R.INVALIDATED
    assert out.realized_r is None


def test_conditional_expiry_clamped_when_zero():
    # expiry_candles=0 must clamp to 1 (mirrors live entry_monitor), not span the window.
    mb = _mb(entry_trigger="candle_close_above", entry_trigger_price=102.0,
             invalidation_price=98.0, expiry_candles=0)
    f = _frame([
        (100, 101, 99, 100.5),     # signal bar (bar 0) — no trigger (close 100.5 < 102)
        (101, 103, 100, 102.5),    # would trigger, but expiry clamped to 1 already passed bar 0
    ])
    out = resolve_signal(mb, _anchor(f), f)
    assert out.status == R.TRIGGER_EXPIRED


def test_conditional_trigger_expires():
    mb = _mb(entry_trigger="candle_close_above", entry_trigger_price=102.0,
             invalidation_price=98.0, expiry_candles=2)
    f = _frame([
        (100, 101, 99, 100.5),
        (100, 101, 99, 100),       # no trigger, no inval
        (100, 101, 99, 100),       # expiry reached
    ])
    out = resolve_signal(mb, _anchor(f), f)
    assert out.status == R.TRIGGER_EXPIRED


def test_immediate_outside_zone_is_no_fill():
    f = _frame([
        (110, 112, 109, 111),      # signal bar: open 110 outside [98, 103.02] -> NO_FILL
        (110, 112, 109, 111),
    ])
    out = resolve_signal(_mb(), _anchor(f), f)
    assert out.status == R.NO_FILL
    assert out.realized_r is None


def test_data_incomplete_when_frame_starts_after_anchor():
    f = _frame([(100, 101, 99, 100.5), (100, 101, 99, 100.5)])
    anchor = f.index[0] - pd.Timedelta(hours=1)   # earliest bar is AFTER the signal
    out = resolve_signal(_mb(), anchor, f)
    assert out.status == R.DATA_INCOMPLETE


def test_open_when_no_terminal_event_by_frame_end():
    f = _frame([
        (100, 101, 99, 100.5),
        (100.5, 101, 100, 100.5),  # fill
        (100, 105, 98, 102),       # meanders, no stop/tp
        (102, 106, 99, 103),
    ])
    out = resolve_signal(_mb(), _anchor(f), f)
    assert out.status == R.OPEN
    assert out.entry_fill is not None and out.realized_r is None


def test_single_target_resolves_fully_at_tp1():
    # tp1-only (no tp2) → full size exits at TP1; must terminate at TP1, not TP1_TIME/OPEN.
    mb = MachineBlock.from_raw({
        "entry_low": 100.0, "entry_high": 101.0, "stop_loss": 95.0,
        "tp1": 110.0, "tp1_exit_pct": 100, "direction": "LONG",
        "setup_type": "trend_continuation", "setup_quality": "textbook",
        "confidence": 7, "leverage": 5,
    })
    f = _frame([
        (100, 101, 99, 100.5),     # signal bar
        (100.5, 101, 100, 100.5),
        (105, 111, 104, 108),      # TP1 (full size) — terminal here
        (108, 109, 94, 95),        # must NOT be reached (would otherwise stop the runner)
    ])
    out = resolve_signal(mb, _anchor(f), f)
    assert out.status == R.RESOLVED and out.exit_reason == "TP1"
    assert out.realized_r > 0


def test_short_immediate_runs_to_tp2():
    mb = _mb(direction="SHORT", stop_loss=105.0, tp1=90.0, tp2=80.0)
    f = _frame([
        (100, 101, 99, 100.5),
        (100.5, 101, 100, 100.5),  # fill (sell)
        (100, 101, 88, 89),        # TP1 (low<=90)
        (89, 90, 79, 80),          # TP2 (low<=80)
    ])
    out = resolve_signal(mb, _anchor(f), f)
    assert out.status == R.RESOLVED and out.exit_reason == "TP1_TP2"
    assert out.realized_r > 1.5


def test_limit_at_is_flagged_low_fidelity():
    mb = _mb(entry_trigger="limit_at", entry_trigger_price=100.0,
             invalidation_price=97.0, expiry_candles=5)
    f = _frame([
        (100, 101, 99, 100.5),
        (102, 103, 99.5, 102),     # range spans the 100 limit -> fill, low fidelity
        (103, 121, 102, 120),
    ])
    out = resolve_signal(mb, _anchor(f), f)
    assert out.low_fidelity is True
    assert out.entry_fill is not None


# --- exit_policy=v2: BE-at-TP1 rescue (T-020.11) ---------------------------------

def test_v2_be_at_tp1_beats_v1_on_remainder_stop():
    # TP1 fills, then the remainder reverses to the original stop. v1 loses the
    # remainder at the stop (TP1_SL); v2 ratchets it to BE so it exits at entry
    # (TP1_BE) — strictly higher R on the same frame.
    f = _frame([
        (100, 101, 99, 100.5),
        (100.5, 101, 100, 100.5),  # fill
        (105, 111, 104, 108),      # TP1 (high>=110), no stop touch
        (100, 101, 94, 96),        # remainder: v1 stops at 95; v2 stops at BE (entry)
    ])
    v1 = resolve_signal(_mb(), _anchor(f), f, exit_policy="v1")
    v2 = resolve_signal(_mb(), _anchor(f), f, exit_policy="v2")
    assert v1.exit_reason == "TP1_SL" and v2.exit_reason == "TP1_BE"
    assert v2.realized_r > v1.realized_r          # the BE move changed the outcome
    assert v2.exit_price > v1.exit_price          # BE (entry) above the original stop


def test_v2_be_at_tp1_short_remainder_stop():
    # SHORT mirror: stop comparator is `hi >= effective_stop`, BE geometry is
    # entry_fill < original stop (BE tightens DOWN). TP1 fills (low<=tp1), then price
    # reverses UP: v1 stops at the higher original stop, v2 at the lower BE.
    mb = _mb(direction="SHORT", stop_loss=105.0, tp1=90.0, tp2=80.0)
    f = _frame([
        (100, 101, 99, 100.5),
        (100, 101, 100, 100.5),    # nothing (hi 101<105, lo 100>90)
        (95, 96, 88, 89),          # TP1 (low<=90), no stop (hi 96<105)
        (98, 106, 97, 100),        # reverse up: v1 stops at 105; v2 stops at BE (entry)
    ])
    v1 = resolve_signal(mb, _anchor(f), f, exit_policy="v1")
    v2 = resolve_signal(mb, _anchor(f), f, exit_policy="v2")
    assert v1.exit_reason == "TP1_SL" and v2.exit_reason == "TP1_BE"
    assert v2.realized_r > v1.realized_r
    assert v2.exit_price < v1.exit_price          # SHORT BE (entry) below the original stop


def test_v2_full_winner_unchanged():
    # TP1 then TP2: BE is armed at the TP1 fill but never hit (price runs to TP2),
    # so v2 == v1 — the rescue only bites when the remainder reverses.
    f = _frame([
        (100, 101, 99, 100.5),
        (100.5, 101, 100, 100.5),
        (105, 111, 104, 108),      # TP1
        (108, 121, 107, 119),      # TP2 (no pullback to BE in between)
    ])
    v1 = resolve_signal(_mb(), _anchor(f), f, exit_policy="v1")
    v2 = resolve_signal(_mb(), _anchor(f), f, exit_policy="v2")
    assert v1.exit_reason == "TP1_TP2" and v2.exit_reason == "TP1_TP2"
    assert v2.realized_r == pytest.approx(v1.realized_r)


def test_v2_no_rescue_before_tp1_matches_v1():
    # Reversal to stop BEFORE TP1: v2 has no pre-TP1 ratchet (BE-at-TP1 only), so it
    # rides to the original stop identically to v1.
    f = _frame([
        (100, 101, 99, 100.5),
        (100.5, 101, 100, 100.5),  # fill
        (100, 101, 94, 96),        # low<=stop before any TP1 touch
    ])
    v1 = resolve_signal(_mb(), _anchor(f), f, exit_policy="v1")
    v2 = resolve_signal(_mb(), _anchor(f), f, exit_policy="v2")
    assert v1.exit_reason == "SL" and v2.exit_reason == "SL"
    assert v2.realized_r == pytest.approx(v1.realized_r)


def test_unknown_exit_policy_raises():
    f = _frame([(100, 101, 99, 100.5), (100.5, 101, 100, 100.5)])
    with pytest.raises(ValueError, match="unknown exit_policy"):
        resolve_signal(_mb(), _anchor(f), f, exit_policy="v3")
