"""Tests for position sizing."""

import pytest

from scanner.execution.sizing import calculate_position, reanchor_exit_prices
from scanner.models import MachineBlock

# T-016 C3: sizing now takes a typed MachineBlock. Valid-block fixtures wrap
# their dict through the canonical from_raw (exactly as production does); an
# override value of None means "field omitted" (the prior dict-era intent).
_LONG = {
    "direction": "LONG", "entry_low": 61800, "entry_high": 62200,
    "stop_loss": 59800, "tp1": 65000, "tp2": 68000, "leverage": 5,
    "tp1_exit_pct": 50, "tp2_exit_pct": 50, "confidence": 6,
}
_SHORT = {
    "direction": "SHORT", "entry_low": 61800, "entry_high": 62200,
    "stop_loss": 64200, "tp1": 59000, "tp2": 56000, "leverage": 5,
    "tp1_exit_pct": 50, "tp2_exit_pct": 50, "confidence": 6,
}


def _long_block(**overrides):
    """Valid LONG MachineBlock (None override = field omitted)."""
    base = {k: v for k, v in {**_LONG, **overrides}.items() if v is not None}
    return MachineBlock.from_raw(base)


def _short_block(**overrides):
    """Valid SHORT MachineBlock (None override = field omitted)."""
    base = {k: v for k, v in {**_SHORT, **overrides}.items() if v is not None}
    return MachineBlock.from_raw(base)


def _mk(**over):
    """Direct MachineBlock construction (bypasses from_raw's validate gate) —
    for exercising defensive sizing guards with geometry the validated
    boundary would reject (e.g. zero stop distance, absent TP). Those guards
    stay in real-money sizing code, so they keep a test even though the typed
    boundary makes them unreachable via the normal path.
    """
    f = dict(entry_low=61800.0, entry_high=62200.0, stop_loss=59800.0,
             tp1=65000.0, tp1_exit_pct=50.0, direction="LONG",
             confidence=6.0, leverage=5.0, tp2=68000.0, tp2_exit_pct=50.0)
    f.update(over)
    return MachineBlock(**f)


class TestCalculatePosition:
    def test_long_basic(self):
        result = calculate_position(_long_block(), balance=10000, risk_pct=1.0)
        assert result is not None
        # entry_mid = 62000, stop = 59800, distance = 2200
        # risk = 10000 * 0.01 = 100, qty = 100/2200 = 0.04545...
        assert abs(result.quantity - 100 / 2200) < 0.0001
        assert abs(result.risk_amount - 100) < 0.01
        assert result.stop_distance == 2200
        assert result.leverage == 5

    def test_short_basic(self):
        result = calculate_position(_short_block(), balance=10000, risk_pct=1.0)
        assert result is not None
        # entry_mid = 62000, stop = 64200, distance = 2200
        assert abs(result.stop_distance - 2200) < 0.01
        # R:R: reward = 62000 - 59000 = 3000, rr = 3000/2200 ≈ 1.36
        assert abs(result.rr_ratio - 3000 / 2200) < 0.01

    def test_leverage_capped(self):
        result = calculate_position(_long_block(leverage=20), balance=10000, risk_pct=1.0, max_leverage=10)
        assert result is not None
        assert result.leverage == 10

    def test_leverage_not_capped_when_below(self):
        result = calculate_position(_long_block(leverage=3), balance=10000, risk_pct=1.0, max_leverage=10)
        assert result is not None
        assert result.leverage == 3

    def test_tp_split(self):
        result = calculate_position(
            _long_block(tp1_exit_pct=60, tp2_exit_pct=40),
            balance=10000, risk_pct=1.0,
        )
        assert result is not None
        assert abs(result.tp1_quantity - result.quantity * 0.6) < 0.0001
        assert abs(result.tp2_quantity - result.quantity * 0.4) < 0.0001

    def test_tp1_only(self):
        result = calculate_position(_long_block(tp2=None), balance=10000, risk_pct=1.0)
        assert result is not None
        assert result.tp1_quantity == result.quantity
        assert result.tp2_quantity == 0

    def test_zero_risk_returns_none(self):
        # entry_mid == stop_loss → zero stop distance. from_raw/validate would
        # reject this (LONG stop must be below entry_low), so build directly
        # to exercise calculate_position's stop_distance==0 guard.
        block = _mk(stop_loss=62000.0)  # entry_mid 62000 == stop
        result = calculate_position(block, balance=10000, risk_pct=1.0)
        assert result is None

    def test_missing_fields_rejected_at_boundary(self):
        # Pre-C3 calculate_position({}) returned None on missing fields. Post
        # T-016 the rejection moves to the typed boundary: an incomplete block
        # cannot become a MachineBlock, so calculate_position never sees one in
        # production. Assert the boundary rejects it.
        assert MachineBlock.from_raw({}) is None

    def test_margin_calculation(self):
        result = calculate_position(_long_block(), balance=10000, risk_pct=1.0)
        assert result is not None
        expected_notional = result.quantity * 62000  # entry_mid
        expected_margin = expected_notional / 5  # leverage=5
        assert abs(result.notional_value - expected_notional) < 0.01
        assert abs(result.margin_required - expected_margin) < 0.01

    def test_uses_entry_midpoint(self):
        block = _long_block(entry_low=61000, entry_high=63000)
        result = calculate_position(block, balance=10000, risk_pct=1.0)
        assert result is not None
        # entry_mid = 62000
        assert abs(result.notional_value - result.quantity * 62000) < 0.01

    def test_higher_risk_pct(self):
        r1 = calculate_position(_long_block(), balance=10000, risk_pct=1.0)
        r2 = calculate_position(_long_block(), balance=10000, risk_pct=2.0)
        assert r2.quantity == pytest.approx(r1.quantity * 2, rel=1e-6)

    def test_rr_ratio_long(self):
        result = calculate_position(_long_block(), balance=10000, risk_pct=1.0)
        # entry_mid=62000, stop=59800, tp1=65000
        # reward = 65000-62000 = 3000, risk = 2200
        assert abs(result.rr_ratio - 3000 / 2200) < 0.01

    def test_no_tp_gives_zero_rr(self):
        # No usable TP → rr_ratio 0. A valid MachineBlock always has tp1>0
        # (required + validated), so this defensive branch is only reachable
        # via direct construction.
        result = calculate_position(_mk(tp1=0.0, tp2=None), balance=10000, risk_pct=1.0)
        assert result is not None
        assert result.rr_ratio == 0

    def test_tp_split_normalizes_when_sum_under_100(self):
        """60+30=90 should normalize to 2/3 and 1/3."""
        result = calculate_position(
            _long_block(tp1_exit_pct=60, tp2_exit_pct=30),
            balance=10000, risk_pct=1.0,
        )
        assert result is not None
        assert abs(result.tp1_quantity - result.quantity * 60 / 90) < 1e-10
        assert abs(result.tp2_quantity - result.quantity * 30 / 90) < 1e-10

    def test_tp_split_normalizes_when_sum_over_100(self):
        """60+50=110 should normalize to 6/11 and 5/11. validate rejects
        sum>100, so this is calculate_position's defensive normalization —
        unit-tested via direct construction (bypassing the validate gate).
        """
        result = calculate_position(
            _mk(tp1_exit_pct=60.0, tp2_exit_pct=50.0),
            balance=10000, risk_pct=1.0,
        )
        assert result is not None
        assert abs(result.tp1_quantity - result.quantity * 60 / 110) < 1e-10
        assert abs(result.tp2_quantity - result.quantity * 50 / 110) < 1e-10

    def test_tp_split_quantities_sum_to_total(self):
        """Normalized tp1+tp2 must equal total quantity exactly across all
        ratios (incl. sum>100 which validate rejects) — calculate_position
        split arithmetic, unit-tested via direct construction.
        """
        for tp1_pct, tp2_pct in [(60, 30), (60, 50), (70, 70), (33, 33)]:
            result = calculate_position(
                _mk(tp1_exit_pct=float(tp1_pct), tp2_exit_pct=float(tp2_pct)),
                balance=10000, risk_pct=1.0,
            )
            assert result is not None
            assert result.tp1_quantity + result.tp2_quantity == result.quantity


class TestReanchorExitPrices:
    def test_long_positive_delta(self):
        """LONG filled above entry_mid → all levels shift up."""
        mb = _long_block()  # entry_mid=62000, sl=59800, tp1=65000, tp2=68000
        result = reanchor_exit_prices(mb, fill_price=62300)
        assert result["delta"] == 300
        assert result["stop_loss"] == 60100  # 59800 + 300
        assert result["tp1"] == 65300        # 65000 + 300
        assert result["tp2"] == 68300        # 68000 + 300

    def test_long_negative_delta(self):
        """LONG filled below entry_mid → all levels shift down."""
        mb = _long_block()
        result = reanchor_exit_prices(mb, fill_price=61700)
        assert result["delta"] == -300
        assert result["stop_loss"] == 59500
        assert result["tp1"] == 64700
        assert result["tp2"] == 67700

    def test_short_positive_delta(self):
        """SHORT filled above entry_mid → levels shift up (wider SL, closer TP)."""
        mb = _short_block()  # entry_mid=62000, sl=64200, tp1=59000, tp2=56000
        result = reanchor_exit_prices(mb, fill_price=62500)
        assert result["delta"] == 500
        assert result["stop_loss"] == 64700
        assert result["tp1"] == 59500
        assert result["tp2"] == 56500

    def test_short_negative_delta(self):
        """SHORT filled below entry_mid → levels shift down."""
        mb = _short_block()
        result = reanchor_exit_prices(mb, fill_price=61500)
        assert result["delta"] == -500
        assert result["stop_loss"] == 63700
        assert result["tp1"] == 58500
        assert result["tp2"] == 55500

    def test_zero_delta(self):
        """Fill at entry_mid → no shift."""
        mb = _long_block()
        result = reanchor_exit_prices(mb, fill_price=62000)
        assert result["delta"] == 0
        assert result["stop_loss"] == 59800
        assert result["tp1"] == 65000
        assert result["tp2"] == 68000

    def test_missing_tp2(self):
        """Only tp1 present → tp2 not in result."""
        mb = _long_block(tp2=None)
        result = reanchor_exit_prices(mb, fill_price=62300)
        assert "tp1" in result
        assert "tp2" not in result
        assert result["tp1"] == 65300

    # Removed pre-C3 test_missing_both_tps: a MachineBlock always has tp1
    # (required, validated > 0), so "no TPs at all" is structurally
    # impossible under the typed contract — reanchor's `if mb.tp1 is not
    # None` is now provably-always-true defensive code. The tp2-absent
    # branch stays covered by test_missing_tp2 above.
