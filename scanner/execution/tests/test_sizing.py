"""Tests for position sizing."""

import pytest
from scanner.execution.sizing import calculate_position, PositionSize


def _long_block(**overrides):
    """Standard LONG machine_block for BTC ~62000."""
    base = {
        "direction": "LONG",
        "entry_low": 61800,
        "entry_high": 62200,
        "stop_loss": 59800,
        "tp1": 65000,
        "tp2": 68000,
        "leverage": 5,
        "tp1_exit_pct": 50,
        "tp2_exit_pct": 50,
    }
    base.update(overrides)
    return base


def _short_block(**overrides):
    """Standard SHORT machine_block for BTC ~62000."""
    base = {
        "direction": "SHORT",
        "entry_low": 61800,
        "entry_high": 62200,
        "stop_loss": 64200,
        "tp1": 59000,
        "tp2": 56000,
        "leverage": 5,
        "tp1_exit_pct": 50,
        "tp2_exit_pct": 50,
    }
    base.update(overrides)
    return base


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
        block = _long_block(stop_loss=62000)  # entry_mid = stop = no risk
        result = calculate_position(block, balance=10000, risk_pct=1.0)
        assert result is None

    def test_missing_fields_returns_none(self):
        result = calculate_position({}, balance=10000, risk_pct=1.0)
        assert result is None

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
        result = calculate_position(_long_block(tp1=None, tp2=None), balance=10000, risk_pct=1.0)
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
        """60+50=110 should normalize to 6/11 and 5/11."""
        result = calculate_position(
            _long_block(tp1_exit_pct=60, tp2_exit_pct=50),
            balance=10000, risk_pct=1.0,
        )
        assert result is not None
        assert abs(result.tp1_quantity - result.quantity * 60 / 110) < 1e-10
        assert abs(result.tp2_quantity - result.quantity * 50 / 110) < 1e-10

    def test_tp_split_quantities_sum_to_total(self):
        """Normalized tp1+tp2 must equal total quantity exactly."""
        for tp1_pct, tp2_pct in [(60, 30), (60, 50), (70, 70), (33, 33)]:
            result = calculate_position(
                _long_block(tp1_exit_pct=tp1_pct, tp2_exit_pct=tp2_pct),
                balance=10000, risk_pct=1.0,
            )
            assert result is not None
            assert result.tp1_quantity + result.tp2_quantity == result.quantity
