"""Tests for price sanity checks."""

import pytest

from scanner.execution.config import ExecutionConfig
from scanner.execution.sanity import check_price_sanity, is_price_in_entry_zone
from scanner.models import MachineBlock


@pytest.fixture
def config():
    return ExecutionConfig()


# T-016 C3: check_price_sanity now takes a typed MachineBlock. Valid-block
# fixtures route through from_raw (production path; runs the validate gate).
# Tests that originally used validate-invalid geometry to trip a sanity guard
# are recrafted to valid geometry that still trips it, since in production
# sanity only ever runs on validated blocks. None override = field omitted.
_LONG = {
    "direction": "LONG", "entry_low": 61800, "entry_high": 62200,
    "stop_loss": 59800, "tp1": 65000, "tp2": 68000, "leverage": 5,
    "tp1_exit_pct": 50, "confidence": 6,
}
_SHORT = {
    "direction": "SHORT", "entry_low": 61800, "entry_high": 62200,
    "stop_loss": 64200, "tp1": 59000, "tp2": 56000, "leverage": 5,
    "tp1_exit_pct": 50, "confidence": 6,
}


def _long_block(**overrides):
    base = {k: v for k, v in {**_LONG, **overrides}.items() if v is not None}
    return MachineBlock.from_raw(base)


def _short_block(**overrides):
    base = {k: v for k, v in {**_SHORT, **overrides}.items() if v is not None}
    return MachineBlock.from_raw(base)


def _mk(**over):
    """Direct construction (bypasses from_raw's validate gate) for sanity
    guards reachable only with otherwise-invalid params (e.g. absent TP)."""
    f = dict(entry_low=61800.0, entry_high=62200.0, stop_loss=59800.0,
             tp1=65000.0, tp1_exit_pct=50.0, direction="LONG",
             confidence=6.0, leverage=5.0, tp2=68000.0, tp2_exit_pct=50.0)
    f.update(over)
    return MachineBlock(**f)


class TestIsPriceInEntryZone:
    def test_inside(self):
        assert is_price_in_entry_zone(62000, 61800, 62200) is True

    def test_at_boundary_low(self):
        assert is_price_in_entry_zone(61800, 61800, 62200) is True

    def test_at_boundary_high(self):
        assert is_price_in_entry_zone(62200, 61800, 62200) is True

    def test_outside_low(self):
        assert is_price_in_entry_zone(61000, 61800, 62200) is False

    def test_outside_high(self):
        assert is_price_in_entry_zone(63000, 61800, 62200) is False

    # Direction-aware buffer tests
    def test_short_below_zone_within_buffer(self):
        # SHORT: price dropped below entry_low — thesis confirming, should accept
        # Zone [61800, 62200], mid=62000, 2% buffer = 1240 → floor = 60560
        assert is_price_in_entry_zone(61000, 61800, 62200, direction="SHORT", buffer_pct=2.0) is True

    def test_short_below_zone_beyond_buffer(self):
        # Price dropped too far below (> 2% of midpoint below entry_low)
        assert is_price_in_entry_zone(60000, 61800, 62200, direction="SHORT", buffer_pct=2.0) is False

    def test_short_above_zone_within_buffer(self):
        # SHORT: price above entry_high — favorable entry, accept within buffer
        # Zone [61800, 62200], mid=62000, 2% buffer = 1240 → ceiling = 63440
        assert is_price_in_entry_zone(63000, 61800, 62200, direction="SHORT", buffer_pct=2.0) is True

    def test_short_above_zone_beyond_buffer(self):
        # SHORT: price too far above entry_high — reject
        assert is_price_in_entry_zone(64000, 61800, 62200, direction="SHORT", buffer_pct=2.0) is False

    def test_long_above_zone_within_buffer(self):
        # LONG: price pushed above entry_high — thesis confirming, should accept
        # Zone [61800, 62200], mid=62000, 2% buffer = 1240 → ceiling = 63440
        assert is_price_in_entry_zone(63000, 61800, 62200, direction="LONG", buffer_pct=2.0) is True

    def test_long_above_zone_beyond_buffer(self):
        # Price pushed too far above
        assert is_price_in_entry_zone(64000, 61800, 62200, direction="LONG", buffer_pct=2.0) is False

    def test_long_below_zone_within_buffer(self):
        # LONG: price below entry_low — favorable entry, accept within buffer
        # Zone [61800, 62200], mid=62000, 2% buffer = 1240 → floor = 60560
        assert is_price_in_entry_zone(61000, 61800, 62200, direction="LONG", buffer_pct=2.0) is True

    def test_long_below_zone_beyond_buffer(self):
        # LONG: price too far below entry_low — reject
        assert is_price_in_entry_zone(60000, 61800, 62200, direction="LONG", buffer_pct=2.0) is False

    def test_no_buffer_backwards_compatible(self):
        # Without buffer args, behavior is unchanged
        assert is_price_in_entry_zone(61000, 61800, 62200) is False
        assert is_price_in_entry_zone(63000, 61800, 62200) is False

    def test_zero_buffer_no_extension(self):
        assert is_price_in_entry_zone(61000, 61800, 62200, direction="SHORT", buffer_pct=0.0) is False


class TestCheckPriceSanity:
    def test_valid_long(self, config):
        result = check_price_sanity(_long_block(), 62000, 500, config)
        assert result.passed is True
        assert len(result.errors) == 0

    def test_valid_short(self, config):
        result = check_price_sanity(_short_block(), 62000, 500, config)
        assert result.passed is True
        assert len(result.errors) == 0

    def test_entry_too_far(self, config):
        # Price is 10% away from entry zone
        result = check_price_sanity(_long_block(), 56000, 500, config)
        assert result.passed is False
        assert any("Entry zone midpoint" in e for e in result.errors)

    def test_stop_too_tight(self, config):
        # stop 61780 (< entry_low 61800 → valid LONG), entry_mid 62000 →
        # stop_distance 220, /ATR 500 = 0.44x < min 0.5
        block = _long_block(stop_loss=61780)
        result = check_price_sanity(block, 62000, 500, config)
        assert result.passed is False
        assert any("min" in e.lower() for e in result.errors)

    def test_stop_too_wide(self, config):
        # Stop 5000 away, with ATR 500 = 10x (max 5.0)
        block = _long_block(stop_loss=57000)
        result = check_price_sanity(block, 62000, 500, config)
        assert result.passed is False
        assert any("max" in e.lower() for e in result.errors)

    def test_rr_below_min(self, config):
        # tp1 62500 (> entry_high 62200 → valid LONG), reward 500 /
        # stop_distance 2200 = 0.23 R:R < min 0.5
        block = _long_block(tp1=62500, tp2=None)
        result = check_price_sanity(block, 62000, 500, config)
        assert result.passed is False
        assert any("R:R ratio" in e and "below" in e for e in result.errors)

    def test_rr_above_max(self, config):
        # TP way above entry: ~30 R:R
        block = _long_block(tp1=128000, tp2=None)
        result = check_price_sanity(block, 62000, 500, config)
        assert result.passed is False
        assert any("R:R ratio" in e and "above" in e for e in result.errors)

    def test_missing_fields_rejected_at_boundary(self):
        # Pre-C3 check_price_sanity({}) flagged "Missing required price
        # fields". Post T-016 that rejection moves to the typed boundary —
        # an incomplete block cannot become a MachineBlock, so sanity never
        # sees one in production. Assert the boundary rejects it.
        assert MachineBlock.from_raw({}) is None

    def test_no_tp_skips_rr_check(self, config):
        # No usable TP → R:R check skipped. A valid MachineBlock always has
        # tp1 > 0 (required + validated), so this defensive skip is only
        # reachable via direct construction.
        block = _mk(tp1=0.0, tp2=None)
        result = check_price_sanity(block, 62000, 500, config)
        assert all("R:R" not in e for e in result.errors)
