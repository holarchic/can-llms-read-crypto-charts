"""Tests for price sanity checks."""

import pytest
from scanner.execution.sanity import check_price_sanity, is_price_in_entry_zone, SanityResult
from scanner.execution.config import ExecutionConfig


@pytest.fixture
def config():
    return ExecutionConfig()


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
    }
    base.update(overrides)
    return base


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

    def test_short_above_zone_no_buffer(self):
        # SHORT: price above zone — wrong direction, buffer doesn't help
        assert is_price_in_entry_zone(63000, 61800, 62200, direction="SHORT", buffer_pct=2.0) is False

    def test_long_above_zone_within_buffer(self):
        # LONG: price pushed above entry_high — thesis confirming, should accept
        # Zone [61800, 62200], mid=62000, 2% buffer = 1240 → ceiling = 63440
        assert is_price_in_entry_zone(63000, 61800, 62200, direction="LONG", buffer_pct=2.0) is True

    def test_long_above_zone_beyond_buffer(self):
        # Price pushed too far above
        assert is_price_in_entry_zone(64000, 61800, 62200, direction="LONG", buffer_pct=2.0) is False

    def test_long_below_zone_no_buffer(self):
        # LONG: price below zone — wrong direction, buffer doesn't help
        assert is_price_in_entry_zone(61000, 61800, 62200, direction="LONG", buffer_pct=2.0) is False

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
        # Stop only 100 away, with ATR 500 = 0.2x (min 0.5)
        block = _long_block(stop_loss=61900)
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
        # TP barely above entry: ~0.1 R:R
        block = _long_block(tp1=62100, tp2=None)
        result = check_price_sanity(block, 62000, 500, config)
        assert result.passed is False
        assert any("R:R ratio" in e and "below" in e for e in result.errors)

    def test_rr_above_max(self, config):
        # TP way above entry: ~30 R:R
        block = _long_block(tp1=128000, tp2=None)
        result = check_price_sanity(block, 62000, 500, config)
        assert result.passed is False
        assert any("R:R ratio" in e and "above" in e for e in result.errors)

    def test_price_magnitude_error(self, config):
        # TP is 50% away from current price (max 20%)
        block = _long_block(tp1=93000, tp2=None)
        result = check_price_sanity(block, 62000, 500, config)
        assert result.passed is False
        assert any("tp1" in e for e in result.errors)

    def test_missing_fields(self, config):
        result = check_price_sanity({}, 62000, 500, config)
        assert result.passed is False
        assert any("Missing" in e for e in result.errors)

    def test_no_tp_skips_rr_check(self, config):
        block = _long_block(tp1=None, tp2=None)
        result = check_price_sanity(block, 62000, 500, config)
        # Should still pass other checks
        assert all("R:R" not in e for e in result.errors)
