"""Tests for grid search sweep."""

import pytest

from scanner.strategy.sweep import _parse_range


class TestParseRange:
    def test_basic_range(self):
        values = _parse_range("1.0:0.5:3.0")
        assert values == [1.0, 1.5, 2.0, 2.5, 3.0]

    def test_single_value(self):
        assert _parse_range("2.5") == [2.5]

    def test_integer_range(self):
        values = _parse_range("1:1:3")
        assert values == [1.0, 2.0, 3.0]

    def test_fine_step(self):
        values = _parse_range("0.01:0.005:0.025")
        assert len(values) == 4
        assert values[0] == 0.01
        assert values[-1] == 0.025

    def test_single_step(self):
        values = _parse_range("1.0:1.0:1.0")
        assert values == [1.0]

    def test_large_step_gives_single(self):
        values = _parse_range("1.0:10.0:3.0")
        assert values == [1.0]
