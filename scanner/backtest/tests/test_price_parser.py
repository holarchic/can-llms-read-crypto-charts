"""Tests for LLM price text parsing."""

import pytest

from scanner.backtest.resolver import parse_price


class TestParsePrice:
    """Price extraction from LLM text output."""

    def test_simple_dollar(self):
        assert parse_price("$1234.56") == pytest.approx(1234.56)

    def test_dollar_with_commas(self):
        assert parse_price("$1,234.56") == pytest.approx(1234.56)

    def test_large_number(self):
        assert parse_price("$98,765.43") == pytest.approx(98765.43)

    def test_no_dollar_sign(self):
        assert parse_price("1234.56") == pytest.approx(1234.56)

    def test_integer(self):
        assert parse_price("$1234") == pytest.approx(1234.0)

    def test_range_dash(self):
        assert parse_price("$100-$110") == pytest.approx(105.0)

    def test_range_en_dash(self):
        assert parse_price("$100–$110") == pytest.approx(105.0)

    def test_range_spaced(self):
        assert parse_price("$100 - $110") == pytest.approx(105.0)

    def test_range_to(self):
        assert parse_price("$100 to $110") == pytest.approx(105.0)

    def test_na_returns_none(self):
        assert parse_price("N/A") is None

    def test_none_input(self):
        assert parse_price(None) is None

    def test_empty_string(self):
        assert parse_price("") is None

    def test_dash_returns_none(self):
        assert parse_price("-") is None

    def test_text_with_embedded_price(self):
        assert parse_price("around $85,000.50 area") == pytest.approx(85000.50)

    def test_none_string(self):
        assert parse_price("None") is None
