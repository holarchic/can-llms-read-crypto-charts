"""Tests for scanner.analysis — cost estimation."""

from types import SimpleNamespace

from scanner.analysis import _estimate_cost


class TestEstimateCost:

    def test_no_cache_base_rate(self):
        """No cache attributes → full input charged at base rate."""
        usage = SimpleNamespace(input_tokens=1000, output_tokens=500)
        cost = _estimate_cost(usage, "claude-sonnet-4-5-20250929")
        # input: (1000/1e6) * 3.0 = 0.003
        # output: (500/1e6) * 15.0 = 0.0075
        assert cost == round(0.003 + 0.0075, 4)

    def test_with_cache_no_double_count(self):
        """Cached tokens subtracted from input_tokens before base rate."""
        usage = SimpleNamespace(
            input_tokens=1000, output_tokens=500,
            cache_creation_input_tokens=200, cache_read_input_tokens=600,
        )
        cost = _estimate_cost(usage, "claude-sonnet-4-5-20250929")
        # non_cached: 1000 - 200 - 600 = 200
        # input: (200/1e6) * 3.0 = 0.0006
        # cache_write: (200/1e6) * 3.0 * 1.25 = 0.00075
        # cache_read: (600/1e6) * 3.0 * 0.1 = 0.00018
        # output: (500/1e6) * 15.0 = 0.0075
        expected = 0.0006 + 0.00075 + 0.00018 + 0.0075
        assert cost == round(expected, 4)

    def test_negative_non_cached_floors_to_zero(self):
        """If cache tokens exceed input_tokens, non_cached floors to 0."""
        usage = SimpleNamespace(
            input_tokens=100, output_tokens=0,
            cache_creation_input_tokens=50, cache_read_input_tokens=100,
        )
        cost = _estimate_cost(usage, "claude-sonnet-4-5-20250929")
        # non_cached: max(0, 100 - 50 - 100) = 0
        # cache_write: (50/1e6) * 3.0 * 1.25 = 0.0001875
        # cache_read: (100/1e6) * 3.0 * 0.1 = 0.00003
        expected = 0.0 + 0.0001875 + 0.00003 + 0.0
        assert cost == round(expected, 4)
        assert cost > 0  # still positive from cache costs

    def test_none_cache_attributes(self):
        """getattr returning None handled by `or 0`."""
        usage = SimpleNamespace(
            input_tokens=1000, output_tokens=500,
            cache_creation_input_tokens=None, cache_read_input_tokens=None,
        )
        cost = _estimate_cost(usage, "claude-sonnet-4-5-20250929")
        # None or 0 → 0, so no cache costs, full input at base rate
        expected = (1000 / 1e6) * 3.0 + (500 / 1e6) * 15.0
        assert cost == round(expected, 4)
