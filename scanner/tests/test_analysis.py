"""Tests for scanner.analysis — cost estimation + SDK retry config."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


class TestAnthropicRetryConfig:
    """SPEC #37: SDK max_retries=5 captures short 529 blips via ~15s of
    exponential backoff before LLMOverloaded fires."""

    def _bs(self):
        from scanner.models import (
            BoundarySignal,
            Direction,
            MarketContext,
            SetupSignal,
            SetupType,
            Structure,
        )
        ctx = MarketContext(
            structure=Structure.BULLISH, swings=[], sr_levels=[], msbs=[],
            range_bounds=None, atr=1.0, current_price=100.0, fib_levels=None,
        )
        return BoundarySignal.from_setup_signal(
            SetupSignal(
                coin="BTC", setup_type=SetupType.A, direction=Direction.LONG,
                confidence=0.5, key_level=100.0, context=ctx, description="x",
            ),
            "4h",
        )

    def test_max_retries_is_five(self):
        from scanner.analysis import analyze_for_scanner

        # Verify the SDK construction kwargs by short-circuiting at messages.create
        # (the construction happens before the API call, so any post-construction
        # exception still lets us inspect MockClient.call_args).
        with patch("scanner.analysis.anthropic.Anthropic") as MockClient:
            instance = MagicMock()
            instance.messages.create.side_effect = RuntimeError("short-circuit")
            MockClient.return_value = instance
            try:
                analyze_for_scanner(self._bs(), [], None)
            except RuntimeError:
                pass  # expected — only the SDK construction matters here

        # The contract: SDK is constructed with max_retries=5
        # (captures ~15s of exponential backoff for short 529 blips).
        MockClient.assert_called_once()
        kwargs = MockClient.call_args.kwargs
        assert kwargs.get("max_retries") == 5


class TestLLMOverloadedTranslation:
    """SPEC #37 + architecture-pass MED: 529 SDK errors get translated to a
    scanner-domain LLMOverloaded exception inside analyze_for_scanner, so
    callers don't depend on the anthropic SDK class hierarchy. Non-529 status
    errors propagate unchanged."""

    def _bs(self):
        from scanner.models import (
            BoundarySignal,
            Direction,
            MarketContext,
            SetupSignal,
            SetupType,
            Structure,
        )
        ctx = MarketContext(
            structure=Structure.BULLISH, swings=[], sr_levels=[], msbs=[],
            range_bounds=None, atr=1.0, current_price=100.0, fib_levels=None,
        )
        return BoundarySignal.from_setup_signal(
            SetupSignal(
                coin="BTC", setup_type=SetupType.A, direction=Direction.LONG,
                confidence=0.5, key_level=100.0, context=ctx, description="x",
            ),
            "4h",
        )

    def _make_api_status_error(self, status_code: int, request_id: str | None):
        import anthropic
        response = MagicMock()
        response.headers.get.return_value = request_id
        err = anthropic.APIStatusError(
            message="x", response=response, body={"error": {"type": "x"}}
        )
        err.status_code = status_code  # type: ignore[misc]
        err.request_id = request_id
        return err

    def test_529_translates_to_llm_overloaded(self):
        from scanner.analysis import LLMOverloaded, analyze_for_scanner

        err = self._make_api_status_error(529, "req_test_529")
        with patch("scanner.analysis.anthropic.Anthropic") as MockClient:
            instance = MagicMock()
            instance.messages.create.side_effect = err
            MockClient.return_value = instance

            try:
                analyze_for_scanner(self._bs(), [], None)
                raised = None
            except LLMOverloaded as e:
                raised = e

        assert raised is not None
        assert raised.request_id == "req_test_529"

    def test_non_529_status_error_propagates_unchanged(self):
        import anthropic

        from scanner.analysis import LLMOverloaded, analyze_for_scanner

        err = self._make_api_status_error(500, "req_test_500")
        with patch("scanner.analysis.anthropic.Anthropic") as MockClient:
            instance = MagicMock()
            instance.messages.create.side_effect = err
            MockClient.return_value = instance

            try:
                analyze_for_scanner(self._bs(), [], None)
                raised = None
            except LLMOverloaded:
                raised = "overload"  # would be wrong
            except anthropic.APIStatusError as e:
                raised = e

        assert isinstance(raised, anthropic.APIStatusError)
        assert raised.status_code == 500
