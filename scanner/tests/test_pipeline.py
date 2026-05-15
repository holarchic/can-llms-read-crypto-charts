"""Tests for pipeline helpers (diversity selection, candle collection)."""

from unittest.mock import MagicMock, patch

from scanner.config import ScannerConfig
from scanner.models import (
    Direction,
    MarketContext,
    ScanResult,
    SetupSignal,
    SetupType,
    Structure,
)
from datetime import datetime, timezone

from scanner.pipeline import _select_with_diversity, stage1_screen, stage4_execute


def _make_signal(
    coin="BTC",
    setup_type=SetupType.A,
    direction=Direction.LONG,
    confidence=0.5,
):
    ctx = MarketContext(
        structure=Structure.BULLISH,
        swings=[],
        sr_levels=[],
        msbs=[],
        range_bounds=None,
        atr=1.0,
        current_price=100.0,
        fib_levels=None,
    )
    return SetupSignal(
        coin=coin,
        setup_type=setup_type,
        direction=direction,
        confidence=confidence,
        key_level=100.0,
        context=ctx,
        description=f"Test {coin} {setup_type.value}",
    )


class TestSelectWithDiversity:
    def test_empty_list(self):
        assert _select_with_diversity([], 10) == []

    def test_under_max_returned_unchanged(self):
        signals = [_make_signal(coin=f"C{i}") for i in range(5)]
        result = _select_with_diversity(signals, 10)
        assert result == signals

    def test_exact_max_returned_unchanged(self):
        signals = [_make_signal(coin=f"C{i}") for i in range(10)]
        result = _select_with_diversity(signals, 10)
        assert result == signals

    def test_total_length_preserved(self):
        """Selected + remaining = original length."""
        signals = [_make_signal(coin=f"C{i}") for i in range(30)]
        result = _select_with_diversity(signals, 10)
        assert len(result) == 30

    def test_dominant_type_penalized(self):
        """15 Type B + 5 Type E with max=10 → Type E gets more slots than 5/20 proportion."""
        signals = (
            [_make_signal(coin=f"B{i}", setup_type=SetupType.B, confidence=0.6) for i in range(15)]
            + [_make_signal(coin=f"E{i}", setup_type=SetupType.E, confidence=0.5) for i in range(5)]
        )
        result = _select_with_diversity(signals, 10)
        selected = result[:10]
        type_e_count = sum(1 for s in selected if s.setup_type == SetupType.E)
        # Diversity penalty interleaves B and E picks. Without penalty, random would give
        # ~2.5 Type E (5/20). With decay=0.85, B's higher confidence (0.6 vs 0.5) still
        # wins alternating slots, but E gets 4 of 10 — much better than random.
        assert type_e_count == 4

    def test_diverse_types_all_represented(self):
        """One signal per type — all should be selected."""
        types = [SetupType.A, SetupType.B, SetupType.C, SetupType.D, SetupType.E, SetupType.F, SetupType.G]
        signals = [_make_signal(coin=f"C{i}", setup_type=t, confidence=0.5) for i, t in enumerate(types)]
        result = _select_with_diversity(signals, 5)
        selected_types = {s.setup_type for s in result[:5]}
        assert len(selected_types) == 5  # 5 distinct types in 5 slots

    def test_decay_one_no_diversity_effect(self):
        """decay=1.0 means no penalty — selection purely by confidence."""
        signals = (
            [_make_signal(coin=f"B{i}", setup_type=SetupType.B, confidence=0.9) for i in range(10)]
            + [_make_signal(coin=f"E{i}", setup_type=SetupType.E, confidence=0.1) for i in range(10)]
        )
        result = _select_with_diversity(signals, 10, decay=1.0)
        selected = result[:10]
        type_b_count = sum(1 for s in selected if s.setup_type == SetupType.B)
        # With no decay, all high-confidence Type Bs dominate
        assert type_b_count == 10

    def test_decay_zero_strict_round_robin(self):
        """decay=0.0 means after first pick, score=0 for same type → strict round-robin."""
        signals = (
            [_make_signal(coin=f"B{i}", setup_type=SetupType.B, confidence=0.9) for i in range(10)]
            + [_make_signal(coin=f"E{i}", setup_type=SetupType.E, confidence=0.1) for i in range(10)]
        )
        result = _select_with_diversity(signals, 10, decay=0.0)
        selected = result[:10]
        type_b_count = sum(1 for s in selected if s.setup_type == SetupType.B)
        type_e_count = sum(1 for s in selected if s.setup_type == SetupType.E)
        # First pick: B (0.9). Second: E (0.1 > 0.0). Then all scores are 0.
        # After that, remaining picks are all score=0 ties (first in list wins).
        # So we should have at least 1 B and at least 1 E.
        assert type_b_count >= 1
        assert type_e_count >= 1

    def test_selected_first_remaining_after(self):
        """First max_count items are the selected ones, rest are remaining."""
        signals = (
            [_make_signal(coin=f"B{i}", setup_type=SetupType.B, confidence=0.6) for i in range(15)]
            + [_make_signal(coin=f"E{i}", setup_type=SetupType.E, confidence=0.5) for i in range(5)]
        )
        result = _select_with_diversity(signals, 10)
        _selected = result[:10]
        _remaining = result[10:]
        # All signals accounted for, no duplicates
        assert len(result) == 20
        all_coins = [s.coin for s in result]
        assert len(set(all_coins)) == 20

    def test_confidence_tiebreak_within_type(self):
        """Among same type with same penalty, higher confidence wins."""
        signals = [
            _make_signal(coin="LOW", setup_type=SetupType.A, confidence=0.3),
            _make_signal(coin="HIGH", setup_type=SetupType.A, confidence=0.9),
            _make_signal(coin="MED", setup_type=SetupType.A, confidence=0.6),
        ]
        result = _select_with_diversity(signals, 2)
        # First pick: HIGH (0.9). Second: MED (0.6 * 0.85 = 0.51 > LOW 0.3 * 0.85 = 0.255)
        assert result[0].coin == "HIGH"
        assert result[1].coin == "MED"


class TestTestnetPairFilter:
    """stage1_screen filters pairs not available on testnet when testnet_coins provided."""

    def test_filters_pairs_not_on_testnet(self):
        """Pairs with coins not in testnet_coins are removed before candle fetch."""
        testnet_coins = {"BTC", "ETH"}
        pairs = [
            {"coin": "BTC", "symbol": "BTC/USDC:USDC", "max_leverage": 50},
            {"coin": "ETH", "symbol": "ETH/USDC:USDC", "max_leverage": 25},
            {"coin": "ENA", "symbol": "ENA/USDC:USDC", "max_leverage": 10},
        ]

        with patch("scanner.pipeline.fetch_pairs_with_volume", return_value=pairs), \
             patch("scanner.pipeline.fetch_candles_batch", return_value={}) as mock_batch, \
             patch("scanner.pipeline.fetch_candles"), \
             patch("scanner.pipeline.get_connection") as mock_conn, \
             patch("scanner.pipeline.get_strategy", return_value=None), \
             patch("scanner.pipeline.save_scan_universe"), \
             patch("scanner.pipeline.save_signals"):
            mock_conn.return_value = MagicMock()

            stage1_screen(ScannerConfig(), verbose=False, testnet_coins=testnet_coins)

            # fetch_candles_batch should only get BTC and ETH
            called_coins = mock_batch.call_args[0][0]
            assert "ENA" not in called_coins
            assert set(called_coins) == {"BTC", "ETH"}

    def test_no_filter_when_testnet_coins_none(self):
        """All pairs pass through when testnet_coins is None (non-testnet mode)."""
        pairs = [
            {"coin": "BTC", "symbol": "BTC/USDC:USDC", "max_leverage": 50},
            {"coin": "ENA", "symbol": "ENA/USDC:USDC", "max_leverage": 10},
        ]

        with patch("scanner.pipeline.fetch_pairs_with_volume", return_value=pairs), \
             patch("scanner.pipeline.fetch_candles_batch", return_value={}) as mock_batch, \
             patch("scanner.pipeline.fetch_candles"), \
             patch("scanner.pipeline.get_connection") as mock_conn, \
             patch("scanner.pipeline.get_strategy", return_value=None), \
             patch("scanner.pipeline.save_scan_universe"), \
             patch("scanner.pipeline.save_signals"):
            mock_conn.return_value = MagicMock()

            stage1_screen(ScannerConfig(), verbose=False, testnet_coins=None)

            called_coins = mock_batch.call_args[0][0]
            assert set(called_coins) == {"BTC", "ENA"}


def _make_scan_result(coin="BTC"):
    return ScanResult(
        scan_id=f"scan-{coin}",
        coin=coin,
        timestamp=datetime.now(timezone.utc),
        setup_signal=_make_signal(coin=coin),
        decision="CONFIRM",
        confluence_score=None,
        analysis_text=None,
        parsed_decision=None,
        chart_paths=[],
        cost_usd=0.0,
        machine_block={
            "direction": "LONG",
            "entry_trigger": "limit_at",
            "entry_trigger_price": 95.0,
            "invalidation_price": 90.0,
        },
    )


class TestStage4ConditionalRouting:
    """A CONFIRM that qualifies as a conditional entry must never fall through
    to an immediate market order when save_conditional_entry returns False.
    Falling through bypassed pending dedupe and double-exposed when the
    existing pending later triggered (Codex audit 2026-05-15, R1)."""

    def _run(self, saved):
        confirm = _make_scan_result()
        exec_config = MagicMock(enabled=True, mode="testnet",
                                conditional_entry_enabled=True)
        with patch("scanner.pipeline.get_connection", return_value=MagicMock()), \
             patch("scanner.execution.entry_monitor.save_conditional_entry",
                   return_value=saved), \
             patch("scanner.execution.executor.execute_trade") as mock_exec, \
             patch("scanner.execution.executor.send_skip_alert"):
            results = stage4_execute([confirm], ScannerConfig(), exec_config,
                                     verbose=False)
        return results, mock_exec

    def test_reject_does_not_market_fill(self):
        results, mock_exec = self._run(saved=False)
        mock_exec.assert_not_called()
        assert len(results) == 1
        assert results[0].skip_reason == "conditional_rejected"

    def test_saved_records_pending(self):
        results, mock_exec = self._run(saved=True)
        mock_exec.assert_not_called()
        assert len(results) == 1
        assert results[0].skip_reason == "conditional_pending"
