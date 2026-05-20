"""Tests for pipeline helpers (diversity selection, candle collection)."""

from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from scanner.config import ScannerConfig
from scanner.models import (
    Direction,
    MachineBlock,
    MarketContext,
    ScanResult,
    SetupSignal,
    SetupType,
    Structure,
)
from scanner.pipeline import (
    _build_summary,
    _empty_summary,
    _filter_pending_conditional,
    _post_scan_ops_log,
    _print_summary,
    _report_dry_run,
    _resolve_testnet_coins,
    _select_with_diversity,
    run_scan,
    stage1_screen,
    stage4_execute,
)


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
        # T-016 C5: ScanResult.machine_block is a typed MachineBlock. A real
        # conditional CONFIRM always carries a complete, valid block — build
        # one through the canonical boundary (LONG: stop below entry, tp
        # above, inv >= stop).
        machine_block=MachineBlock.from_raw({
            "direction": "LONG",
            "entry_low": 99.0, "entry_high": 101.0,
            "stop_loss": 89.0, "tp1": 110.0, "tp1_exit_pct": 50,
            "confidence": 7, "leverage": 5,
            "entry_trigger": "limit_at",
            "entry_trigger_price": 95.0,
            "invalidation_price": 90.0,
        }),
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


class TestStage4StalenessRefAdapter:
    """The detected_at-or-timestamp staleness fallback moved from execute_trade
    to the stage4 ExecutionRequest adapter (SPEC #21). Guard it at the layer
    that now owns it: the request's resolved `staleness_ref` must be
    `confirm.detected_at` when set, else `confirm.timestamp`."""

    def _plain_confirm(self, detected_at, timestamp):
        # Non-conditional block (no trigger fields) so stage4 builds the
        # ExecutionRequest and calls execute_trade rather than routing to
        # save_conditional_entry.
        return ScanResult(
            scan_id="scan-stale",
            coin="BTC",
            timestamp=timestamp,
            setup_signal=_make_signal(coin="BTC"),
            decision="CONFIRM",
            confluence_score=None,
            analysis_text=None,
            parsed_decision=None,
            chart_paths=[],
            cost_usd=0.0,
            machine_block=MachineBlock.from_raw({
                "direction": "LONG",
                "entry_low": 99.0, "entry_high": 101.0,
                "stop_loss": 89.0, "tp1": 110.0, "tp1_exit_pct": 50,
                "confidence": 7, "leverage": 5,
            }),
            detected_at=detected_at,
        )

    def _captured_request(self, confirm):
        exec_config = MagicMock(enabled=True, mode="testnet",
                                conditional_entry_enabled=True)
        with patch("scanner.pipeline.get_connection", return_value=MagicMock()), \
             patch("scanner.execution.executor.execute_trade") as mock_exec, \
             patch("scanner.execution.executor.send_skip_alert"):
            mock_exec.return_value = MagicMock(executed=True, skip_reason="")
            stage4_execute([confirm], ScannerConfig(), exec_config, verbose=False)
        mock_exec.assert_called_once()
        return mock_exec.call_args[0][0]

    def test_detected_at_used_when_present(self):
        det = datetime(2026, 5, 16, 9, 0, tzinfo=timezone.utc)
        ts = datetime(2026, 5, 16, 12, 0, tzinfo=timezone.utc)
        req = self._captured_request(self._plain_confirm(det, ts))
        assert req.staleness_ref == det

    def test_falls_back_to_timestamp_when_detected_at_none(self):
        ts = datetime(2026, 5, 16, 12, 0, tzinfo=timezone.utc)
        req = self._captured_request(self._plain_confirm(None, ts))
        assert req.staleness_ref == ts


# --- T-016 Phase 4b: run_scan → thin dispatcher (SPEC #28) ---
# Pure relocation. These pin (a) the one non-verbatim edit — the 2→1
# empty-summary dedup — and the closed-over-state threading the codex
# plan-review R1 HIGH flagged (`alerted`, raw vs rounded `elapsed`); and
# (b) run_scan dispatcher wiring/short-circuits (codex R2 — nothing in-tree
# exercised run_scan before this phase).


class TestPhase4bHelpers:
    def test_empty_summary_is_the_exact_pre_split_literal(self):
        # The two pre-split early-return literals (pipeline L442 & L461) were
        # byte-identical; this is their single owner — value-identical.
        assert _empty_summary() == {
            "signals": 0, "results": 0, "confirms": 0, "alerted": 0,
        }

    def test_report_dry_run_shape(self):
        sigs = [_make_signal(coin=f"C{i}") for i in range(3)]
        assert _report_dry_run(sigs, verbose=False) == {
            "signals": 3, "dry_run": True,
        }
        assert _report_dry_run([], verbose=False) == {
            "signals": 0, "dry_run": True,
        }

    def test_resolve_testnet_coins_testnet_mode_returns_market_set(self):
        cfg = ScannerConfig()
        exec_config = MagicMock(mode="testnet")
        with patch("scanner.data.get_testnet_coins",
                   return_value={"BTC", "ETH"}) as gtc:
            out = _resolve_testnet_coins(cfg, exec_config, "HyperLiquid",
                                         verbose=False)
        gtc.assert_called_once_with(cfg.exchange)
        assert out == {"BTC", "ETH"}

    def test_resolve_testnet_coins_none_when_not_testnet(self):
        cfg = ScannerConfig()
        assert _resolve_testnet_coins(cfg, None, "HL", verbose=False) is None
        assert _resolve_testnet_coins(
            cfg, MagicMock(mode="live"), "HL", verbose=False) is None

    def test_resolve_testnet_coins_none_when_lookup_raises(self):
        # Verbatim try/except: a failed testnet lookup → proceed unfiltered.
        cfg = ScannerConfig()
        exec_config = MagicMock(mode="testnet")
        with patch("scanner.data.get_testnet_coins",
                   side_effect=RuntimeError("boom")):
            assert _resolve_testnet_coins(
                cfg, exec_config, "HL", verbose=False) is None

    def test_build_summary_threads_alerted_and_keeps_rounding(self):
        # codex R1 HIGH regression-pin: `alerted` is a closed-over local the
        # summary reads, and `elapsed_sec` must stay round(elapsed, 1) while
        # the raw float is what _print_summary consumes.
        # Stub derives is_confirm/is_reject/is_wait from `decision` at access
        # time (real-property semantics), so future predicate refactors are
        # caught — vs. a MagicMock with hard-set booleans that would silently
        # pass even if the property logic regressed.
        @dataclass
        class _R:
            decision: str | None
            cost_usd: float
            @property
            def is_confirm(self) -> bool: return self.decision == "CONFIRM"
            @property
            def is_reject(self) -> bool: return self.decision == "REJECT"
            @property
            def is_wait(self) -> bool: return self.decision == "WAIT"
        results = [
            _R("CONFIRM", 0.1),
            _R("REJECT", 0.2),
            _R("WAIT", 0.05),
            _R(None, 0.0),
        ]
        exec_results = [MagicMock(executed=True), MagicMock(executed=False)]
        summary = _build_summary(
            signals=[1, 2, 3], results=results, exec_results=exec_results,
            alerted=7, elapsed=123.456,
        )
        assert summary == {
            "signals": 3,
            "results": 4,
            "confirms": 1,
            "rejects": 1,
            "waits": 1,
            "alerted": 7,
            "executed": 1,
            "cost_usd": round(0.1 + 0.2 + 0.05 + 0.0, 4),
            "elapsed_sec": round(123.456, 1),
        }

    def test_filter_pending_conditional_drops_pending_coins(self):
        sigs = [_make_signal(coin="BTC"), _make_signal(coin="ETH")]
        exec_config = MagicMock(exchange="hyperliquid", mode="live")
        with patch("scanner.pipeline.get_connection",
                   return_value=MagicMock()), \
             patch("scanner.execution.storage.get_active_pending_entries",
                   return_value=[{"coin": "BTC"}]):
            out = _filter_pending_conditional(sigs, exec_config, verbose=False)
        assert [s.coin for s in out] == ["ETH"]

    def test_filter_pending_conditional_passthrough_when_no_pending(self):
        sigs = [_make_signal(coin="BTC"), _make_signal(coin="ETH")]
        exec_config = MagicMock(exchange="hyperliquid", mode="live")
        with patch("scanner.pipeline.get_connection",
                   return_value=MagicMock()), \
             patch("scanner.execution.storage.get_active_pending_entries",
                   return_value=[]):
            out = _filter_pending_conditional(sigs, exec_config, verbose=False)
        assert out == sigs

    def test_raw_vs_rounded_elapsed_boundary(self, capsys):
        # codex phase-review (correctness LOW) regression-pin for the codex
        # plan-review R1 HIGH: the verbose duration line MUST print the RAW
        # elapsed (`{elapsed:.0f}`) while the ops-log Duration reads the
        # ROUNDED `elapsed_sec`. 123.456 is a boundary where these differ —
        # so this test proves run_scan threading the raw float is
        # load-bearing, not cosmetic.
        elapsed = 123.456
        raw_s = f"{elapsed:.0f}s"                 # "123s"
        rounded_s = f"{round(elapsed, 1):.0f}s"   # "124s"
        assert raw_s != rounded_s  # the boundary actually bites

        summary = _build_summary(
            signals=[1], results=[], exec_results=[],
            alerted=0, elapsed=elapsed,
        )
        assert summary["elapsed_sec"] == round(elapsed, 1)

        _print_summary(summary, elapsed)
        printed = capsys.readouterr().out
        assert f"Scan complete in {raw_s}" in printed
        assert f"Scan complete in {rounded_s}" not in printed

        with patch("scanner.pipeline.send_ops_log") as ops:
            _post_scan_ops_log(summary, [], "HyperLiquid", MagicMock())
        assert ops.call_args.args[1]["Duration"] == rounded_s


class TestRunScanDispatcher:
    """run_scan is now a thin dispatcher. Pin the short-circuits and the
    full-path stage/helper call order + summary handoff (codex R2)."""

    def test_dry_run_short_circuits_before_stage2(self):
        sig = _make_signal(coin="BTC")
        with patch("scanner.pipeline.stage1_screen",
                   return_value=([sig], {})), \
             patch("scanner.pipeline.stage2_analyze") as s2, \
             patch("scanner.pipeline.stage3_alert") as s3, \
             patch("scanner.pipeline.stage4_execute") as s4:
            out = run_scan(dry_run=True, verbose=False)
        assert out == {"signals": 1, "dry_run": True}
        s2.assert_not_called()
        s3.assert_not_called()
        s4.assert_not_called()

    def test_no_signals_short_circuits_to_empty_summary(self):
        with patch("scanner.pipeline.stage1_screen",
                   return_value=([], {})), \
             patch("scanner.pipeline.stage2_analyze") as s2:
            out = run_scan(verbose=False, skip_discord=True)
        assert out == _empty_summary()
        s2.assert_not_called()

    def test_full_path_call_order_and_summary_handoff(self):
        sig = _make_signal(coin="BTC")
        result = _make_scan_result(coin="BTC")  # decision == "CONFIRM"
        exec_config = MagicMock(conditional_entry_enabled=True, enabled=True,
                                mode="live", exchange="hyperliquid")
        sentinel = {"sentinel": "summary"}
        mgr = MagicMock()
        mgr.s1.return_value = ([sig], {})
        mgr.filt.return_value = [sig]
        mgr.s2.return_value = [result]
        mgr.s3.return_value = 2
        mgr.s4.return_value = []
        with patch("scanner.pipeline.stage1_screen", mgr.s1), \
             patch("scanner.pipeline._filter_pending_conditional", mgr.filt), \
             patch("scanner.pipeline.stage2_analyze", mgr.s2), \
             patch("scanner.pipeline.stage3_alert", mgr.s3), \
             patch("scanner.pipeline.stage4_execute", mgr.s4), \
             patch("scanner.pipeline._build_summary",
                   return_value=sentinel) as bs, \
             patch("scanner.pipeline._post_scan_ops_log") as ops:
            out = run_scan(exec_config=exec_config, verbose=False,
                           skip_discord=True)

        # Returned summary is exactly _build_summary's output.
        assert out is sentinel
        # _build_summary received the stage3 return as `alerted`.
        assert bs.call_args.args[3] == 2
        # skip_discord=True ⇒ ops-log suppressed.
        ops.assert_not_called()
        # Dispatch order: stage1 → pending-filter → stage2 → stage3 → stage4.
        ordered = [c[0] for c in mgr.mock_calls if c[0] in
                   {"s1", "filt", "s2", "s3", "s4"}]
        assert ordered == ["s1", "filt", "s2", "s3", "s4"]
