"""Tests for walk-forward trade resolution."""

import json

import pytest

from scanner.backtest.resolver import TradeSetup, build_llm_setup, estimate_slippage, resolve_signal

from .conftest import make_candles, make_candles_from_hlc


def _setup(direction="LONG", entry=100.0, stop=95.0, target=110.0, ts="2026-01-01T00:00:00+00:00"):
    return TradeSetup(
        coin="BTC", direction=direction, entry=entry, stop=stop, target=target,
        timestamp=ts, setup_type="C", confidence=0.7, structure="BULLISH",
        sent_to_llm=False, source="screener",
    )


class TestLongResolution:
    """Walk-forward for LONG setups."""

    def test_win_when_target_hit(self):
        # Candles drift up, hitting target of 110
        candles = make_candles_from_hlc([
            (102, 99, 101),   # bar 1
            (105, 100, 104),  # bar 2
            (108, 103, 107),  # bar 3
            (111, 106, 110),  # bar 4 — high >= 110 target
        ])
        setup = _setup()
        outcome = resolve_signal(setup, candles, fee_rate=0.0)

        assert outcome.result == "WIN"
        assert outcome.r_achieved == pytest.approx(2.0, abs=0.01)  # reward=10, risk=5 → 2R
        assert outcome.bars_to_exit == 4

    def test_loss_when_stop_hit(self):
        # Candles drift down, hitting stop of 95
        candles = make_candles_from_hlc([
            (101, 98, 99),
            (100, 96, 97),
            (98, 94, 95),  # bar 3 — low <= 95 stop
        ])
        setup = _setup()
        outcome = resolve_signal(setup, candles, fee_rate=0.0)

        assert outcome.result == "LOSS"
        assert outcome.r_achieved == pytest.approx(-1.0, abs=0.01)
        assert outcome.bars_to_exit == 3

    def test_both_hit_same_candle_is_loss(self):
        # Wide candle hits both stop and target
        candles = make_candles_from_hlc([
            (101, 99, 100),
            (115, 90, 100),  # bar 2 — hits stop=95 AND target=110
        ])
        setup = _setup()
        outcome = resolve_signal(setup, candles, fee_rate=0.0)

        assert outcome.result == "LOSS"
        assert outcome.bars_to_exit == 2

    def test_expired_when_timeout(self):
        # Price drifts sideways, never hits stop or target
        candles = make_candles_from_hlc(
            [(101, 99, 100)] * 200  # 200 bars of nothing
        )
        setup = _setup()
        outcome = resolve_signal(setup, candles, timeout_bars=10, fee_rate=0.0)

        assert outcome.result == "EXPIRED"
        assert outcome.bars_to_exit == 10

    def test_expired_no_future_candles(self):
        # Signal time is after all candles
        candles = make_candles(n=5, start="2025-12-01")
        setup = _setup(ts="2026-02-01T00:00:00+00:00")
        outcome = resolve_signal(setup, candles, fee_rate=0.0)

        assert outcome.result == "EXPIRED"

    def test_unresolved_when_insufficient_bars(self):
        # Only 5 bars of forward data with timeout=72 — not enough to resolve
        candles = make_candles_from_hlc(
            [(101, 99, 100)] * 5
        )
        setup = _setup()
        outcome = resolve_signal(setup, candles, timeout_bars=72, fee_rate=0.0)

        assert outcome.result == "UNRESOLVED"
        assert outcome.r_achieved == 0.0

    def test_win_still_resolves_with_few_bars(self):
        # Target hit on bar 2 — should still be WIN even with few total bars
        candles = make_candles_from_hlc([
            (101, 99, 100),
            (111, 100, 110),  # hits target
        ])
        setup = _setup()
        outcome = resolve_signal(setup, candles, timeout_bars=72, fee_rate=0.0)

        assert outcome.result == "WIN"

    def test_mae_tracks_worst_drawdown(self):
        candles = make_candles_from_hlc([
            (101, 96, 100),   # bar 1 — dips to 96, adverse = (100-96)/5 = 0.8R
            (102, 97, 101),   # bar 2
            (111, 100, 110),  # bar 3 — hits target
        ])
        setup = _setup()
        outcome = resolve_signal(setup, candles, fee_rate=0.0)

        assert outcome.result == "WIN"
        assert outcome.mae == pytest.approx(0.8, abs=0.01)

    def test_mfe_tracks_best_excursion(self):
        candles = make_candles_from_hlc([
            (108, 99, 105),   # bar 1 — high reaches 108, favor = (108-100)/5 = 1.6R
            (100, 94, 95),    # bar 2 — hits stop
        ])
        setup = _setup()
        outcome = resolve_signal(setup, candles, fee_rate=0.0)

        assert outcome.result == "LOSS"
        assert outcome.mfe == pytest.approx(1.6, abs=0.01)

    def test_fees_deducted(self):
        candles = make_candles_from_hlc([
            (111, 100, 110),  # bar 1 — hits target immediately
        ])
        setup = _setup()
        outcome = resolve_signal(setup, candles, fee_rate=0.001)

        # fee_r = (0.001 * 2 * 100) / 5 = 0.04R
        assert outcome.result == "WIN"
        assert outcome.fees_paid == pytest.approx(0.04, abs=0.001)
        assert outcome.r_achieved == pytest.approx(2.0 - 0.04, abs=0.01)


class TestShortResolution:
    """Walk-forward for SHORT setups."""

    def test_short_win(self):
        # entry=100, stop=105, target=90 → risk=5, reward=10
        candles = make_candles_from_hlc([
            (101, 97, 98),
            (99, 93, 94),
            (95, 89, 90),  # bar 3 — low <= 90 target
        ])
        setup = _setup(direction="SHORT", entry=100.0, stop=105.0, target=90.0)
        outcome = resolve_signal(setup, candles, fee_rate=0.0)

        assert outcome.result == "WIN"
        assert outcome.r_achieved == pytest.approx(2.0, abs=0.01)

    def test_short_loss(self):
        candles = make_candles_from_hlc([
            (101, 98, 99),
            (103, 99, 102),
            (106, 101, 105),  # bar 3 — high >= 105 stop
        ])
        setup = _setup(direction="SHORT", entry=100.0, stop=105.0, target=90.0)
        outcome = resolve_signal(setup, candles, fee_rate=0.0)

        assert outcome.result == "LOSS"
        assert outcome.r_achieved == pytest.approx(-1.0, abs=0.01)

    def test_short_both_same_candle_is_loss(self):
        candles = make_candles_from_hlc([
            (110, 85, 95),  # bar 1 — hits stop=105 AND target=90
        ])
        setup = _setup(direction="SHORT", entry=100.0, stop=105.0, target=90.0)
        outcome = resolve_signal(setup, candles, fee_rate=0.0)

        assert outcome.result == "LOSS"

    def test_short_mae_tracks_upside(self):
        candles = make_candles_from_hlc([
            (104, 98, 99),   # bar 1 — adverse = (104-100)/5 = 0.8R
            (100, 89, 90),   # bar 2 — hits target
        ])
        setup = _setup(direction="SHORT", entry=100.0, stop=105.0, target=90.0)
        outcome = resolve_signal(setup, candles, fee_rate=0.0)

        assert outcome.result == "WIN"
        assert outcome.mae == pytest.approx(0.8, abs=0.01)


class TestEdgeCases:
    """Edge cases for resolver."""

    def test_zero_risk_returns_expired(self):
        setup = _setup(entry=100.0, stop=100.0, target=110.0)
        candles = make_candles(n=10)
        outcome = resolve_signal(setup, candles, fee_rate=0.0)
        assert outcome.result == "EXPIRED"

    def test_setup_attached_to_outcome(self):
        candles = make_candles_from_hlc([(111, 100, 110)])
        setup = _setup()
        outcome = resolve_signal(setup, candles, fee_rate=0.0)
        assert outcome.setup is setup
        assert outcome.setup.coin == "BTC"


class TestFundingRate:
    """Funding rate deduction from R-multiples."""

    def test_funding_deducted_on_win(self):
        # Target hit at bar 8 → 1 funding period (8 bars / 8 = 1)
        # funding_cost = 0.0001 * 1 * 100 / 5 = 0.002R
        candles = make_candles_from_hlc(
            [(101, 99, 100)] * 7 + [(111, 100, 110)]
        )
        setup = _setup()
        outcome = resolve_signal(setup, candles, fee_rate=0.0, funding_rate_8h=0.0001)

        assert outcome.result == "WIN"
        assert outcome.funding_paid == pytest.approx(0.002, abs=0.0001)
        assert outcome.r_achieved == pytest.approx(2.0 - 0.002, abs=0.001)

    def test_funding_deducted_on_loss(self):
        # Stop hit at bar 16 → 2 funding periods
        # funding_cost = 0.0001 * 2 * 100 / 5 = 0.004R
        candles = make_candles_from_hlc(
            [(101, 99, 100)] * 15 + [(100, 94, 95)]
        )
        setup = _setup()
        outcome = resolve_signal(setup, candles, fee_rate=0.0, funding_rate_8h=0.0001)

        assert outcome.result == "LOSS"
        assert outcome.funding_paid == pytest.approx(0.004, abs=0.0001)
        assert outcome.r_achieved == pytest.approx(-1.0 - 0.004, abs=0.001)

    def test_funding_on_expired(self):
        # 168 bars → 21 funding periods
        # funding_cost = 0.0001 * 21 * 100 / 5 = 0.042R
        candles = make_candles_from_hlc(
            [(101, 99, 100)] * 200
        )
        setup = _setup()
        outcome = resolve_signal(setup, candles, timeout_bars=168, fee_rate=0.0, funding_rate_8h=0.0001)

        assert outcome.result == "EXPIRED"
        assert outcome.funding_paid == pytest.approx(0.042, abs=0.001)

    def test_zero_funding_rate_no_cost(self):
        candles = make_candles_from_hlc([(111, 100, 110)])
        setup = _setup()
        outcome = resolve_signal(setup, candles, fee_rate=0.0, funding_rate_8h=0.0)

        assert outcome.funding_paid == 0.0
        assert outcome.r_achieved == pytest.approx(2.0)

    def test_funding_plus_fees_combined(self):
        # Bar 8 win: fee_r = (0.001 * 2 * 100) / 5 = 0.04R, funding_r = 0.002R
        candles = make_candles_from_hlc(
            [(101, 99, 100)] * 7 + [(111, 100, 110)]
        )
        setup = _setup()
        outcome = resolve_signal(setup, candles, fee_rate=0.001, funding_rate_8h=0.0001)

        assert outcome.result == "WIN"
        expected_r = 2.0 - 0.04 - 0.002
        assert outcome.r_achieved == pytest.approx(expected_r, abs=0.001)


class TestSlippage:
    """Slippage estimation and integration."""

    def test_high_volume_low_slippage(self):
        assert estimate_slippage(100_000_000) == 0.0002

    def test_medium_volume_medium_slippage(self):
        assert estimate_slippage(20_000_000) == 0.0008

    def test_low_volume_high_slippage(self):
        assert estimate_slippage(5_000_000) == 0.0015

    def test_none_volume_worst_case(self):
        assert estimate_slippage(None) == 0.0015

    def test_zero_volume_worst_case(self):
        assert estimate_slippage(0) == 0.0015

    def test_slippage_deducted_from_r(self):
        # slippage_rate = 0.001 → slippage_r = (0.001 * 2 * 100) / 5 = 0.04R
        candles = make_candles_from_hlc([(111, 100, 110)])
        setup = _setup()
        outcome = resolve_signal(setup, candles, fee_rate=0.0, slippage_rate=0.001)

        assert outcome.result == "WIN"
        assert outcome.r_achieved == pytest.approx(2.0 - 0.04, abs=0.001)

    def test_slippage_plus_fees_combined(self):
        # fee=0.001 + slippage=0.0005 → total_per_side=0.0015
        # cost_r = (0.0015 * 2 * 100) / 5 = 0.06R
        candles = make_candles_from_hlc([(111, 100, 110)])
        setup = _setup()
        outcome = resolve_signal(setup, candles, fee_rate=0.001, slippage_rate=0.0005)

        assert outcome.result == "WIN"
        assert outcome.r_achieved == pytest.approx(2.0 - 0.06, abs=0.001)


class TestBuildLlmSetupMachineBlock:
    """Machine block integration in build_llm_setup."""

    def _row(self, machine_block=None, **overrides):
        base = {
            "coin": "BTC",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "setup_type": "C",
            "screener_confidence": 0.7,
            "entry_zone": "$61,500 - $62,000",
            "stop_loss": "$59,800",
            "take_profit": "$65,000",
            "llm_confidence": 7,
        }
        if machine_block is not None:
            base["machine_block"] = json.dumps(machine_block) if isinstance(machine_block, dict) else machine_block
        base.update(overrides)
        return base

    def test_uses_machine_block_when_present(self):
        mb = {
            "entry_low": 61500.0, "entry_high": 62000.0,
            "stop_loss": 59800.0, "tp1": 65000.0, "tp1_exit_pct": 50,
            "direction": "LONG", "confidence": 7, "leverage": 5,
        }
        setup = build_llm_setup(self._row(machine_block=mb))
        assert setup is not None
        assert setup.entry == pytest.approx(61750.0)  # midpoint
        assert setup.stop == 59800.0
        assert setup.target == 65000.0  # TP1
        assert setup.direction == "LONG"

    def test_falls_back_when_absent(self):
        setup = build_llm_setup(self._row())
        assert setup is not None
        # Text parsing: entry midpoint of $61,500 - $62,000
        assert setup.entry == pytest.approx(61750.0)
        assert setup.stop == 59800.0

    def test_falls_back_on_malformed(self):
        setup = build_llm_setup(self._row(machine_block="not json"))
        assert setup is not None
        # Should fall through to text parsing
        assert setup.entry == pytest.approx(61750.0)

    def test_uses_tp1_as_target(self):
        mb = {
            "entry_low": 100.0, "entry_high": 100.0,
            "stop_loss": 95.0, "tp1": 110.0, "tp1_exit_pct": 50,
            "tp2": 120.0, "tp2_exit_pct": 50,
            "direction": "LONG", "confidence": 8, "leverage": 3,
        }
        setup = build_llm_setup(self._row(machine_block=mb))
        assert setup is not None
        assert setup.target == 110.0  # TP1, not TP2
