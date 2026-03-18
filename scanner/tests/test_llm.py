"""Tests for scanner.llm — parse_machine_block, validate_execution_params, parse_decision_block."""

import json

from scanner.llm import parse_decision_block, parse_machine_block, validate_execution_params

# --- Helpers ---


def _valid_block(**overrides) -> dict:
    base = {
        "entry_low": 61500.0, "entry_high": 62000.0,
        "stop_loss": 59800.0, "tp1": 65000.0, "tp1_exit_pct": 50,
        "direction": "LONG", "confidence": 7, "leverage": 5,
    }
    base.update(overrides)
    return base


def _wrap(data: dict) -> str:
    return f"<!-- MACHINE_BLOCK\n{json.dumps(data)}\nMACHINE_BLOCK -->"


# --- TestParseMachineBlock ---

class TestParseMachineBlock:

    def test_valid_block(self):
        result = parse_machine_block(_wrap(_valid_block()))
        assert result is not None
        assert result["entry_low"] == 61500.0
        assert result["direction"] == "LONG"

    def test_missing_block(self):
        assert parse_machine_block("No machine block here") is None

    def test_malformed_json(self):
        text = "<!-- MACHINE_BLOCK\n{bad json!!}\nMACHINE_BLOCK -->"
        assert parse_machine_block(text) is None

    def test_missing_required_field(self):
        data = _valid_block()
        del data["stop_loss"]
        assert parse_machine_block(_wrap(data)) is None

    def test_wrong_type_string_for_number(self):
        data = _valid_block(entry_low="not a number")
        assert parse_machine_block(_wrap(data)) is None

    def test_int_coerced_to_float(self):
        data = _valid_block(entry_low=61500, entry_high=62000, confidence=7)
        result = parse_machine_block(_wrap(data))
        assert result is not None
        assert isinstance(result["entry_low"], float)
        assert isinstance(result["confidence"], float)

    def test_optional_tp2(self):
        data = _valid_block(tp2=68000.0, tp2_exit_pct=50)
        result = parse_machine_block(_wrap(data))
        assert result is not None
        assert result["tp2"] == 68000.0
        assert result["tp2_exit_pct"] == 50.0

    def test_optional_tp2_wrong_type(self):
        data = _valid_block(tp2="high")
        assert parse_machine_block(_wrap(data)) is None

    def test_embedded_in_full_analysis(self):
        text = (
            "## DECISION: CONFIRM\n"
            "**Asset:** BTC/USDT\n"
            "**Direction:** LONG\n"
            "**Entry Zone:** $61,500 - $62,000\n"
            "**Stop Loss:** $59,800\n"
            "**Take Profit:** $65,000\n\n"
            + _wrap(_valid_block()) + "\n"
        )
        result = parse_machine_block(text)
        assert result is not None
        assert result["entry_low"] == 61500.0

    def test_whitespace_tolerance(self):
        text = "<!--  MACHINE_BLOCK \n" + json.dumps(_valid_block()) + "\nMACHINE_BLOCK  -->"
        result = parse_machine_block(text)
        assert result is not None

    def test_non_dict_json_rejected(self):
        text = "<!-- MACHINE_BLOCK\n[1, 2, 3]\nMACHINE_BLOCK -->"
        assert parse_machine_block(text) is None

    def test_rejects_long_stop_above_entry(self):
        """LONG with stop above entry_low rejected at parse time."""
        assert parse_machine_block(_wrap(_valid_block(stop_loss=63000.0))) is None

    def test_rejects_zero_price(self):
        assert parse_machine_block(_wrap(_valid_block(entry_low=0))) is None

    def test_rejects_confidence_out_of_range(self):
        assert parse_machine_block(_wrap(_valid_block(confidence=0))) is None

    def test_valid_block_passes_with_validation(self):
        """Regression: default valid block still passes."""
        result = parse_machine_block(_wrap(_valid_block()))
        assert result is not None


# --- TestValidateExecutionParams ---

class TestValidateExecutionParams:

    def test_valid_long(self):
        valid, errors = validate_execution_params(_valid_block())
        assert valid is True
        assert errors == []

    def test_valid_short(self):
        params = {
            "entry_low": 62000.0, "entry_high": 62500.0,
            "stop_loss": 64000.0, "tp1": 59000.0, "tp1_exit_pct": 50,
            "direction": "SHORT", "confidence": 6, "leverage": 3,
        }
        valid, errors = validate_execution_params(params)
        assert valid is True

    def test_invalid_direction(self):
        valid, errors = validate_execution_params(_valid_block(direction="FLAT"))
        assert valid is False
        assert any("direction" in e for e in errors)

    def test_long_stop_wrong_side(self):
        valid, errors = validate_execution_params(_valid_block(stop_loss=63000.0))
        assert valid is False
        assert any("stop" in e.lower() for e in errors)

    def test_long_tp_wrong_side(self):
        valid, errors = validate_execution_params(_valid_block(tp1=60000.0))
        assert valid is False
        assert any("tp1" in e.lower() for e in errors)

    def test_short_stop_wrong_side(self):
        params = _valid_block(direction="SHORT", entry_low=100.0, entry_high=101.0,
                              stop_loss=99.0, tp1=90.0)
        valid, errors = validate_execution_params(params)
        assert valid is False
        assert any("stop" in e.lower() for e in errors)

    def test_short_tp_wrong_side(self):
        params = _valid_block(direction="SHORT", entry_low=100.0, entry_high=101.0,
                              stop_loss=105.0, tp1=110.0)
        valid, errors = validate_execution_params(params)
        assert valid is False
        assert any("tp1" in e.lower() for e in errors)

    def test_entry_low_gt_entry_high(self):
        valid, errors = validate_execution_params(_valid_block(entry_low=63000.0, entry_high=62000.0))
        assert valid is False
        assert any("entry_low" in e for e in errors)

    def test_exit_pcts_over_100(self):
        valid, errors = validate_execution_params(
            _valid_block(tp1_exit_pct=60, tp2=68000.0, tp2_exit_pct=50)
        )
        assert valid is False
        assert any("100" in e for e in errors)

    def test_confidence_too_low(self):
        valid, errors = validate_execution_params(_valid_block(confidence=0))
        assert valid is False
        assert any("confidence" in e for e in errors)

    def test_confidence_too_high(self):
        valid, errors = validate_execution_params(_valid_block(confidence=11))
        assert valid is False

    def test_leverage_zero(self):
        valid, errors = validate_execution_params(_valid_block(leverage=0))
        assert valid is False
        assert any("leverage" in e for e in errors)

    def test_max_leverage_exceeded(self):
        valid, errors = validate_execution_params(_valid_block(leverage=10), max_leverage=5)
        assert valid is False
        assert any("max" in e.lower() for e in errors)

    def test_max_leverage_ok(self):
        valid, errors = validate_execution_params(_valid_block(leverage=5), max_leverage=10)
        assert valid is True

    def test_negative_price(self):
        valid, errors = validate_execution_params(_valid_block(entry_low=-100.0))
        assert valid is False
        assert any("positive" in e for e in errors)

    def test_tp2_ordering_long(self):
        params = _valid_block(tp2=64000.0, tp2_exit_pct=50)
        valid, errors = validate_execution_params(params)
        assert valid is False
        assert any("tp2" in e.lower() for e in errors)

    def test_multiple_errors_collected(self):
        params = _valid_block(direction="INVALID", confidence=0, leverage=-1)
        valid, errors = validate_execution_params(params)
        assert valid is False
        assert len(errors) >= 3


# --- TestParseDecisionBlock ---

SAMPLE_DECISION = """
## DECISION: CONFIRM

**Asset:** BTC/USDT
**Timeframe:** 4H
**Market Structure:** BULLISH

**Setup:** C: S/R Flip Continuation
**Confluence Score:** 9/17 — A+
**Direction:** LONG

**Entry Trigger:** Limit order at prior resistance-turned-support
**Entry Zone:** $61,500 - $62,000
**Invalidation:** $59,800 — below the swing low
**Target:** $68,000

**Risk Guidance:** 1-2%
**Reasoning:** Clean S/R flip with trend alignment and fresh level.
**What Would Change This:** A close below $59,800 on 4H would invalidate.
"""


class TestParseDecisionBlock:
    def test_extracts_decision(self):
        result = parse_decision_block(SAMPLE_DECISION)
        assert result["decision"] == "CONFIRM"

    def test_extracts_asset(self):
        result = parse_decision_block(SAMPLE_DECISION)
        assert result["asset"] == "BTC/USDT"

    def test_extracts_direction(self):
        result = parse_decision_block(SAMPLE_DECISION)
        assert result["direction"] == "LONG"

    def test_extracts_entry_zone(self):
        result = parse_decision_block(SAMPLE_DECISION)
        assert "$61,500" in result["entry_zone"]

    def test_extracts_invalidation(self):
        result = parse_decision_block(SAMPLE_DECISION)
        assert "$59,800" in result["invalidation"]

    def test_extracts_target(self):
        result = parse_decision_block(SAMPLE_DECISION)
        assert result["target"] == "$68,000"

    def test_reject_decision(self):
        text = "## DECISION: REJECT\n**Asset:** ETH/USDT"
        result = parse_decision_block(text)
        assert result["decision"] == "REJECT"

    def test_wait_decision(self):
        text = "## DECISION: WAIT\n**Asset:** SOL/USDT"
        result = parse_decision_block(text)
        assert result["decision"] == "WAIT"

    def test_empty_text(self):
        result = parse_decision_block("")
        assert result == {}

    def test_extracts_llm_confidence(self):
        text = "## DECISION: CONFIRM\n**LLM Confidence:** 8 — Solid setup with clear invalidation"
        result = parse_decision_block(text)
        assert result["llm_confidence"] == "8 — Solid setup with clear invalidation"

    def test_missing_fields_graceful(self):
        text = "## DECISION: CONFIRM\n**Asset:** BTC"
        result = parse_decision_block(text)
        assert result["decision"] == "CONFIRM"
        assert result["asset"] == "BTC"
        assert "entry_zone" not in result
