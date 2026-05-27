"""Tests for scanner/setup_validator.py — the post-LLM structural gate (Chunk 8a, SPEC #33).

Real `BoundarySignal`/`SpineContext`/`ScanResult` fixtures (no mocks) so the deep path
runs: the confluence-count floor, the v1 no-op, the demotion mechanics, that the
validator trusts its typed input (no downstream re-validation of producer-enforced
preconditions — arch-pass R1/R2), the config-boundary validation, and the
summary-surface partitioning.
"""

from datetime import datetime, timezone

import pytest

from scanner.config import ScannerConfig
from scanner.models import (
    BoundarySignal,
    ConfluenceFactor,
    Direction,
    MachineBlock,
    ScanResult,
    SetupTypeV2,
    SpineContext,
    SpineLevel,
    Structure,
)
from scanner.pipeline import _build_summary, _print_summary
from scanner.setup_validator import (
    REJECT_CONFLUENCE_FLOOR,
    REJECT_NON_TEXTBOOK,
    validate_confirms,
)

_FACTORS = list(ConfluenceFactor)  # 10 members → counts 0..10
_DEFAULT_MB = object()  # sentinel — distinguishes "not supplied" from explicit None


# --- builders ------------------------------------------------------------------


def _mb(setup_quality: str | None = "textbook") -> MachineBlock:
    """Minimal `MachineBlock` for validator tests. Capital-path fields are inert
    (the validator only reads `setup_quality`); kept valid so `from_raw` would
    accept the equivalent dict — same shape the calibration probe uses."""
    return MachineBlock(
        entry_low=1.0, entry_high=1.0, stop_loss=0.9, tp1=1.1, tp1_exit_pct=100.0,
        direction="LONG", confidence=5.0, leverage=1.0,
        setup_quality=setup_quality,
    )


def _spine(regime="trending", atr=2.0, price=100.0) -> SpineContext:
    return SpineContext(
        structure=Structure.BULLISH,
        sr_levels=[SpineLevel(price=price)],
        range_bounds=None,
        atr=atr,
        current_price=price,
        fib_levels=None,
        regime=regime,
    )


def _bsig(
    setup_type=SetupTypeV2.TREND_CONTINUATION,
    n_confluence=2,
    confidence=0.5,
    regime="trending",
    atr=2.0,
    price=100.0,
    coin="BTC",
) -> BoundarySignal:
    return BoundarySignal(
        coin=coin,
        setup_type=setup_type,
        direction=Direction.LONG,
        confidence=confidence,
        key_level=price,
        context=_spine(regime=regime, atr=atr, price=price),
        description="x",
        setup_tf="4h",
        basis_price=price,
        confluence=frozenset(_FACTORS[:n_confluence]),
    )


def _result(decision="CONFIRM", coin="BTC", machine_block=_DEFAULT_MB, **bsig_kwargs) -> ScanResult:
    # default to a textbook `machine_block` so the new textbook-quality gate (Sub-chunk E)
    # is bypassed and existing count-floor / regime / batch / summary tests still exercise
    # their target seams. Pass `machine_block=None` or `machine_block=_mb("variant")` to
    # drive the textbook gate explicitly.
    if machine_block is _DEFAULT_MB:
        machine_block = _mb("textbook")
    return ScanResult(
        scan_id=f"s-{coin}",
        coin=coin,
        timestamp=datetime.now(timezone.utc),
        setup_signal=_bsig(coin=coin, **bsig_kwargs),
        decision=decision,
        confluence_score=None,
        analysis_text=None,
        parsed_decision=None,
        chart_paths=[],
        cost_usd=0.1,
        machine_block=machine_block,
    )


def _cfg(setup_version="v2", v2_min_confluence=2) -> ScannerConfig:
    return ScannerConfig(setup_version=setup_version, v2_min_confluence=v2_min_confluence)


def _run(result, cfg=None):
    out = validate_confirms([result], cfg or _cfg())
    return out[0]


# --- 0. textbook-quality gate (Sub-chunk E, SPEC #33 step 1) -------------------


def test_textbook_confirm_passes_gate():
    # textbook MB (default) + count above floor → CONFIRM survives both gates
    r = _run(_result(n_confluence=2))
    assert r.is_confirm
    assert r.validator_verdict is None


def test_variant_confirm_demoted_non_textbook():
    r = _run(_result(machine_block=_mb("variant"), n_confluence=4))
    assert not r.is_confirm
    assert r.decision == "REJECT"
    assert r.validator_verdict == REJECT_NON_TEXTBOOK


def test_missing_machine_block_demoted_non_textbook():
    # fail-closed: no quality signal ⇒ reject (encodes "the v2 contract is that CONFIRMs
    # must carry a setup_quality"; a missing MB is itself a defect signal)
    r = _run(_result(machine_block=None, n_confluence=4))
    assert not r.is_confirm
    assert r.decision == "REJECT"
    assert r.validator_verdict == REJECT_NON_TEXTBOOK


def test_unknown_setup_quality_demoted_non_textbook():
    # only the literal "textbook" passes — any other value (incl. a typo / future tier)
    # fails closed at the validator regardless of what `from_raw` coerced upstream
    r = _run(_result(machine_block=_mb("excellent"), n_confluence=4))
    assert r.validator_verdict == REJECT_NON_TEXTBOOK


def test_variant_with_low_count_demotes_as_non_textbook_not_floor():
    # the `continue` invariant: textbook gate fires FIRST. If a regression drops the
    # `continue` after the textbook demotion, the count floor would overwrite the verdict
    # to REJECT_CONFLUENCE_FLOOR and break summary attribution. Pins the precedence.
    r = _run(_result(machine_block=_mb("variant"), n_confluence=1))
    assert r.decision == "REJECT"
    assert r.validator_verdict == REJECT_NON_TEXTBOOK
    assert r.validator_verdict != REJECT_CONFLUENCE_FLOOR


# --- 1. confluence-count floor -------------------------------------------------


def test_floor_demotes_below_threshold():
    r = _run(_result(n_confluence=1))  # floor default 2
    assert not r.is_confirm
    assert r.decision == "REJECT"
    assert r.validator_verdict == REJECT_CONFLUENCE_FLOOR


def test_floor_preserves_at_boundary():
    r = _run(_result(n_confluence=2))
    assert r.is_confirm
    assert r.validator_verdict is None


def test_floor_preserves_above_threshold():
    r = _run(_result(n_confluence=4))
    assert r.is_confirm
    assert r.validator_verdict is None


def test_floor_respects_configured_threshold():
    # count=2 passes default(2) but fails a stricter floor of 3
    r = _run(_result(n_confluence=2), _cfg(v2_min_confluence=3))
    assert r.validator_verdict == REJECT_CONFLUENCE_FLOOR


# --- 2. validator trusts typed input (no downstream re-validation) -------------


@pytest.mark.parametrize(
    "setup_type",
    [SetupTypeV2.TREND_CONTINUATION, SetupTypeV2.SFP_RANGE, SetupTypeV2.RSI_DIVERGENCE_RANGE,
     SetupTypeV2.TREND_RECLAIM, SetupTypeV2.DEVIATION_RECLAIM, SetupTypeV2.GAP_FILL],
)
@pytest.mark.parametrize("regime", ["trending", "transitional", "rotational"])
def test_any_setup_passes_any_regime(setup_type, regime):
    # Per arch-pass R1: the validator does NOT re-assert per-setup regime policy (the
    # detector hard-gates it at construction; a within-scan recheck is a tautology).
    r = _run(_result(setup_type=setup_type, n_confluence=2, regime=regime))
    assert r.is_confirm
    assert r.validator_verdict is None


@pytest.mark.parametrize(
    "ctx_kwargs",
    [{"atr": float("nan")}, {"atr": 0.0}, {"price": 0.0}, {"confidence": float("nan")}, {"regime": ""}],
)
def test_degraded_context_is_not_revalidated(ctx_kwargs):
    # Per arch-pass R2: the structural preconditions (finite ATR/price, regime) are
    # enforced at the detector/producer boundary; the validator trusts its typed input
    # and re-checks NONE of them — only the confluence-count floor gates. A degraded
    # context with sufficient confluence is preserved (pins the no-re-validation call).
    r = _run(_result(n_confluence=3, **ctx_kwargs))
    assert r.is_confirm
    assert r.validator_verdict is None


# --- 5. v1 untouched + non-CONFIRM untouched -----------------------------------


def test_v1_path_is_noop_even_for_weak_confirm():
    r = _run(_result(n_confluence=1), _cfg(setup_version="v1"))
    assert r.is_confirm  # would fail the floor on v2, but v1 early-returns
    assert r.validator_verdict is None


def test_v1_path_is_noop_even_for_variant_quality():
    # v1 never reaches the textbook gate either — early return at `setup_version != 'v2'`
    # is byte-identical regardless of `setup_quality` (v1 emits None; this guards against
    # accidentally extending the textbook gate to v1)
    r = _run(_result(machine_block=_mb("variant"), n_confluence=4), _cfg(setup_version="v1"))
    assert r.is_confirm
    assert r.validator_verdict is None


def test_v1_path_is_noop_even_for_missing_machine_block():
    r = _run(_result(machine_block=None, n_confluence=4), _cfg(setup_version="v1"))
    assert r.is_confirm
    assert r.validator_verdict is None


@pytest.mark.parametrize("decision", ["REJECT", "WAIT", None])
def test_non_confirm_results_untouched(decision):
    r = _run(_result(decision=decision, n_confluence=1))  # would fail floor if CONFIRM
    assert r.decision == decision
    assert r.validator_verdict is None


# --- 7. batch + demotion mechanics ---------------------------------------------


def test_batch_demotes_only_failing_confirms():
    good = _result(coin="ETH", n_confluence=3)
    bad = _result(coin="BTC", n_confluence=1)
    rej = _result(coin="SOL", decision="REJECT", n_confluence=1)
    out = validate_confirms([good, bad, rej], _cfg())
    survivors = [r for r in out if r.is_confirm]  # the exact stage3/stage4 filter
    assert [r.coin for r in survivors] == ["ETH"]
    assert bad.validator_verdict == REJECT_CONFLUENCE_FLOOR
    assert good.validator_verdict is None
    assert rej.validator_verdict is None  # LLM reject, never touched


def test_returns_same_list_object():
    results = [_result()]
    assert validate_confirms(results, _cfg()) is results


# --- 8. summary surfaces (partition + v1 byte-identity) ------------------------


def test_summary_partitions_validator_rejects():
    results = [_result(coin="ETH", n_confluence=3), _result(coin="BTC", n_confluence=1)]
    validate_confirms(results, _cfg())
    summary = _build_summary([1, 2], results, [], alerted=0, elapsed=1.0)
    assert summary["confirms"] == 1
    assert summary["rejects"] == 0  # the demotion is NOT counted as an LLM reject
    assert summary["validator_rejects"] == 1


def test_summary_v1_has_no_validator_rejects_key():
    # v1 (and v2-no-demotion) keep the exact pre-8a summary key set
    results = [_result(n_confluence=1)]
    validate_confirms(results, _cfg(setup_version="v1"))  # no-op
    summary = _build_summary([1], results, [], alerted=0, elapsed=1.0)
    assert "validator_rejects" not in summary
    assert summary["confirms"] == 1


def test_print_and_ops_surfaces_show_only_when_nonzero(capsys):
    results = [_result(coin="BTC", n_confluence=1)]
    validate_confirms(results, _cfg())
    summary = _build_summary([1], results, [], alerted=0, elapsed=1.0)
    _print_summary(summary, 1.0)
    assert "Validator-rejected: 1" in capsys.readouterr().out

    # zero case → the token never appears (v1 line byte-identical)
    clean = _build_summary([1], [_result(n_confluence=3)], [], alerted=0, elapsed=1.0)
    _print_summary(clean, 1.0)
    assert "Validator-rejected" not in capsys.readouterr().out


# --- 9. config boundary validation ---------------------------------------------


def test_config_rejects_floor_below_one():
    with pytest.raises(ValueError, match="v2_min_confluence"):
        ScannerConfig(v2_min_confluence=0)
    # 1 is valid (inert), 2 is the active default
    assert ScannerConfig(v2_min_confluence=1).v2_min_confluence == 1
