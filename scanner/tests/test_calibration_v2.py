"""Tests for scanner/calibration_v2.py — the Chunk-8b offline calibration harness (Acceptance #9 a/b).

Real `BoundarySignal`/`SpineContext`/`ScanResult` fixtures (no mocks) so the floor sweep drives the
live `setup_validator.validate_confirms` — the harness must validate the real gate, not a copy of it.
"""

from datetime import datetime, timezone

from scanner.calibration_v2 import (
    calibration_record,
    emit_distribution,
    emit_rate,
    floor_sweep,
    over_emission_ok,
    recommend_floor,
)
from scanner.models import (
    BoundarySignal,
    ConfluenceFactor,
    Direction,
    ScanResult,
    SetupTypeV2,
    SpineContext,
    SpineLevel,
    Structure,
)
from scanner.setup_validator import REJECT_CONFLUENCE_FLOOR

_FACTORS = list(ConfluenceFactor)  # 10 members → confluence counts 0..10


# --- builders ------------------------------------------------------------------


def _spine(price=100.0) -> SpineContext:
    return SpineContext(
        structure=Structure.BULLISH,
        sr_levels=[SpineLevel(price=price)],
        range_bounds=None,
        atr=2.0,
        current_price=price,
        fib_levels=None,
        regime="trending",
    )


def _bsig(setup_type=SetupTypeV2.TREND_CONTINUATION, n_confluence=2, coin="BTC") -> BoundarySignal:
    return BoundarySignal(
        coin=coin,
        setup_type=setup_type,
        direction=Direction.LONG,
        confidence=0.5,
        key_level=100.0,
        context=_spine(),
        description="x",
        setup_tf="4h",
        basis_price=100.0,
        confluence=frozenset(_FACTORS[:n_confluence]),
    )


def _result(decision="CONFIRM", n_confluence=2, setup_type=SetupTypeV2.TREND_CONTINUATION, coin="BTC") -> ScanResult:
    return ScanResult(
        scan_id=f"s-{coin}-{n_confluence}",
        coin=coin,
        timestamp=datetime.now(timezone.utc),
        setup_signal=_bsig(setup_type=setup_type, n_confluence=n_confluence, coin=coin),
        decision=decision,
        confluence_score=None,
        analysis_text="",
        parsed_decision={},
        chart_paths=[],
        cost_usd=0.0,
        prompt_version="10.00",
    )


# --- emit distribution (#9a) ---------------------------------------------------


def test_emit_distribution_counts_by_type_and_confluence():
    signals = [
        _bsig(SetupTypeV2.TREND_CONTINUATION, n_confluence=2),
        _bsig(SetupTypeV2.TREND_CONTINUATION, n_confluence=3),
        _bsig(SetupTypeV2.SFP_RANGE, n_confluence=2),
    ]
    dist = emit_distribution(signals)
    assert dist.total == 3
    assert dist.by_setup_type == {"trend_continuation": 2, "sfp_range": 1}
    assert dist.by_confluence_count == {2: 2, 3: 1}


def test_emit_distribution_empty():
    dist = emit_distribution([])
    assert dist.total == 0 and dist.by_setup_type == {} and dist.by_confluence_count == {}


def test_emit_rate_and_empty_universe():
    dist = emit_distribution([_bsig(), _bsig(), _bsig()])
    assert emit_rate(dist, 60) == 3 / 60
    assert emit_rate(dist, 0) == 0.0  # no division surprise


def test_over_emission_guard():
    dist = emit_distribution([_bsig() for _ in range(30)])  # rate 0.30 over 100 coins
    # baseline 0.20, tolerance 1.5 → ceiling 0.30 → at the boundary is OK (<=)
    assert over_emission_ok(dist, 100, baseline_rate=0.20, tolerance=1.5) is True
    # tighter tolerance 1.0 → ceiling 0.20 → 0.30 over → NOT ok
    assert over_emission_ok(dist, 100, baseline_rate=0.20, tolerance=1.0) is False


# --- validator floor sweep (#9b) -----------------------------------------------


def _labeled():
    # over-eager: a CONFIRM with a single confluence factor; legit: a CONFIRM with three.
    over_eager = (_result(n_confluence=1, coin="OE"), True)
    legit = (_result(n_confluence=3, coin="LG"), False)
    return [over_eager, legit]


def test_floor_sweep_drives_real_validator():
    reports = {r.floor: r for r in floor_sweep(_labeled(), floors=[1, 2, 3, 4])}
    # floor 1: nothing below 1 → neither demoted
    assert reports[1].over_eager_demoted == 0 and reports[1].legit_demoted == 0
    # floor 2: over-eager (1<2) demoted, legit (3>=2) kept
    assert reports[2].over_eager_demoted == 1 and reports[2].legit_demoted == 0
    assert reports[2].catches_all_over_eager and reports[2].demotes_no_legit
    # floor 4: both below 4 → both demoted (legit bleed)
    assert reports[4].over_eager_demoted == 1 and reports[4].legit_demoted == 1
    assert not reports[4].demotes_no_legit


def test_floor_sweep_does_not_mutate_input():
    labeled = _labeled()
    floor_sweep(labeled, floors=[2, 4])
    # the originals are untouched — still CONFIRM, no validator verdict (deep-copied internally)
    for res, _ in labeled:
        assert res.decision == "CONFIRM"
        assert res.validator_verdict is None


def test_floor_sweep_demotion_carries_verdict():
    # anti-vacuity: a demotion is the REAL validator's verdict, not a re-implementation
    [report] = floor_sweep([(_result(n_confluence=1), True)], floors=[2])
    assert report.over_eager_demoted == 1
    # confirm the live constant is what the validator stamps (sweep counts that, not a private rule)
    assert REJECT_CONFLUENCE_FLOOR == "REJECT_CONFLUENCE_FLOOR"


def test_recommend_floor_smallest_separating():
    reports = floor_sweep(_labeled(), floors=[1, 2, 3, 4])
    assert recommend_floor(reports) == 2  # smallest that catches all over-eager, demotes no legit


def test_recommend_floor_none_when_inseparable():
    # over-eager and legit both at confluence 2 → no floor separates them
    labeled = [(_result(n_confluence=2, coin="OE"), True), (_result(n_confluence=2, coin="LG"), False)]
    assert recommend_floor(floor_sweep(labeled, floors=[1, 2, 3, 4])) is None


# --- go/no-go record -----------------------------------------------------------


def test_calibration_record_go():
    signals = [_bsig() for _ in range(10)]  # rate 0.10 over 100
    rec = calibration_record(
        signals, universe_size=100, baseline_rate=0.20,
        labeled=_labeled(), floors=[1, 2, 3, 4],
    )
    assert rec.go is True
    assert rec.over_emission_ok is True
    assert rec.recommended_floor == 2
    assert rec.reasons == []


def test_calibration_record_nogo_over_emission():
    signals = [_bsig() for _ in range(40)]  # rate 0.40 over 100, baseline 0.20 → over
    rec = calibration_record(
        signals, universe_size=100, baseline_rate=0.20,
        labeled=_labeled(), floors=[1, 2, 3, 4],
    )
    assert rec.go is False
    assert any("over-emission" in r for r in rec.reasons)


def test_calibration_record_nogo_no_separating_floor():
    labeled = [(_result(n_confluence=2, coin="OE"), True), (_result(n_confluence=2, coin="LG"), False)]
    rec = calibration_record(
        [_bsig()], universe_size=100, baseline_rate=0.20,
        labeled=labeled, floors=[1, 2, 3, 4],
    )
    assert rec.go is False
    assert rec.recommended_floor is None
    assert any("no confluence floor separates" in r for r in rec.reasons)
