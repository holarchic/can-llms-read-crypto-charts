"""Tests for scanner/calibration_v2.py — the Chunk-8b offline calibration harness (Acceptance #9 a/b).

Real `BoundarySignal`/`SpineContext`/`ScanResult` fixtures (no mocks) so the floor sweep drives the
live `setup_validator.validate_confirms` — the harness must validate the real gate, not a copy of it.
Probes seed structurally-textbook signals (non-trend_continuation types) post-T-020.4 (SPEC #44) so
the count-floor sweep stays the seam under test; `floor_impact` measures the count-floor bite over
the structural-textbook survivors only.
"""

from datetime import datetime, timezone

import pytest

from scanner.calibration_v2 import (
    LabeledConfirm,
    calibration_record,
    collect_v2_signals,
    emit_distribution,
    emit_rate,
    floor_impact,
    floor_sweep,
    over_emission_ok,
    recommend_floor,
    run_calibration,
    seed_labeled_confirms,
)
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
from scanner.setup_validator import REJECT_CONFLUENCE_FLOOR, REJECT_NON_TEXTBOOK, validate_confirms
from scanner.setups_v2 import setup_textbook

_FACTORS = list(ConfluenceFactor)  # 10 members → confluence counts 0..10
_CFG = ScannerConfig()  # base config passed explicitly (the harness never reads env itself)


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


def _bsig(setup_type=SetupTypeV2.SFP_RANGE, n_confluence=2, coin="BTC") -> BoundarySignal:
    # Default SFP_RANGE is structurally textbook at any confluence count (the predicate gates only
    # trend_continuation), so the count-floor tests isolate their seam. Pass TREND_CONTINUATION to
    # drive the structural gate.
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


def _mb() -> MachineBlock:
    """Minimal `MachineBlock` mirroring the calibration probe — every harness probe carries one so
    it is an executable CONFIRM (non-None MB). `setup_quality` is inert post-T-020.4 (the structural
    predicate is the gate); left at its None default, as the v2 prompt now emits."""
    return MachineBlock(
        entry_low=1.0, entry_high=1.0, stop_loss=0.9, tp1=1.1, tp1_exit_pct=100.0,
        direction="LONG", confidence=5.0, leverage=1.0,
    )


def _result(decision="CONFIRM", n_confluence=2, setup_type=SetupTypeV2.SFP_RANGE,
            coin="BTC") -> ScanResult:
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
        machine_block=_mb(),
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
    over_eager = LabeledConfirm(_result(n_confluence=1, coin="OE"), over_eager=True)
    legit = LabeledConfirm(_result(n_confluence=3, coin="LG"), over_eager=False)
    return [over_eager, legit]


def test_floor_sweep_drives_real_validator():
    reports = {r.floor: r for r in floor_sweep(_labeled(), floors=[1, 2, 3, 4], base_config=_CFG)}
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
    floor_sweep(labeled, floors=[2, 4], base_config=_CFG)
    # the originals are untouched — still CONFIRM, no validator verdict (deep-copied internally)
    for lc in labeled:
        assert lc.result.decision == "CONFIRM"
        assert lc.result.validator_verdict is None


def test_floor_sweep_demotion_carries_verdict():
    # anti-vacuity: a demotion is the REAL validator's verdict, not a re-implementation
    labeled = [LabeledConfirm(_result(n_confluence=1), over_eager=True)]
    [report] = floor_sweep(labeled, floors=[2], base_config=_CFG)
    assert report.over_eager_demoted == 1
    # confirm the live constant is what the validator stamps (sweep counts that, not a private rule)
    assert REJECT_CONFLUENCE_FLOOR == "REJECT_CONFLUENCE_FLOOR"


def test_recommend_floor_smallest_separating():
    reports = floor_sweep(_labeled(), floors=[1, 2, 3, 4], base_config=_CFG)
    assert recommend_floor(reports) == 2  # smallest that catches all over-eager, demotes no legit


def test_recommend_floor_none_when_inseparable():
    # over-eager and legit both at confluence 2 → no floor separates them
    labeled = [
        LabeledConfirm(_result(n_confluence=2, coin="OE"), over_eager=True),
        LabeledConfirm(_result(n_confluence=2, coin="LG"), over_eager=False),
    ]
    assert recommend_floor(floor_sweep(labeled, floors=[1, 2, 3, 4], base_config=_CFG)) is None


# --- go/no-go record -----------------------------------------------------------


def test_calibration_record_go():
    signals = [_bsig() for _ in range(10)]  # rate 0.10 over 100
    rec = calibration_record(
        signals, universe_size=100, baseline_rate=0.20,
        labeled=_labeled(), floors=[1, 2, 3, 4], base_config=_CFG,
    )
    assert rec.go is True
    assert rec.over_emission_ok is True
    assert rec.recommended_floor == 2
    assert rec.reasons == []


def test_calibration_record_nogo_over_emission():
    signals = [_bsig() for _ in range(40)]  # rate 0.40 over 100, baseline 0.20 → over
    rec = calibration_record(
        signals, universe_size=100, baseline_rate=0.20,
        labeled=_labeled(), floors=[1, 2, 3, 4], base_config=_CFG,
    )
    assert rec.go is False
    assert any("over-emission" in r for r in rec.reasons)


def test_calibration_record_nogo_no_separating_floor():
    labeled = [
        LabeledConfirm(_result(n_confluence=2, coin="OE"), over_eager=True),
        LabeledConfirm(_result(n_confluence=2, coin="LG"), over_eager=False),
    ]
    rec = calibration_record(
        [_bsig()], universe_size=100, baseline_rate=0.20,
        labeled=labeled, floors=[1, 2, 3, 4], base_config=_CFG,
    )
    assert rec.go is False
    assert rec.recommended_floor is None
    assert any("no confluence floor separates" in r for r in rec.reasons)


# --- operational run: seed / floor_impact / run_calibration / collect ----------


def test_seed_labeled_confirms_separates_at_floor_2():
    seed = seed_labeled_confirms()
    over = [lc for lc in seed if lc.over_eager]
    legit = [lc for lc in seed if not lc.over_eager]
    # over-eager confluence 0-1, legit 2-4 — confirm the labels match the counts
    assert {len(lc.result.setup_signal.confluence) for lc in over} == {0, 1}
    assert {len(lc.result.setup_signal.confluence) for lc in legit} == {2, 3, 4}
    # the canonical seed's smallest separating floor is 2 (= the shipped default)
    assert recommend_floor(floor_sweep(seed, [1, 2, 3, 4], base_config=_CFG)) == 2
    # deterministic/pure — two calls build equal objects (fixed timestamps, no wall clock)
    assert seed_labeled_confirms() == seed_labeled_confirms()


def test_floor_impact_counts_real_emission_demotions():
    # real emissions at confluence counts 1, 2, 3 (all SFP_RANGE → structurally textbook) — floor 2
    # should demote only the count-1 one. floor_impact returns a FloorImpact (T-020.4).
    signals = [_bsig(n_confluence=1), _bsig(n_confluence=2), _bsig(n_confluence=3)]
    result = floor_impact(signals, [1, 2, 3], base_config=_CFG)
    assert result.structural_textbook == 3 and result.structural_excluded == 0  # none excluded
    impact = {fr.floor: fr for fr in result.reports}
    assert impact[1].legit_demoted == 0            # nothing below 1
    assert impact[2].legit_demoted == 1            # only the count-1 survivor
    assert impact[3].legit_demoted == 2            # counts 1 and 2
    assert impact[2].legit_total == 3              # all survivors counted


def test_floor_impact_excludes_structurally_non_textbook():
    # a real trend_continuation without RETEST is structurally non-textbook → excluded from the
    # count-floor sweep (it would demote on REJECT_NON_TEXTBOOK, not the count floor), counted
    # separately. The SFP_RANGE emission is a survivor.
    tc = _bsig(SetupTypeV2.TREND_CONTINUATION, n_confluence=2)  # {AT_SR, TREND_ALIGNED}, no RETEST
    sfp = _bsig(SetupTypeV2.SFP_RANGE, n_confluence=2)
    result = floor_impact([tc, sfp], [2], base_config=_CFG)
    assert result.structural_textbook == 1
    assert result.structural_excluded == 1
    assert result.reports[0].legit_total == 1  # only the survivor is swept


def test_run_calibration_go_over_real_signals():
    signals = [_bsig() for _ in range(10)]  # rate 0.10 over 100, baseline 0.20 → OK
    rec = run_calibration(signals, universe_size=100, baseline_rate=0.20, base_config=_CFG)
    assert rec.go is True
    assert rec.recommended_floor == 2  # from the canonical seed


def test_run_calibration_nogo_on_over_emission():
    signals = [_bsig() for _ in range(40)]  # rate 0.40 over 100, baseline 0.20 → over
    rec = run_calibration(signals, universe_size=100, baseline_rate=0.20, base_config=_CFG)
    assert rec.go is False
    assert any("over-emission" in r for r in rec.reasons)


# --- structural textbook-gate contract (T-020.4 / SPEC #44 step 1) --------------

def test_seeded_probes_are_structurally_textbook():
    # T-020.4 contract: every calibration probe must be STRUCTURALLY textbook (so floor_sweep's
    # precondition holds) AND carry an execution block. The seeds use non-trend_continuation
    # types to stay textbook at any confluence count — isolating the count-floor seam. If the
    # seeds regressed to trend_continuation, every low-confluence probe would demote on
    # REJECT_NON_TEXTBOOK and the recommended floor would go None / wrong silently.
    for lc in seed_labeled_confirms():
        assert lc.result.machine_block is not None, f"probe {lc.result.coin} missing machine_block"
        assert setup_textbook(lc.result.setup_signal), f"probe {lc.result.coin} not textbook"
        assert lc.result.setup_signal.setup_type is not SetupTypeV2.TREND_CONTINUATION


def test_recommend_floor_still_returns_2_under_two_step_gate():
    # the canonical seed (over-eager confluence 0-1, legit 2-4) must still resolve to floor=2 once
    # the structural textbook gate fires first — pins the seeded contract against accidental probe
    # drift that would route every probe to REJECT_NON_TEXTBOOK.
    reports = floor_sweep(seed_labeled_confirms(), [1, 2, 3, 4], base_config=_CFG)
    assert recommend_floor(reports) == 2
    # the live run that the harness scores: each over-eager probe must demote on the
    # count floor (not the structural one); the legit probes survive
    for r in reports:
        if r.floor == 2:
            assert r.over_eager_demoted == r.over_eager_total
            assert r.legit_demoted == 0


def test_floor_sweep_rejects_non_textbook_probes():
    # floor_sweep's structural-textbook precondition is explicit (T-020.4) — a non-textbook probe
    # (a trend_continuation with no RETEST) would be scored "not demoted by the count floor" while
    # actually demoting on REJECT_NON_TEXTBOOK, silently mis-scoring. The boundary raises instead.
    bad = LabeledConfirm(
        _result(n_confluence=2, setup_type=SetupTypeV2.TREND_CONTINUATION), over_eager=False,
    )
    with pytest.raises(ValueError, match="structurally-textbook"):
        floor_sweep([bad], [2], base_config=_CFG)


def test_floor_sweep_rejects_missing_machine_block():
    probe = _result(n_confluence=3)
    probe.machine_block = None  # simulate the upstream parse-failure path
    bad = LabeledConfirm(probe, over_eager=False)
    with pytest.raises(ValueError, match="execution block"):
        floor_sweep([bad], [2], base_config=_CFG)


def test_structurally_textbook_probe_routes_to_count_floor():
    # anti-vacuity: a structurally-textbook probe (SFP_RANGE, the _result default) with count<floor
    # demotes on REJECT_CONFLUENCE_FLOOR (the seam the harness measures), NOT REJECT_NON_TEXTBOOK.
    # `_CFG` defaults v1 (no-op); floor_sweep `replace`s to v2 — mirror that here.
    cfg_v2 = ScannerConfig(setup_version="v2", v2_min_confluence=2)
    probe = _result(n_confluence=1)  # structurally textbook + count below floor 2
    validate_confirms([probe], cfg_v2)
    assert probe.validator_verdict == REJECT_CONFLUENCE_FLOOR
    assert probe.validator_verdict != REJECT_NON_TEXTBOOK  # explicit anti-flip pin


def test_collect_v2_signals_wires_screener(monkeypatch):
    # patch the function-local imports at their SOURCE modules (deferred-import mock rule)
    import types as _types

    closed = {"v": False}

    class _Conn:
        def close(self):
            closed["v"] = True

    monkeypatch.setattr("scanner.db.get_connection", lambda: _Conn())
    monkeypatch.setattr("scanner.strategy.store.get_strategy",
                        lambda sid, conn: _types.SimpleNamespace(disabled_types_set=set()))
    monkeypatch.setattr("scanner.data.fetch_pairs_with_volume",
                        lambda config: [{"coin": "BTC", "max_leverage": 10},
                                        {"coin": "ETH", "max_leverage": 20}])
    sentinel = [_bsig(coin="BTC")]
    monkeypatch.setattr("scanner.screener_v2.screen_v2",
                        lambda coins, lev, disabled, config, conn, verbose: (sentinel, {}))

    signals, universe = collect_v2_signals(_CFG, verbose=False)
    assert signals is sentinel
    assert universe == 2          # two coins resolved
    assert closed["v"] is True    # conn closed in finally
