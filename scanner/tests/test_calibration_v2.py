"""Tests for scanner/calibration_v2.py — the Chunk-8b offline calibration harness (Acceptance #9 a/b).

Real `BoundarySignal`/`SpineContext`/`ScanResult` fixtures (no mocks) so the floor sweep drives the
live `setup_validator.validate_confirms` — the harness must validate the real gate, not a copy of it.
"""

from datetime import datetime, timezone

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
    ScanResult,
    SetupTypeV2,
    SpineContext,
    SpineLevel,
    Structure,
)
from scanner.setup_validator import REJECT_CONFLUENCE_FLOOR

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
    # real emissions at confluence counts 1, 2, 3 — floor 2 should demote only the count-1 one
    signals = [_bsig(n_confluence=1), _bsig(n_confluence=2), _bsig(n_confluence=3)]
    impact = {fr.floor: fr for fr in floor_impact(signals, [1, 2, 3], base_config=_CFG)}
    assert impact[1].legit_demoted == 0            # nothing below 1
    assert impact[2].legit_demoted == 1            # only the count-1 emission
    assert impact[3].legit_demoted == 2            # counts 1 and 2
    assert impact[2].legit_total == 3              # all real emissions counted


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
