"""Offline replay / calibration harness for the v10 v2 path (T-019 Chunk 8b).

Pure compute over already-produced signals/results — the data ACQUISITION (fetching candles,
running ``screener_v2.screen_v2`` over a universe, sweeping detector thresholds) is the caller's
job: an operational run, deferred until a candle source is available. This module is the
calibration LOGIC the cutover gate (Acceptance #9 a/b) reads:

- **Emit-distribution + over-emission guard (#9a):** the shape of what the v2 detectors emit and
  a guard that the per-coin emit rate has not blown past the V9.x baseline (the over-emission the
  V9.11 cohort exposed — decision #32).
- **Validator floor sweep (#9b):** runs the *real* ``setup_validator.validate_confirms`` (the
  single owner of the two-step gate — structural textbook predicate then confluence-count floor —
  never re-implemented here) across candidate ``v2_min_confluence`` floors over a labeled set of
  seeded over-eager vs legit confirms, and recommends the smallest floor that demotes every
  over-eager confirm without demoting a legit one. The harness scopes its measurement to the count
  floor by SEEDING structurally-textbook probes (non-``trend_continuation`` types —
  ``seed_labeled_confirms``; T-020.4) so the structural textbook gate passes and the count floor
  stays the seam under test; ``floor_sweep`` fail-fast raises on a non-textbook probe or a missing
  execution block to make external misuse loud. The real-emission ``floor_impact`` measures the
  count-floor bite over the structural-textbook survivors only, reporting the structural-gate
  exclusions separately (the two gates are not conflated).
- **go/no-go record** combining both, surfacing the reasons a run is NOT a go.

NOT in scope here (the data-dependent operational run): the detector-threshold sweep (Stoch RSI
<20/>80, SFP 1-3 bars, F1 ER-drop, compression predicate, trend-continuation cap) — that re-runs
the chunks-1-5 detectors over historical candles with varied params and lands when a candle source
is wired. This harness scores whatever emit/labeled sets that run produces.
"""

import copy
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from .config import ScannerConfig, SetupVersion
from .models import (
    BoundarySignal,
    ConfluenceFactor,
    Direction,
    MachineBlock,
    ScanResult,
    SetupTypeV2,
    SpineContext,
    Structure,
)
from .setup_validator import REJECT_CONFLUENCE_FLOOR, validate_confirms
from .setups_v2 import setup_textbook

__all__ = [
    "EmitDistribution",
    "LabeledConfirm",
    "FloorReport",
    "FloorImpact",
    "CalibrationRecord",
    "emit_distribution",
    "emit_rate",
    "over_emission_ok",
    "floor_sweep",
    "recommend_floor",
    "calibration_record",
    "seed_labeled_confirms",
    "floor_impact",
    "run_calibration",
    "collect_v2_signals",
]


@dataclass(frozen=True)
class LabeledConfirm:
    """A seeded confirm for the floor sweep, with its ground-truth label named (not a positional
    ``(result, bool)`` tuple): ``over_eager`` True = a confirm the floor SHOULD demote."""

    result: ScanResult
    over_eager: bool


# --- emit distribution (Acceptance #9a) ----------------------------------------


@dataclass(frozen=True)
class EmitDistribution:
    """The shape of one v2 screener run's emissions — by setup type and by confluence count
    (the latter shows where a candidate floor bites). ``total`` is the emitted signal count."""

    total: int
    by_setup_type: dict[str, int]
    by_confluence_count: dict[int, int]


def emit_distribution(signals: list[BoundarySignal]) -> EmitDistribution:
    """Tally emissions by ``setup_type.value`` and by confluence-factor count. Pure."""
    by_setup_type: dict[str, int] = {}
    by_confluence_count: dict[int, int] = {}
    for s in signals:
        key = s.setup_type.value
        by_setup_type[key] = by_setup_type.get(key, 0) + 1
        n = len(s.confluence)
        by_confluence_count[n] = by_confluence_count.get(n, 0) + 1
    return EmitDistribution(
        total=len(signals),
        by_setup_type=by_setup_type,
        by_confluence_count=by_confluence_count,
    )


def emit_rate(dist: EmitDistribution, universe_size: int) -> float:
    """Emitted signals per coin scanned. 0.0 for an empty universe (no division surprise)."""
    return dist.total / universe_size if universe_size > 0 else 0.0


def over_emission_ok(
    dist: EmitDistribution,
    universe_size: int,
    baseline_rate: float,
    tolerance: float = 1.5,
) -> bool:
    """The over-emission guard: the v2 emit rate must not exceed the V9.x ``baseline_rate`` by
    more than ``tolerance``x. ``baseline_rate`` (V9.x signals/coin) is a calibration input the
    operational run supplies — this is the comparison, not the baseline."""
    return emit_rate(dist, universe_size) <= baseline_rate * tolerance


# --- validator floor sweep (Acceptance #9b) ------------------------------------


@dataclass(frozen=True)
class FloorReport:
    """One candidate ``v2_min_confluence`` floor scored against a labeled set: how many seeded
    over-eager confirms it demotes vs how many legit confirms it wrongly demotes."""

    floor: int
    over_eager_demoted: int
    over_eager_total: int
    legit_demoted: int
    legit_total: int

    @property
    def catches_all_over_eager(self) -> bool:
        return self.over_eager_total > 0 and self.over_eager_demoted == self.over_eager_total

    @property
    def demotes_no_legit(self) -> bool:
        return self.legit_demoted == 0


def floor_sweep(
    labeled: list[LabeledConfirm],
    floors: list[int],
    base_config: ScannerConfig,
) -> list[FloorReport]:
    """Score each candidate floor by running the REAL ``validate_confirms`` (single owner of the
    floor rule) on a deep copy of the results, so the harness validates the live gate rather than
    a re-implementation. The input is never mutated. ``base_config`` is passed explicitly — the
    harness never constructs config from ambient env (it stays env-independent / pure); ``replace``
    only overrides the two fields the gate reads.

    Precondition: every probe is structurally textbook (`setup_textbook`) AND carries an
    execution block — the sweep measures the count-floor seam, so the upstream gates (the
    structural textbook predicate then the missing-MB fail-closed, SPEC #44 step 1) must pass for
    every input. A non-textbook (or MB-less) probe would demote on `REJECT_NON_TEXTBOOK` and be
    scored as "not demoted" by the count-floor counter — silently mis-scoring the floor. Fail
    fast at the boundary rather than hide the misuse.
    """
    for lc in labeled:
        if lc.result.machine_block is None:
            raise ValueError(
                f"floor_sweep requires probes with an execution block; "
                f"{lc.result.coin} has machine_block=None"
            )
        if not setup_textbook(lc.result.setup_signal):
            raise ValueError(
                f"floor_sweep requires structurally-textbook probes (the count-floor seam); "
                f"{lc.result.coin} ({lc.result.setup_signal.setup_type.value}) is non-textbook"
            )
    reports: list[FloorReport] = []
    for floor in floors:
        cfg = replace(base_config, setup_version=SetupVersion.V2, v2_min_confluence=floor)
        results = [copy.deepcopy(lc.result) for lc in labeled]
        validate_confirms(results, cfg)
        oe_demoted = oe_total = legit_demoted = legit_total = 0
        for lc, res in zip(labeled, results):
            demoted = res.validator_verdict == REJECT_CONFLUENCE_FLOOR
            if lc.over_eager:
                oe_total += 1
                oe_demoted += int(demoted)
            else:
                legit_total += 1
                legit_demoted += int(demoted)
        reports.append(
            FloorReport(floor, oe_demoted, oe_total, legit_demoted, legit_total)
        )
    return reports


def recommend_floor(reports: list[FloorReport]) -> int | None:
    """The smallest floor that demotes EVERY seeded over-eager confirm and ZERO legit confirms.
    ``None`` when no floor separates them — the operator must widen the confluence model rather
    than accept a floor that bleeds (surface the gap, don't guess a compromise floor)."""
    clean = [r for r in reports if r.catches_all_over_eager and r.demotes_no_legit]
    return min((r.floor for r in clean), default=None)


# --- go/no-go record -----------------------------------------------------------


@dataclass(frozen=True)
class CalibrationRecord:
    """The Chunk-9 cutover gate's go/no-go: the emit distribution, the over-emission verdict, the
    recommended floor (or None), the per-floor reports, and the reasons a run is NOT a go."""

    emit: EmitDistribution
    over_emission_ok: bool
    recommended_floor: int | None
    floor_reports: list[FloorReport]
    reasons: list[str]

    @property
    def go(self) -> bool:
        return self.over_emission_ok and self.recommended_floor is not None


def calibration_record(
    signals: list[BoundarySignal],
    universe_size: int,
    baseline_rate: float,
    labeled: list[LabeledConfirm],
    floors: list[int],
    base_config: ScannerConfig,
    tolerance: float = 1.5,
) -> CalibrationRecord:
    """Assemble the go/no-go: over-emission guard (#9a) AND a separating confluence floor (#9b).
    ``go`` requires both; ``reasons`` names each failing gate for the operator."""
    dist = emit_distribution(signals)
    oe_ok = over_emission_ok(dist, universe_size, baseline_rate, tolerance)
    reports = floor_sweep(labeled, floors, base_config)
    floor = recommend_floor(reports)
    reasons: list[str] = []
    if not oe_ok:
        reasons.append(
            f"over-emission: rate {emit_rate(dist, universe_size):.3f} exceeds "
            f"{tolerance}x baseline {baseline_rate:.3f}"
        )
    if floor is None:
        reasons.append(
            "no confluence floor separates seeded over-eager confirms from legit confirms"
        )
    return CalibrationRecord(
        emit=dist,
        over_emission_ok=oe_ok,
        recommended_floor=floor,
        floor_reports=reports,
        reasons=reasons,
    )


# --- operational run (the Chunk-8b RUN — wires the live screener into the harness) ----

# Fixed timestamp for synthetic probe records so seed_labeled_confirms() is deterministic/pure
# (the validator ignores timestamps; a wall-clock `now()` would only break that property).
_SEED_TS = datetime(2020, 1, 1, tzinfo=timezone.utc)


def _confirm_result(signal: BoundarySignal) -> ScanResult:
    """Wrap a `BoundarySignal` as a minimal CONFIRM `ScanResult`. The validator reads
    ``decision`` + ``setup_signal`` (the structural textbook predicate) + ``setup_signal.confluence``
    (the count floor); ``machine_block`` only needs to be non-None (an executable CONFIRM —
    structural textbook is decided from the signal, not the LLM ``setup_quality`` self-label, which
    T-020.4 dropped). The MachineBlock is constructed fresh per probe (not a module-level singleton)
    — the dataclass is frozen but its ``extra`` dict is not, so per-probe construction prevents any
    future mutation from leaking across probes. Fixed timestamp keeps it pure."""
    mb = MachineBlock(
        entry_low=1.0, entry_high=1.0, stop_loss=0.9, tp1=1.1, tp1_exit_pct=100.0,
        direction="LONG", confidence=5.0, leverage=1.0,
    )
    return ScanResult(
        scan_id=f"cal-{signal.coin}-{len(signal.confluence)}",
        coin=signal.coin,
        timestamp=_SEED_TS,
        setup_signal=signal,
        decision="CONFIRM",
        confluence_score=None,
        analysis_text="",
        parsed_decision={},
        chart_paths=[],
        cost_usd=0.0,
        machine_block=mb,
    )


def _synthetic_signal(setup_type: SetupTypeV2, n_confluence: int, coin: str) -> BoundarySignal:
    """A synthetic v2 signal carrying exactly ``n_confluence`` factors — the only property the
    confluence-count floor reads. Context is an inert placeholder."""
    ctx = SpineContext(
        structure=Structure.BULLISH, sr_levels=[], range_bounds=None,
        atr=1.0, current_price=1.0, fib_levels=None, regime="trending",
    )
    return BoundarySignal(
        coin=coin, setup_type=setup_type, direction=Direction.LONG, confidence=0.5,
        key_level=1.0, context=ctx, description="seed", setup_tf="4h",
        confluence=frozenset(list(ConfluenceFactor)[:n_confluence]),
        detected_at=_SEED_TS,  # fixed → seed_labeled_confirms() is deterministic
    )


def seed_labeled_confirms() -> list[LabeledConfirm]:
    """The canonical seeded set the floor is validated against (Acceptance #9b): over-eager confirms
    (confluence 0-1 — the floor SHOULD demote) and legit confirms (confluence 2-4 — the floor should
    keep), spread across setup types. The types are the **non-`trend_continuation`** setups so every
    probe is structurally textbook at any confluence count (T-020.4: the predicate gates only
    `trend_continuation` via RETEST/weakness; the count floor is type-blind, so the spread is
    cosmetic) — this isolates the count-floor seam from the structural gate. Deterministic / pure;
    the smallest separating floor over this set is 2 (= the shipped `v2_min_confluence` default)."""
    types = [t for t in SetupTypeV2 if t is not SetupTypeV2.TREND_CONTINUATION]
    seeds: list[LabeledConfirm] = []
    for i, n in enumerate((0, 1)):
        st = types[i % len(types)]
        seeds.append(LabeledConfirm(_confirm_result(_synthetic_signal(st, n, f"OE{n}")), over_eager=True))
    for i, n in enumerate((2, 3, 4)):
        st = types[i % len(types)]
        seeds.append(LabeledConfirm(_confirm_result(_synthetic_signal(st, n, f"LG{n}")), over_eager=False))
    return seeds


@dataclass(frozen=True)
class FloorImpact:
    """`floor_impact` result under the two-gate ordering (T-020.4). `reports` is the count-floor
    bite measured ONLY over the structural-textbook survivors; `structural_excluded` is how many
    real emissions the structural textbook gate removed upstream — reported separately so the
    count-floor metric is never silently conflated with the structural gate's bite."""

    reports: list[FloorReport]
    structural_textbook: int
    structural_excluded: int


def floor_impact(
    signals: list[BoundarySignal],
    floors: Sequence[int],
    base_config: ScannerConfig,
) -> FloorImpact:
    """How aggressively each candidate floor bites REAL emissions — measured over the
    **structural-textbook survivors** only (T-020.4). The structural textbook gate is the upstream
    filter: a real emission that is structurally non-textbook (e.g. a `trend_continuation` with no
    retest) demotes on `REJECT_NON_TEXTBOOK`, not the count floor, so it is excluded + counted
    separately rather than run through the sweep. Each survivor is wrapped as a CONFIRM labeled
    legit, so each `FloorReport.legit_demoted` is the count of survivors that floor would demote
    (the count-floor bite). Pure — reuses the single-owner `floor_sweep` over the survivors."""
    survivors = [s for s in signals if setup_textbook(s)]
    labeled = [LabeledConfirm(_confirm_result(s), over_eager=False) for s in survivors]
    return FloorImpact(
        reports=floor_sweep(labeled, list(floors), base_config),
        structural_textbook=len(survivors),
        structural_excluded=len(signals) - len(survivors),
    )


def run_calibration(
    signals: list[BoundarySignal],
    universe_size: int,
    baseline_rate: float,
    base_config: ScannerConfig,
    floors: Sequence[int] = (1, 2, 3, 4),
    tolerance: float = 1.5,
) -> CalibrationRecord:
    """The go/no-go over a real v2 emit run: #9a over-emission (real ``signals`` vs ``baseline_rate``)
    + #9b floor recommendation (the canonical seeded set). Pure given the emitted signals."""
    return calibration_record(
        signals, universe_size, baseline_rate,
        labeled=seed_labeled_confirms(), floors=list(floors),
        base_config=base_config, tolerance=tolerance,
    )


def collect_v2_signals(config: ScannerConfig, *, verbose: bool = True) -> tuple[list[BoundarySignal], int]:
    """Impure boundary: resolve the live universe + run the real v2 screener → (emitted signals,
    universe_size). The only part of this module that touches the exchange/DB; function-local imports
    keep the pure-harness import surface light. Needs exchange access (run on the VM)."""
    from .data import fetch_pairs_with_volume
    from .db import get_connection
    from .screener_v2 import screen_v2
    from .strategy.store import get_strategy

    conn = get_connection()
    try:
        strategy = get_strategy("v0", conn)
        disabled = strategy.disabled_types_set if strategy else None
        pairs = fetch_pairs_with_volume(config)
        coins = [p["coin"] for p in pairs]
        leverage_map = {p["coin"]: p.get("max_leverage") for p in pairs}
        signals, _bundle = screen_v2(coins, leverage_map, disabled, config, conn, verbose)
        return signals, len(coins)
    finally:
        conn.close()
