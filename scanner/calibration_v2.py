"""Offline replay / calibration harness for the v10 v2 path (T-019 Chunk 8b).

Pure compute over already-produced signals/results — the data ACQUISITION (fetching candles,
running ``screener_v2.screen_v2`` over a universe, sweeping detector thresholds) is the caller's
job: an operational run, deferred until a candle source is available. This module is the
calibration LOGIC the cutover gate (Acceptance #9 a/b) reads:

- **Emit-distribution + over-emission guard (#9a):** the shape of what the v2 detectors emit and
  a guard that the per-coin emit rate has not blown past the V9.x baseline (the over-emission the
  V9.11 cohort exposed — decision #32).
- **Validator floor sweep (#9b):** runs the *real* ``setup_validator.validate_confirms`` (the
  single owner of the confluence-count floor — never re-implemented here) across candidate
  ``v2_min_confluence`` floors over a labeled set of seeded over-eager vs legit confirms, and
  recommends the smallest floor that demotes every over-eager confirm without demoting a legit one.
- **go/no-go record** combining both, surfacing the reasons a run is NOT a go.

NOT in scope here (the data-dependent operational run): the detector-threshold sweep (Stoch RSI
<20/>80, SFP 1-3 bars, F1 ER-drop, compression predicate, trend-continuation cap) — that re-runs
the chunks-1-5 detectors over historical candles with varied params and lands when a candle source
is wired. This harness scores whatever emit/labeled sets that run produces.
"""

import copy
from dataclasses import dataclass, replace

from .config import ScannerConfig
from .models import BoundarySignal, ScanResult
from .setup_validator import REJECT_CONFLUENCE_FLOOR, validate_confirms

__all__ = [
    "EmitDistribution",
    "LabeledConfirm",
    "FloorReport",
    "CalibrationRecord",
    "emit_distribution",
    "emit_rate",
    "over_emission_ok",
    "floor_sweep",
    "recommend_floor",
    "calibration_record",
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
    only overrides the two fields the gate reads."""
    reports: list[FloorReport] = []
    for floor in floors:
        cfg = replace(base_config, setup_version="v2", v2_min_confluence=floor)
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
