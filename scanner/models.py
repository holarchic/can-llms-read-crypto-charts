"""Data models for the scanner pipeline."""

import copy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum

from scanner.expectancy import ExpectancyReport


class Structure(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGING = "RANGING"


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class SetupType(Enum):
    A = "A"  # Same-Way Level
    B = "B"  # Breakout/Breakdown
    C = "C"  # S/R Flip Continuation
    D = "D"  # Deeper Retrace
    E = "E"  # Reversal / Failed Continuation
    F = "F"  # Deviation / Failed Breakout
    G = "G"  # Spike / SFP / Stop Run


SETUP_NAMES = {
    SetupType.A: "Same-Way Level",
    SetupType.B: "Breakout/Breakdown",
    SetupType.C: "S/R Flip Continuation",
    SetupType.D: "Deeper Retrace",
    SetupType.E: "Reversal / Failed Continuation",
    SetupType.F: "Deviation / Failed Breakout",
    SetupType.G: "Spike / SFP / Stop Run",
}


@dataclass
class SwingPoint:
    index: int
    timestamp: datetime
    price: float
    type: str  # "high" or "low"
    body_price: float


@dataclass
class SRLevel:
    price: float
    strength: int  # touch count
    source: str  # e.g. "swing_cluster"
    first_seen: datetime | None = None
    last_tested: datetime | None = None
    direction_tests: dict = field(default_factory=lambda: {"from_above": 0, "from_below": 0})
    degraded: bool = False


@dataclass
class SRLevelV2:
    """Top-down S/R *zone* (v10 v2 framework, T-019 Chunk 3 + Sub-chunk F).

    Additive sibling to the v1 ``SRLevel`` — the v1 record (swing-cluster source,
    ``direction_tests``/``degraded``) stays frozen for the live v1 path; this is the
    v2 analog produced by ``scanner.sr_levels`` and consumed by the v2 setup
    detectors (Chunk 5b). Two distinct timestamps by design: ``formed_at`` is when
    the establishing pivot formed (bracket-eligibility reads this); ``last_test`` is
    the most-recent touch across contributors (quality recency reads this). For a
    freshly-found single pivot they are equal. ``confirmations`` is a distinct-TF
    -lineage count (``1d``/``3d`` count once) — deliberately a *different* unit from
    v1 ``SRLevel.strength`` (raw swing touch count), hence the different name.
    ``quality`` carries None pre-``_assign_quality`` (the final stage of
    ``derive_sr_levels``) and a positive float post-score. Surfacing the unset state as
    None lets the chart dispatch boundary reject a pre-score level forwarded by mistake.
    framework-canonical: "a level is a price or cluster of prices" — every level carries
    a ``[zone_low, zone_high]`` envelope (resistance: ``max(open,close)`` → ``high``;
    support: ``low`` → ``min(open,close)``). ``price`` is the *wick extreme* on the
    zone edge — by invariant ``price == zone_high`` for ``pivot_type="high"``,
    ``price == zone_low`` for ``pivot_type="low"`` — so SFP pierce semantics still
    anchor on the wick line. ``retest_count`` is post-formation same-side re-entries
    into the zone (close-based flip); ``quality`` decays by ``0.7^retest_count``
    and ``retest_count > 5`` is dropped entirely.
    """

    price: float
    tf_source: str             # "1mo" | "1w" | "3d" | "1d" — provenance + rank
    pivot_type: str            # "high" | "low"
    confirmations: int         # distinct-TF-lineage confirmations
    vrvp_confluence: bool
    formed_at: datetime
    last_test: datetime
    zone_low: float            # required — no default (fail loud at construction)
    zone_high: float           # required — no default
    retest_count: int = 0      # derived post-hoc, not at pivot construction
    # Quality stays None pre-``_assign_quality`` (the final stage of ``derive_sr_levels``).
    # Surfacing the unset state as None rather than a lossy 0.0 sentinel is what lets the
    # chart dispatch boundary reject a pre-score level forwarded by mistake.
    quality: float | None = None


@dataclass
class MSB:
    index: int
    timestamp: datetime
    type: str  # "bullish" or "bearish"
    level: float
    swing_broken: SwingPoint


@dataclass
class FibLevels:
    swing_low: float
    swing_high: float
    fib_618: float
    fib_786: float
    ote_zone: tuple[float, float]  # (618 level, 786 level) — always (higher, lower)


@dataclass
class MarketContext:
    structure: Structure
    swings: list[SwingPoint]
    sr_levels: list[SRLevel]
    msbs: list[MSB]
    range_bounds: tuple[float, float] | None  # (low, high) if RANGING
    atr: float
    current_price: float
    fib_levels: FibLevels | None
    vol_percentile: float = 0.0
    efficiency_ratio: float = 0.0
    regime: str = ""


@dataclass
class ScoreComponents:
    """Individual scoring factors — stored for calibration experiments."""
    base: float = 0.0
    alignment: float = 0.0
    strength: float = 0.0
    proximity: float = 0.0
    confluence: float = 0.0


@dataclass
class SetupSignal:
    coin: str
    setup_type: SetupType
    direction: Direction
    confidence: float  # 0.0-1.0
    key_level: float
    context: MarketContext
    description: str  # human-readable, used for logging
    max_leverage: int | None = None  # exchange max leverage for this pair
    score_components: ScoreComponents | None = None  # individual factors for calibration
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SetupTypeV2(Enum):
    """v10 the framework setup types (T-019 Chunk 5). Lowercase snake_case — additive sibling
    to the v1 ``SetupType`` (A–G); the two vocabularies coexist during cutover. The
    three rotational/counter-trend members are declared now (produced in Chunk 5b) so
    the enum is stable across the 5a/5b split."""

    TREND_CONTINUATION = "trend_continuation"
    TREND_RECLAIM = "trend_reclaim"
    DEVIATION_RECLAIM = "deviation_reclaim"
    GAP_FILL = "gap_fill"
    SFP_RANGE = "sfp_range"
    RSI_DIVERGENCE_RANGE = "rsi_divergence_range"


# v2 (v10) display names — extend the shared SETUP_NAMES map now that
# SetupTypeV2 exists, so the alert/store renderers resolve both vocabularies.
SETUP_NAMES.update({
    SetupTypeV2.TREND_CONTINUATION: "Trend Continuation",
    SetupTypeV2.TREND_RECLAIM: "Trend Reclaim",
    SetupTypeV2.DEVIATION_RECLAIM: "Deviation Reclaim",
    SetupTypeV2.GAP_FILL: "Gap Fill",
    SetupTypeV2.SFP_RANGE: "SFP / Range Sweep",
    SetupTypeV2.RSI_DIVERGENCE_RANGE: "RSI Divergence (Range)",
})


class ConfluenceFactor(Enum):
    """Named confluence factors a v2 detector attaches to a signal (extensible — 5b
    adds its own). ``SetupSignalV2.confluence_count`` is the floor the Chunk-8a
    ``setup_validator`` reads."""

    AT_SR = "at_sr"
    TREND_ALIGNED = "trend_aligned"
    RETEST = "retest"
    VERTICAL_ACCUMULATION = "vertical_accumulation"
    LOWER_TF_CONFIRMED = "lower_tf_confirmed"
    # Chunk 5b — rotational / counter-trend factors:
    DIVERGENCE = "divergence"
    COUNTER_TREND = "counter_trend"
    SWEEP_RECLAIM = "sweep_reclaim"
    OVERSOLD_OVERBOUGHT = "oversold_overbought"
    GAPFILL_STALL = "gapfill_stall"


class SoftStopMode(Enum):
    """How a signal's soft stop is evaluated downstream."""

    TREND_RECLAIM_AGAINST = "trend_reclaim_against"  # the dominant trend resumes against the position
    LEVEL_RECLAIM_AGAINST = "level_reclaim_against"  # price reclaims back through the basis level (range setups)


@dataclass
class SoftStop:
    """The two-level-invalidation soft stop: a *rule* (mode) plus a current reference
    price, so downstream exit logic re-evaluates the rule live rather than freezing a
    level. ``TREND_RECLAIM_AGAINST`` ends the trade when the local trend reclaims
    against the position (trend setups; ``reference`` is the EMA36 at detection).
    ``LEVEL_RECLAIM_AGAINST`` ends it when price reclaims back through the basis level
    (range setups; ``reference`` is the swept level / divergence basis)."""

    mode: SoftStopMode
    reference: float


@dataclass
class SetupSignalV2:
    """A v10 the framework setup signal (T-019 Chunk 5). Additive sibling to the v1
    ``SetupSignal``; produced by ``scanner.setups_v2`` detectors — pure (no LLM/DB).
    ``basis_price`` is the setup's reference level — trend-derived (EMA36) for the
    trend detectors, or level-derived (the swept level / divergence body) for the
    range detectors. ``hard_stop`` is the price-level stop and ``soft_stop`` the
    reclaim-against rule (trend- or level-reclaim per its mode); whichever hits first
    ends the trade. No ``setup_quality`` — that is the LLM's additive judgment
    (Decision #32), set downstream, never by a detector."""

    setup_type: SetupTypeV2
    direction: Direction
    basis_price: float
    entry: float
    hard_stop: float
    soft_stop: SoftStop
    confluence: frozenset[ConfluenceFactor]
    weakness_flag: bool

    @property
    def confluence_count(self) -> int:
        return len(self.confluence)


@dataclass
class SpineLevel:
    """The S/R level the stage-2→4 spine carries — the chart-consumer payload,
    version-neutral so the v1 `SRLevel` and the v2 `SRLevelV2` both project onto it
    (the nested-S/R neutralization deferred from 7b.2a, arch-pass R2). v2 has no
    degradation model — a top-down `SRLevelV2` is high-quality by construction — so it
    maps to `degraded=False` (rendered solid/green). Sub-chunk F adds optional zone +
    quality + pivot_type fields so the v2 chart renderer (`_sr_zones`) can draw shaded
    zones with quality-keyed alpha and pivot-type-keyed colour; v1 leaves them None
    and the chart consumer falls back to line rendering. Sub-chunk G adds the FIFTH
    optional v2-only field ``formed_at`` so the renderer can draw the zone
    forward-only from its formation bar (a Rectangle from `searchsorted(formed_at)`
    to chart-right). v1 leaves it None; ``_has_zoned_levels`` rejects a zoned payload
    with ``formed_at=None`` at the dispatch boundary so a producer bug can't silently
    fall open to chart-left back-dating."""

    price: float
    degraded: bool = False
    zone_low: float | None = None
    zone_high: float | None = None
    quality: float | None = None
    pivot_type: str = ""
    formed_at: datetime | None = None

    @classmethod
    def from_sr_level(cls, lvl: SRLevel) -> "SpineLevel":
        return cls(price=lvl.price, degraded=lvl.degraded)

    @classmethod
    def from_sr_level_v2(cls, lvl: SRLevelV2) -> "SpineLevel":
        return cls(
            price=lvl.price, degraded=False,
            zone_low=lvl.zone_low, zone_high=lvl.zone_high,
            quality=lvl.quality, pivot_type=lvl.pivot_type,
            formed_at=lvl.formed_at,
        )


@dataclass
class SpineContext:
    """The structural context the stage-2→4 spine consumes — exactly the
    `MarketContext` fields read downstream of stage 1, version-neutral end to end. The
    top-level shape is producer-neutral (the spine holds no whole v1 `MarketContext`)
    and `sr_levels` is a `SpineLevel` list — the chart-consumer payload onto which
    both the v1 `SRLevel` and the v2 `SRLevelV2` project. Per Sub-chunk F, `SpineLevel`
    additionally carries optional zone/quality/pivot_type so the v2 zoned-render path
    works through the same field; v1 leaves them None and falls back to line rendering.
    `fib_levels` and `vol_percentile` are `| None` — v2 has no fib swing / volume-
    percentile source, so it carries None rather than a fabricated `0.0` (an unknown is
    NULL, not an observed zeroth percentile). Field names match the v1 `MarketContext`
    subset, so consumers' `signal.context.X` access is unchanged across the cutover
    (swings/msbs dropped — no stage-2→4 consumer reads them)."""

    structure: Structure
    sr_levels: list[SpineLevel]
    range_bounds: tuple[float, float] | None
    atr: float
    current_price: float
    fib_levels: FibLevels | None
    vol_percentile: float | None = 0.0
    efficiency_ratio: float = 0.0
    regime: str = ""

    @classmethod
    def from_market_context(cls, ctx: MarketContext) -> "SpineContext":
        """Project a v1 `MarketContext` onto the spine context (the v1 mapping)."""
        return cls(
            structure=ctx.structure,
            sr_levels=[SpineLevel.from_sr_level(lvl) for lvl in ctx.sr_levels],
            range_bounds=ctx.range_bounds,
            atr=ctx.atr,
            current_price=ctx.current_price,
            fib_levels=ctx.fib_levels,
            vol_percentile=ctx.vol_percentile,
            efficiency_ratio=ctx.efficiency_ratio,
            regime=ctx.regime,
        )


@dataclass
class BoundarySignal:
    """The v1/v2 stage-2→4 boundary record (T-019 Chunk 7b, Option C). Both the v1
    adapter (`from_setup_signal`) and the v2 producer (Chunk 7b.2b) target this one
    type; `context` is the `SpineContext` projection rather than a whole v1
    `MarketContext` (its nested S/R payload is the version-neutral `SpineLevel`).
    The v1 fields mirror `SetupSignal`'s names so consumers are unchanged; the v2-only
    execution/confluence fields are empty on the v1 path and populated by the v2
    producer. `setup_tf` is the originating timeframe, supplied explicitly at the
    lift/producer site (single owner)."""

    coin: str
    setup_type: SetupType | SetupTypeV2
    direction: Direction
    confidence: float
    key_level: float
    context: SpineContext
    description: str
    setup_tf: str
    max_leverage: int | None = None
    score_components: ScoreComponents | None = None
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # v2-only — empty on the v1 path, populated by the v2 producer (Chunk 7b.2b)
    basis_price: float | None = None
    entry: float | None = None
    hard_stop: float | None = None
    soft_stop: SoftStop | None = None
    confluence: frozenset[ConfluenceFactor] = frozenset()
    weakness_flag: bool = False

    @classmethod
    def from_setup_signal(cls, sig: SetupSignal, setup_tf: str) -> "BoundarySignal":
        """Lift a v1 `SetupSignal` into the boundary record (the Option-C v1 adapter):
        v1 fields copy across, `context` projects to `SpineContext`, v2-only fields stay
        empty. `setup_tf` is supplied by the caller — the lift site owns that decision."""
        return cls(
            coin=sig.coin,
            setup_type=sig.setup_type,
            direction=sig.direction,
            confidence=sig.confidence,
            key_level=sig.key_level,
            context=SpineContext.from_market_context(sig.context),
            description=sig.description,
            setup_tf=setup_tf,
            max_leverage=sig.max_leverage,
            score_components=sig.score_components,
            detected_at=sig.detected_at,
        )


_MB_REQUIRED_NUMERIC = (
    "entry_low", "entry_high", "stop_loss", "tp1", "tp1_exit_pct",
    "confidence", "leverage",
)
_MB_VALID_TRIGGERS = ("candle_close_above", "candle_close_below", "limit_at")
# Setup-type vocabularies, derived from the enums (single owner — no hardcode):
# v1 UPPERCASE A–G, v2 lowercase snake_case. The union is the from_raw valid-set.
_V1_SETUP_TYPES = frozenset(t.value for t in SetupType)
_V2_SETUP_TYPES = frozenset(t.value for t in SetupTypeV2)
_VALID_SETUP_TYPES = _V1_SETUP_TYPES | _V2_SETUP_TYPES


def normalize_setup_type(raw: str) -> str:
    """Canonical case for a ``setup_type`` token — the single source of truth for
    setup_type casing during the v1/v2 cutover: v1 (``A``–``G``) uppercase, v2
    (snake_case) lowercase. Pure case-canonicalization; does NOT validate
    membership (``MachineBlock.from_raw`` does that against ``_VALID_SETUP_TYPES``)."""
    s = raw.strip()
    return s.upper() if s.upper() in _V1_SETUP_TYPES else s.lower()


# The full modeled key set — anything outside this goes to `extra` (passthrough).
_MB_MODELED = (
    "entry_low", "entry_high", "stop_loss", "tp1", "tp1_exit_pct",
    "direction", "confidence", "leverage",
    "tp2", "tp2_exit_pct",
    "entry_trigger", "entry_trigger_price", "invalidation_price", "expiry_candles",
    "setup_type", "confluence_score", "confirming_event", "setup_quality",
)


@dataclass(frozen=True)
class MachineBlock:
    """Typed, single-owner representation of the LLM execution machine_block.

    Parsed once via `from_raw` (the sole canonical normalizer — it reproduces
    `scanner.llm.parse_machine_block`'s entire rejection contract, including the
    final business-rule `validate_execution_params` gate). `to_raw` is its
    semantic inverse over modeled + `extra` keys (T-016 Phase 1a, SPEC #20).
    Behaviour byte-identical to the prior raw-dict path; persistence JSON shape
    unchanged. `extra` carries unmodeled/future keys so round-tripping cannot
    silently drop research fields across prompt-version evolution.
    """

    entry_low: float
    entry_high: float
    stop_loss: float
    tp1: float
    tp1_exit_pct: float
    direction: str
    confidence: float
    leverage: float
    tp2: float | None = None
    tp2_exit_pct: float | None = None
    entry_trigger: str | None = None
    entry_trigger_price: float | None = None
    invalidation_price: float | None = None
    expiry_candles: int | None = None
    setup_type: str | None = None
    confluence_score: int | None = None
    confirming_event: str | None = None
    # v2 (the framework) LLM primary judgment — "textbook" only on a CONFIRM (SPEC #35).
    # Additive/optional: v1 never emits it. The post-LLM ``setup_validator`` demotes any v2 CONFIRM
    # where this is not "textbook" (or where ``machine_block is None``) with ``REJECT_NON_TEXTBOOK``
    # before the confluence-count floor fires (SPEC #33 step 1). Parser is lenient — ``from_raw``
    # normalizes / drops an invalid value to None rather than rejecting the whole block (a
    # downgrade-to-None still demotes at the validator); strict-reject stays for the capital-path
    # fields above.
    setup_quality: str | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_raw(cls, d: dict) -> "MachineBlock | None":
        """Normalize a raw machine_block dict, or None on exactly the inputs
        `parse_machine_block` rejects today (structural ∪ business-rule).

        Mirrors `scanner.llm.parse_machine_block`'s pre-C3 step sequence. Operates
        on a copy; never mutates the caller's dict.
        """
        if not isinstance(d, dict):
            return None
        data = copy.deepcopy(d)

        required = {
            "entry_low": (int, float),
            "entry_high": (int, float),
            "stop_loss": (int, float),
            "tp1": (int, float),
            "tp1_exit_pct": (int, float),
            "direction": str,
            "confidence": (int, float),
            "leverage": (int, float),
        }
        for fld, types in required.items():
            if fld not in data:
                return None
            if not isinstance(data[fld], types):
                return None
        # bool is an int subclass — reject it for numeric fields (parity guard:
        # parse_machine_block's isinstance(x,(int,float)) also admits bool, so
        # we deliberately do NOT add a stricter check here; behaviour identical).
        for fld in _MB_REQUIRED_NUMERIC:
            data[fld] = float(data[fld])

        if "tp2" in data:
            if not isinstance(data["tp2"], (int, float)):
                return None
            data["tp2"] = float(data["tp2"])
        if "tp2_exit_pct" in data:
            if not isinstance(data["tp2_exit_pct"], (int, float)):
                return None
            data["tp2_exit_pct"] = float(data["tp2_exit_pct"])

        if "entry_trigger" in data:
            if data["entry_trigger"] not in _MB_VALID_TRIGGERS:
                return None
        for opt_num in ("entry_trigger_price", "invalidation_price"):
            if opt_num in data:
                if not isinstance(data[opt_num], (int, float)):
                    return None
                data[opt_num] = float(data[opt_num])
        if "expiry_candles" in data:
            if not isinstance(data["expiry_candles"], (int, float)):
                return None
            data["expiry_candles"] = min(int(data["expiry_candles"]), 24)

        if "setup_type" in data:
            if not isinstance(data["setup_type"], str):
                return None
            st = normalize_setup_type(data["setup_type"])
            if st not in _VALID_SETUP_TYPES:
                return None
            data["setup_type"] = st
        if "confluence_score" in data:
            if not isinstance(data["confluence_score"], (int, float)):
                return None
            score = int(data["confluence_score"])
            if not 0 <= score <= 15:
                return None
            data["confluence_score"] = score
        if "confirming_event" in data:
            if not isinstance(data["confirming_event"], str):
                return None
            ev = data["confirming_event"].strip()
            if not 10 <= len(ev) <= 200:
                return None
            data["confirming_event"] = ev
        if "setup_quality" in data:
            # Lenient at the parser; strict at the validator (SPEC #33/#35). A typo or
            # unrecognised value must not downgrade an otherwise-valid CONFIRM to
            # ``machine_block=None`` here — drop the value (neither modeled nor extra) and let
            # the post-LLM ``setup_validator`` demote on ``REJECT_NON_TEXTBOOK`` instead. "variant"
            # still normalizes lowercase (the LLM may emit it as a regression / hallucination;
            # the validator demotes it deterministically). The raw analysis_text is persisted for
            # forensics. Strict-reject stays for the capital-path fields above.
            sq = data["setup_quality"]
            if isinstance(sq, str) and sq.strip().lower() in ("textbook", "variant"):
                data["setup_quality"] = sq.strip().lower()
            else:
                del data["setup_quality"]

        modeled = {k: data[k] for k in _MB_MODELED if k in data}
        extra = {k: v for k, v in data.items() if k not in _MB_MODELED}
        candidate = cls(**modeled, extra=extra)

        # Final business-rule gate — identical to the pre-C3 parse_machine_block tail.
        # validate_execution_params is now typed (T-016 C3), so it runs on the
        # constructed candidate. Function-level import: models.py stays
        # app-dependency-free at module load (llm.py imports models.py); this
        # matches executor.py:64's idiom. Behaviour-identical to the prior
        # validate(coerced_dict) — same field values, typed access.
        from scanner.llm import validate_execution_params

        valid, _errors = validate_execution_params(candidate)
        if not valid:
            return None
        return candidate

    def to_raw(self) -> dict:
        """Semantic inverse of `from_raw`: the dict `parse_machine_block`
        produces today (json.loads-equal). Required keys always; optionals
        only when present (None == absent); plus `extra` verbatim. Fresh dict.
        """
        out: dict = {
            "entry_low": self.entry_low,
            "entry_high": self.entry_high,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1,
            "tp1_exit_pct": self.tp1_exit_pct,
            "direction": self.direction,
            "confidence": self.confidence,
            "leverage": self.leverage,
        }
        for fld in (
            "tp2", "tp2_exit_pct", "entry_trigger", "entry_trigger_price",
            "invalidation_price", "expiry_candles", "setup_type",
            "confluence_score", "confirming_event", "setup_quality",
        ):
            v = getattr(self, fld)
            if v is not None:
                out[fld] = v
        out.update(copy.deepcopy(self.extra))
        return out

    def reanchored(self, trigger_close: float) -> "MachineBlock":
        """Pure entry-trigger reanchor — the conditional-entry reanchor used
        by entry_monitor._build_execution_request. Arithmetic is byte-identical
        to the pre-Phase-1a in-place mutation (frozen as the equivalence
        baseline in test_machine_block._old_reanchor_inplace): delta from the
        ORIGINAL entry-mid, shift SL/TP, then re-bracket the entry zone to
        trigger_close ± 0.5%.
        """
        orig_mid = (
            (self.entry_low + self.entry_high) / 2
            if self.entry_low
            else trigger_close
        )
        delta = trigger_close - orig_mid
        buffer = trigger_close * 0.005
        return replace(
            self,
            stop_loss=self.stop_loss + delta,
            tp1=self.tp1 + delta,
            tp2=(self.tp2 + delta if self.tp2 is not None else None),
            entry_low=trigger_close - buffer,
            entry_high=trigger_close + buffer,
            # frozen=True does not deep-freeze `extra`; dataclasses.replace
            # would otherwise alias the same dict into the new block. Deep-copy
            # so each MachineBlock genuinely owns its `extra` (upholds the
            # SPEC #20 single-owner/immutability contract — no shared mutable
            # side channel between the original and the reanchored block).
            extra=copy.deepcopy(self.extra),
        )


@dataclass
class ScanResult:
    scan_id: str
    coin: str
    timestamp: datetime
    setup_signal: BoundarySignal
    decision: str | None  # CONFIRM/REJECT/WAIT
    confluence_score: str | None
    analysis_text: str | None
    parsed_decision: dict | None
    chart_paths: list[str]
    cost_usd: float
    alerted: bool = False
    llm_confidence: int | None = None
    machine_block: "MachineBlock | None" = None
    detected_at: datetime | None = None
    prompt_version: str | None = None  # the version-gated prompt that produced this result (7b.3)
    # In-memory only (NOT persisted) — set by the post-LLM `setup_validator` when it
    # demotes a v2 CONFIRM (Chunk 8a, SPEC #33). None ⇒ not evaluated / passed. The
    # demoted `decision` becomes `"REJECT"`; this tag carries the why so an LLM REJECT
    # and a validator demotion stay distinguishable (and partition the scan summary).
    validator_verdict: str | None = None

    @property
    def is_confirm(self) -> bool:
        return self.decision == "CONFIRM"

    @property
    def is_reject(self) -> bool:
        return self.decision == "REJECT"

    @property
    def is_wait(self) -> bool:
        return self.decision == "WAIT"


@dataclass(frozen=True)
class ShadowOutcome:
    """Pure result of walking ONE shadow signal forward (T-020.1, SPEC #41).

    `status` is a terminal state or `OPEN`. `realized_r` is net of fees + slippage and
    is non-None ONLY when the signal became a trade (`status == "RESOLVED"`); the
    non-trade statuses (`TRIGGER_EXPIRED`/`NO_FILL`/`INVALIDATED`/`DATA_INCOMPLETE`) and
    `OPEN` carry `realized_r is None` so they are excluded from the expectancy sample.
    """

    status: str
    entry_fill: float | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    realized_r: float | None = None
    fees_r: float | None = None
    slippage_r: float | None = None
    bars_held: int | None = None
    low_fidelity: bool = False


@dataclass(frozen=True)
class ExecutionRequest:
    """Right-sized typed contract for `execute_trade` (T-016 Phase 1b, SPEC #21).

    The executor formerly demanded an analysis-shaped `ScanResult` (heavy
    `SetupSignal`/`MarketContext`) only to read these seven scalars, forcing
    conditional-entry triggers to fabricate a synthetic `ScanResult`+empty
    `SetupSignal`. Each caller now builds this directly. `screener_direction`
    is the raw screener side for counter-trend detection (None ⇒ no
    counter-trend, matching the prior `setup_signal is None` path).

    `staleness_ref` is the single timestamp the executor's staleness guard
    measures age against — it is NOT necessarily "detection time". Each
    caller resolves what that reference is: the scan path passes
    `confirm.detected_at or confirm.timestamp` (the prior in-executor
    fallback, now collapsed at the build site); the conditional-entry
    trigger path passes trigger-fire `now()`. Naming it for the invariant
    (staleness reference) rather than one caller's source keeps the contract
    discoverable from the type.
    """

    scan_id: str
    coin: str
    machine_block: "MachineBlock | None"
    screener_direction: str | None
    atr: float
    max_leverage: int | None
    staleness_ref: datetime


@dataclass(frozen=True)
class TradeRecord:
    """One closed trade, built by the storage-owned read model (T-016 Phase 2a,
    SPEC #22). Frozen: constructed once in `execution/storage.get_closed_trades`,
    read-only downstream (compute/report).

    `realized_pnl` is dollar PnL **net of taker fees** (exchange convention —
    realized PnL is reported after fees; T-020.2, SPEC #42), so `realized_r`
    derived from it is net too, and every aggregate (win/loss partition,
    profit_factor, total_pnl, drawdown) inherits the same net basis coherently.
    `fees_r` is the round-trip taker fee that was subtracted, expressed in R, for
    visibility — it is NOT a further subtraction. `fees_r` is a RAW pre-`r_mult`
    diagnostic (same basis as `slippage_r`): on counter-trend trades (`r_mult<1`)
    the *weighted* `realized_r` does not move by exactly `fees_r`."""

    scan_id: str
    coin: str
    direction: str
    entry_fill: float
    exit_fill: float
    exit_type: str  # SL, TP1, TP2, TIMEOUT, EMERGENCY, DRIFT
    realized_pnl: float  # NET of taker fees (SPEC #42)
    realized_r: float
    risk_amount: float
    holding_hours: float
    funding_paid: float
    slippage_r: float
    fees_r: float  # round-trip taker fee in R, raw pre-r_mult diagnostic (SPEC #42)
    execution_mode: str
    # Explicit single-owner of the "no/degenerate original SL" classification
    # (T-016 Phase 3, SPEC #26 c; KL#9 c). `usable_risk ≡ risk_amount > 0` by
    # construction — covers both `original_stop_price is None` and degenerate
    # `entry==SL` (stop_distance==0). Replaces the implicit `risk_amount==0`
    # sentinel re-derived across analytics/report sites.
    usable_risk: bool
    exchange: str = "hyperliquid"
    # Provenance joined from `scans.prompt_version` (T-020.3, SPEC #43) — the
    # per-version cohort partition key. `None` for legacy rows with no/NULL
    # `scans.prompt_version`; such trades form the `(unversioned)` cohort.
    prompt_version: str | None = None


@dataclass(frozen=True)
class MetricsWindow:
    """Aggregate performance over a closed-trade window (T-016 Phase 2a, SPEC
    #22). Replaces the untyped metrics dict crossing the analytics→report seam.
    `fill_rate` is a placeholder (always 0.0; CONFIRM-context wiring is out of
    Phase-2a scope). `profit_factor` may be `float('inf')`."""

    count: int
    risked_count: int  # trades with a usable risk anchor (SPEC #26 b; KL#9 b)
    win_rate: float
    expectancy_r: float
    profit_factor: float
    total_pnl: float
    max_drawdown_pct: float
    avg_holding_hours: float
    avg_slippage_r: float
    avg_fees_r: float  # mean round-trip taker fee in R over risked trades (SPEC #42)
    total_funding_r: float
    fill_rate: float
    data_source: str
    quality_warnings: tuple[str, ...]


@dataclass(frozen=True)
class RiskedOutcomes:
    """The single-owner risk+outcome partition (T-016 Phase 3, SPEC #26 b;
    KL#9 b). Named typed seam — `risked`/`wins`/`losses` are all
    `tuple[TradeRecord, ...]` so callers identify them by field, never by
    positional slot (a wins/losses swap on a bare 3-tuple of same-typed lists
    would be invisible). Immutable: built once in `analytics.risked_outcomes`,
    read-only downstream (metrics / report)."""

    risked: tuple[TradeRecord, ...]
    wins: tuple[TradeRecord, ...]
    losses: tuple[TradeRecord, ...]


@dataclass(frozen=True)
class VersionCohort:
    """One prompt-version's closed-trade cohort + its per-trade-R expectancy
    (T-020.3, SPEC #43). `prompt_version is None` is the `(unversioned)` cohort
    (legacy rows). `report` is the shared `scanner.expectancy.ExpectancyReport`
    (n / win% / avg R / PF / bootstrap CI / Acceptance-#8 gate), net of fees (#42)."""

    prompt_version: str | None
    report: ExpectancyReport
