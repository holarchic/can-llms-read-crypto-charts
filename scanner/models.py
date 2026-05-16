"""Data models for the scanner pipeline."""

import copy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum


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


_MB_REQUIRED_NUMERIC = (
    "entry_low", "entry_high", "stop_loss", "tp1", "tp1_exit_pct",
    "confidence", "leverage",
)
_MB_VALID_TRIGGERS = ("candle_close_above", "candle_close_below", "limit_at")
_MB_VALID_SETUP_TYPES = ("A", "B", "C", "D", "E", "F", "G")
# The full modeled key set — anything outside this goes to `extra` (passthrough).
_MB_MODELED = (
    "entry_low", "entry_high", "stop_loss", "tp1", "tp1_exit_pct",
    "direction", "confidence", "leverage",
    "tp2", "tp2_exit_pct",
    "entry_trigger", "entry_trigger_price", "invalidation_price", "expiry_candles",
    "setup_type", "confluence_score", "confirming_event",
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
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_raw(cls, d: dict) -> "MachineBlock | None":
        """Normalize a raw machine_block dict, or None on exactly the inputs
        `parse_machine_block` rejects today (structural ∪ business-rule).

        Mirrors scanner/llm.py:205-289 step-for-step. Operates on a copy;
        never mutates the caller's dict.
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
            st = data["setup_type"].strip().upper()
            if st not in _MB_VALID_SETUP_TYPES:
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

        modeled = {k: data[k] for k in _MB_MODELED if k in data}
        extra = {k: v for k, v in data.items() if k not in _MB_MODELED}
        candidate = cls(**modeled, extra=extra)

        # Final business-rule gate — identical to parse_machine_block:287-289.
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
            "confluence_score", "confirming_event",
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
    setup_signal: SetupSignal
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


@dataclass(frozen=True)
class ExecutionRequest:
    """Right-sized typed contract for `execute_trade` (T-016 Phase 1b, SPEC #21).

    The executor formerly demanded an analysis-shaped `ScanResult` (heavy
    `SetupSignal`/`MarketContext`) only to read these seven scalars, forcing
    conditional-entry triggers to fabricate a synthetic `ScanResult`+empty
    `SetupSignal`. Each caller now builds this directly. `screener_direction`
    is the raw screener side for counter-trend detection (None ⇒ no
    counter-trend, matching the prior `setup_signal is None` path);
    `detected_at` is the already-resolved staleness reference (the prior
    `result.detected_at or result.timestamp`, collapsed at the build site).
    """

    scan_id: str
    coin: str
    machine_block: "MachineBlock | None"
    screener_direction: str | None
    atr: float
    max_leverage: int | None
    detected_at: datetime
