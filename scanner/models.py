"""Data models for the scanner pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
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
