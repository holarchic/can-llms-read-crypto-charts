"""LLM building blocks — prompts, parsers, validators.

Moved from chart_analyzer/analyzer.py. Pure functions + constants, stdlib-only deps.
"""

import base64
import json
import mimetypes
import re
import sys
from pathlib import Path

from .config import STRATEGY_PATH


def load_strategy() -> str:
    if not STRATEGY_PATH.exists():
        print(f"Error: Strategy file not found at {STRATEGY_PATH}", file=sys.stderr)
        sys.exit(1)
    return STRATEGY_PATH.read_text()


def encode_image(path: str) -> tuple[str, str]:
    """Read and base64-encode an image file. Returns (base64_data, media_type)."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        print(f"Error: Image not found at {p}", file=sys.stderr)
        sys.exit(1)

    media_type, _ = mimetypes.guess_type(str(p))
    if media_type not in ("image/png", "image/jpeg", "image/gif", "image/webp"):
        print(f"Error: Unsupported image type '{media_type}'. Use PNG, JPEG, GIF, or WebP.", file=sys.stderr)
        sys.exit(1)

    data = base64.standard_b64encode(p.read_bytes()).decode("utf-8")
    return data, media_type


SYSTEM_PROMPT = """You are a price action chart analyst.

Identify whether a valid trade setup exists on the provided chart(s)
(typically 1D, 4H, 1H) and define its execution plan.

## SCOPE

Identify the setup. Name what confirms entry. Specify prices. A mechanical
execution layer handles the rest:

- Trigger gate — entries wait for the confirmation you specify (1H close
  above/below a level). Setups that never confirm expire without loss.
- Drift — adverse positions move to breakeven at 4H; stagnant ones force-close.
- Post-TP1 trail — after TP1 fills, stop trails 4H swing points.
- Counter-trend scaling — opposing the screener direction applies 0.5x
  risk and 3:1 min R:R automatically.

Offload entry timing to the trigger. When the thesis is valid but timing
is uncertain, CONFIRM with a trigger — do not REJECT.

## DECISION CRITERIA

CONFIRM when all hold:
- Defined setup (A-G per ruleset §3.1)
- Stop and take-profit derivable from chart structure
- Confluence present (trend alignment, fresh level, or multi-TF overlap —
  prefer score ≥ 5 per §3.2)
- You can name the confirming event (printed on last closed candle for
  IMMEDIATE, or a specific future candle close for CONDITIONAL)

REJECT when any hold:
- No defined setup
- Stop or invalidation not articulable
- Confluence weak (score < 5)
- Confirming event cannot be named

## OUTPUT FORMAT

**For REJECT:**

////////////////////////////////////////////////////////////
## DECISION: REJECT
////////////////////////////////////////////////////////////

**Reasoning:** [one sentence]

////////////////////////////////////////////////////////////

**For CONFIRM:**

////////////////////////////////////////////////////////////
## DECISION: CONFIRM
////////////////////////////////////////////////////////////

**Direction:** [LONG | SHORT]
**Setup:** [A-G + short name, e.g. "C (SR flip continuation)"]
**LLM Confidence:** [1-10]
**Reasoning:** [1-2 sentences — include the confirming event]

\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

**Entry Type:** [IMMEDIATE | CONDITIONAL]
**Confirming Event:** [IMMEDIATE: what printed on the last closed candle.
  CONDITIONAL: the specific future event that confirms entry.]
**Entry Zone:** [price range]
**Stop Loss:** [exact price]
**Take Profit:** [exact price or prices]
**Leverage:** [recommended, e.g. "5x"]

\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

////////////////////////////////////////////////////////////

## RULESET

{ruleset}
"""


def parse_decision_block(text: str) -> dict:
    """Parse the DECISION block from analysis output into structured fields."""
    result = {}

    m = re.search(r"##\s*DECISION:\s*(CONFIRM|REJECT|WAIT)\b", text, re.IGNORECASE)
    if m:
        result["decision"] = m.group(1).upper()

    field_patterns = {
        "asset": r"\*\*Asset:\*\*\s*(.+)",
        "timeframe": r"\*\*Timeframe:\*\*\s*(.+)",
        "market_structure": r"\*\*Market Structure:\*\*\s*(.+)",
        "setup": r"\*\*Setup:\*\*\s*(.+)",
        "confluence_score": r"\*\*Confluence Score:\*\*\s*(.+)",
        "direction": r"\*\*Direction:\*\*\s*(.+)",
        "entry_trigger": r"\*\*Entry Trigger:\*\*\s*(.+)",
        "entry_zone": r"\*\*Entry Zone:\*\*\s*(.+)",
        "stop_loss": r"\*\*Stop Loss:\*\*\s*(.+)",
        "take_profit": r"\*\*Take Profit:\*\*\s*(.+)",
        "invalidation": r"\*\*Invalidation:\*\*\s*(.+)",
        "target": r"\*\*Target:\*\*\s*(.+)",
        "risk_guidance": r"\*\*Risk Guidance:\*\*\s*(.+)",
        "leverage": r"\*\*Leverage:\*\*\s*(.+)",
        "reasoning": r"\*\*Reasoning:\*\*\s*(.+)",
        "what_would_change": r"\*\*What Would Change This:\*\*\s*(.+)",
        "expected_hold": r"\*\*Expected Hold:\*\*\s*(.+)",
        "time_invalidation": r"\*\*Time Invalidation:\*\*\s*(.+)",
        "llm_confidence": r"\*\*LLM Confidence:\*\*\s*(.+)",
    }

    for key, pattern in field_patterns.items():
        m = re.search(pattern, text)
        if m:
            result[key] = m.group(1).strip()

    return result


def parse_machine_block(text: str) -> dict | None:
    """Extract and parse the MACHINE_BLOCK JSON from LLM output.

    Looks for: <!-- MACHINE_BLOCK\n{json}\nMACHINE_BLOCK -->
    Returns parsed dict or None if missing/malformed.
    """
    m = re.search(
        r"<!--\s*MACHINE_BLOCK\s*\n(.+?)\nMACHINE_BLOCK\s*-->",
        text,
        re.DOTALL,
    )
    if not m:
        return None

    try:
        data = json.loads(m.group(1).strip())
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    # Required fields and their expected types
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

    for field, types in required.items():
        if field not in data:
            return None
        if not isinstance(data[field], types):
            return None

    # Coerce numeric fields to float
    for field in ("entry_low", "entry_high", "stop_loss", "tp1", "tp1_exit_pct",
                  "confidence", "leverage"):
        data[field] = float(data[field])

    # Coerce optional tp2 fields
    if "tp2" in data:
        if not isinstance(data["tp2"], (int, float)):
            return None
        data["tp2"] = float(data["tp2"])
    if "tp2_exit_pct" in data:
        if not isinstance(data["tp2_exit_pct"], (int, float)):
            return None
        data["tp2_exit_pct"] = float(data["tp2_exit_pct"])

    # Coerce optional conditional entry fields
    _VALID_TRIGGERS = (
        "candle_close_above", "candle_close_below",
        "limit_at", "deviation_reclaim",
    )
    if "entry_trigger" in data:
        if data["entry_trigger"] not in _VALID_TRIGGERS:
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

    # V9.1 observability fields (graceful — missing accepted, invalid rejected).
    # TODO(2026-04-26): review LLM compliance rate for these fields; if > 95%
    # populated, flip to required in V9.2 so CONFIRM decisions without them
    # are rejected rather than logged.
    _VALID_SETUP_TYPES = ("A", "B", "C", "D", "E", "F", "G")
    if "setup_type" in data:
        if not isinstance(data["setup_type"], str):
            return None
        st = data["setup_type"].strip().upper()
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

    # Business-rule validation — reject blocks with bad geometry/ranges
    valid, _errors = validate_execution_params(data)
    if not valid:
        return None

    return data


def validate_execution_params(
    params: dict, max_leverage: int | None = None,
) -> tuple[bool, list[str]]:
    """Validate machine block params for trade execution.

    Returns (valid, errors). Empty errors list = valid.
    """
    errors: list[str] = []
    direction = params.get("direction", "").upper()

    if direction not in ("LONG", "SHORT"):
        errors.append(f"invalid direction: {params.get('direction')}")

    entry_low = params.get("entry_low", 0)
    entry_high = params.get("entry_high", 0)
    stop = params.get("stop_loss", 0)
    tp1 = params.get("tp1", 0)
    confidence = params.get("confidence", 0)
    leverage = params.get("leverage", 0)

    # Price sanity
    for name, val in [("entry_low", entry_low), ("entry_high", entry_high),
                      ("stop_loss", stop), ("tp1", tp1)]:
        if val <= 0:
            errors.append(f"{name} must be positive: {val}")

    if entry_low > entry_high:
        errors.append(f"entry_low ({entry_low}) > entry_high ({entry_high})")

    # Direction-specific price checks
    if direction == "LONG" and stop > 0 and entry_low > 0:
        if stop >= entry_low:
            errors.append(f"LONG stop ({stop}) must be below entry_low ({entry_low})")
        if tp1 > 0 and tp1 <= entry_high:
            errors.append(f"LONG tp1 ({tp1}) must be above entry_high ({entry_high})")
    elif direction == "SHORT" and stop > 0 and entry_high > 0:
        if stop <= entry_high:
            errors.append(f"SHORT stop ({stop}) must be above entry_high ({entry_high})")
        if tp1 > 0 and tp1 >= entry_low:
            errors.append(f"SHORT tp1 ({tp1}) must be below entry_low ({entry_low})")

    # TP2 ordering
    tp2 = params.get("tp2")
    if tp2 is not None and tp1 > 0:
        if direction == "LONG" and tp2 <= tp1:
            errors.append(f"LONG tp2 ({tp2}) must be above tp1 ({tp1})")
        elif direction == "SHORT" and tp2 >= tp1:
            errors.append(f"SHORT tp2 ({tp2}) must be below tp1 ({tp1})")

    # Exit percentages
    tp1_pct = params.get("tp1_exit_pct", 0)
    tp2_pct = params.get("tp2_exit_pct", 0)
    if tp1_pct is not None and tp1_pct <= 0:
        errors.append(f"tp1_exit_pct must be positive: {tp1_pct}")
    if tp2_pct < 0:
        errors.append(f"tp2_exit_pct must be non-negative: {tp2_pct}")
    if tp1_pct + tp2_pct > 100:
        errors.append(f"exit pcts sum ({tp1_pct} + {tp2_pct} = {tp1_pct + tp2_pct}) > 100")

    # Confidence range
    if not (1 <= confidence <= 10):
        errors.append(f"confidence ({confidence}) must be 1-10")

    # Leverage bounds
    if leverage <= 0:
        errors.append(f"leverage must be positive: {leverage}")
    if max_leverage is not None and leverage > max_leverage:
        errors.append(f"leverage ({leverage}) exceeds max ({max_leverage})")

    # Conditional entry cross-validation (all-or-none for trigger fields)
    trigger = params.get("entry_trigger")
    trigger_price = params.get("entry_trigger_price")
    inv_price = params.get("invalidation_price")
    has_any = any(v is not None for v in (trigger, trigger_price, inv_price))
    has_all = all(v is not None for v in (trigger, trigger_price, inv_price))

    if has_any and not has_all:
        errors.append("conditional entry: entry_trigger, entry_trigger_price, and invalidation_price must all be present together")
    elif has_all and direction in ("LONG", "SHORT"):
        # Allow inv_price == stop_loss (unified pre-fill/post-fill kill level).
        # Only reject when inv is a LOOSER gate than stop (closer to entry).
        if direction == "LONG" and inv_price > stop:
            errors.append(f"LONG invalidation_price ({inv_price}) must be at or below stop_loss ({stop})")
        if direction == "SHORT" and inv_price < stop:
            errors.append(f"SHORT invalidation_price ({inv_price}) must be at or above stop_loss ({stop})")

    return (len(errors) == 0, errors)
