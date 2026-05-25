"""LLM building blocks — prompts, parsers, validators.

Moved from chart_analyzer/analyzer.py. Pure functions + constants, stdlib-only deps.
"""

import base64
import json
import mimetypes
import re
import sys
from pathlib import Path

from .models import MachineBlock


def load_strategy(path: Path) -> str:
    """Read a ruleset markdown file. The caller (`prompt_select`) resolves the
    per-version path from the single prompt-artifact registry; this is the thin reader
    (fail-fast on a missing file — both ruleset files are shipped)."""
    if not path.exists():
        print(f"Error: Strategy file not found at {path}", file=sys.stderr)
        sys.exit(1)
    return path.read_text()


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

## SETUP-SPECIFIC RULES

**Setup B (breakout): IMMEDIATE-only.** The breakout candle must have
already closed beyond the level. Do not emit B with a future
`candle_close_above/below` trigger. Per ruleset §3.1, pre-close B is
unresolved — wait for the close, then it is either B (held, IMMEDIATE
entry or limit_at retest) or F (failed, enter via F IMMEDIATE on the
failed close — see below).

**Setup F (deviation): IMMEDIATE-only.** The deviation completes when
price closes back within the level. F-IMMEDIATE means that close-back
just printed on the last closed candle (the wick that broke the level
may be from any of the last 5 candles, but the close-back is the
last-close confirmation). Do not emit F with any future trigger
(`candle_close_above/below`, `limit_at`, or `deviation_reclaim`). F is
event-driven: the rejection candle IS the signal, and its alpha decays
as it ages out — a pending entry fires hours later into degraded
geometry at worse R:R. When B is in progress, wait for the failed close
to print, then emit F IMMEDIATE on that close.

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
**LLM Confidence:** [1-10 — calibrate per scale below]
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

## CONFIDENCE SCALE

LLM Confidence calibration — use the spread:
- 1-4: should be REJECT, not CONFIRM
- 5: marginal — leaning REJECT, confirming on benefit-of-doubt
- 6: baseline confirm — all DECISION CRITERIA gates clear, no exceptional confluence
- 7: strong — clean structure + multiple aligned confluences (trend + level + multi-TF)
- 8: textbook — all confluences stacked, no visible weakness
- 9-10: rare conviction-grade

## RULESET

{ruleset}
"""


# Scanner-mode system-prompt suffix appended after SYSTEM_PROMPT.format(ruleset=...).
# Relocated verbatim from analysis.py (T-019 7b.3) so prompt_select owns assembly;
# v1 byte-identical (hash-verified). v2 prose is rewritten in Chunk 8.
SCANNER_MODE_SUFFIX = (
    '\n'
    '\n'
    '## SCANNER MODE\n'
    'Scans run every 4 hours. Put analysis in thinking tokens; the written\n'
    'response is the decision block only.\n'
    '\n'
    '## ENTRY TYPE — pick one\n'
    'Default to CONDITIONAL. Use IMMEDIATE only when the confirming event\n'
    'is already complete on the last closed candle. **Exceptions** (see\n'
    'SETUP-SPECIFIC RULES):\n'
    '- Setup B (breakout): IMMEDIATE on the confirming close OR\n'
    '  `limit_at` on a retest. No `candle_close_above/below` or\n'
    '  `deviation_reclaim`.\n'
    '- Setup F (deviation): IMMEDIATE only on the confirming close. No\n'
    '  future trigger.\n'
    '\n'
    '**CONDITIONAL (default):** Setup is valid; a specific future event\n'
    'must print before entry. The monitor watches 1H closes (or polls\n'
    'spot price for `limit_at`), enters on trigger, kills on invalidation,\n'
    'or expires.\n'
    "- entry_trigger: 'candle_close_above', 'candle_close_below', or\n"
    "  'limit_at'. (For Setup F: none of these — F is IMMEDIATE-only.\n"
    "  For Setup B: 'limit_at' on a retest only; no `candle_close_*`.)\n"
    '- entry_trigger_price: the 1H close level (or limit_at price) that\n'
    '  confirms entry\n'
    '- invalidation_price: kills the setup before the trigger fires.\n'
    '  Place it between stop_loss and entry — closer to entry than\n'
    '  stop_loss, never past it. Pre-entry must be at least as\n'
    '  cautious as the post-fill stop. Common error: setting inv as\n'
    "  a wider buffer than stop. That's backward — inv triggers\n"
    '  FIRST (smaller adverse move).\n'
    '    LONG  entry 100, stop 95  → inv 95-100 valid; inv 90 invalid\n'
    '    SHORT entry 100, stop 105 → inv 100-105 valid; inv 110 invalid\n'
    '- expiry_candles: 1H candles to wait (1-24; match thesis speed)\n'
    '- entry_low/entry_high: bracket entry_trigger_price (±0.2-0.5%);\n'
    '  execution re-brackets at trigger time\n'
    '\n'
    'Examples of CONDITIONAL confirming events:\n'
    '- LONG pullback (C): 1H close above a broken resistance that is now support\n'
    '- SHORT pullback (C): 1H close below a broken support that is now resistance\n'
    '- LONG OTE retrace (D): 1H close inside the 0.618-0.786 zone after bullish MSB\n'
    '- B retest: `limit_at` at the broken level (no `candle_close_*` trigger)\n'
    '\n'
    '**IMMEDIATE (exception):** The confirming event already printed on\n'
    'the last closed candle. Name it in Confirming Event.\n'
    'Examples:\n'
    '- MSB just confirmed on the last 4H close\n'
    '- Deviation reclaim just printed on the last 1H close\n'
    '- Rejection wick off a level with the body closing back on the\n'
    '  working side\n'
    '- Structural break in LTF anchoring a HTF setup\n'
    '\n'
    'Fields:\n'
    '- entry_low/entry_high: bracket current price (within 0.5% of last\n'
    '  candle close)\n'
    '- Omit entry_trigger, entry_trigger_price, invalidation_price,\n'
    '  expiry_candles\n'
    '\n'
    'Price at a level without confirmation yet = CONDITIONAL. Specify\n'
    'the candle close that confirms it. Setup B and Setup F are\n'
    'exceptions — see SETUP-SPECIFIC RULES (above).\n'
    '\n'
    '## INVALIDATION FIELDS (CONFIRM only)\n'
    "- **Expected Hold:** trade duration estimate (e.g. '1-2 days')\n"
    '- **Time Invalidation:** abandon point if neither SL nor TP hits\n'
    '\n'
    '## MACHINE BLOCK (CONFIRM only)\n'
    'Append JSON after the decision block:\n'
    '\n'
    '<!-- MACHINE_BLOCK\n'
    '{"entry_low": 61500.0, "entry_high": 62000.0, '
    '"stop_loss": 59800.0, "tp1": 65000.0, "tp1_exit_pct": 40, '
    '"tp2": 68000.0, "tp2_exit_pct": 40, "direction": "LONG", '
    '"confidence": 7, "leverage": 5, "entry_trigger": "candle_close_above", '
    '"entry_trigger_price": 62000.0, "invalidation_price": 60500.0, '
    '"expiry_candles": 8}\n'
    'MACHINE_BLOCK -->\n'
    '\n'
    'Field rules:\n'
    '- Prices: plain numbers, no $ or commas\n'
    '- tp1_exit_pct: 30-50 (TP1 is the conservative FTA — §6.4)\n'
    '- tp2_exit_pct: 30-50 (when multi-target)\n'
    '- invalidation_price (CONDITIONAL only): hard constraint —\n'
    '    LONG  : invalidation_price >= stop_loss (between stop and entry)\n'
    '    SHORT : invalidation_price <= stop_loss (between entry and stop)\n'
    '  Setting inv past stop in the adverse direction is auto-rejected.\n'
    '- IMMEDIATE: omit the four conditional fields above\n'
    '- Trailing TP: execution trails post-TP1 — provide a concrete TP1\n'
    '  to anchor the trail\n'
    '\n'
    '## CORE RULES (recap)\n'
    '- Default to CONDITIONAL. IMMEDIATE requires a confirming event on\n'
    '  the last closed candle.\n'
    '- **Setup B:** IMMEDIATE on the confirming close OR `limit_at` on\n'
    '  a retest. No `candle_close_above/below` or `deviation_reclaim`.\n'
    '- **Setup F:** IMMEDIATE only on the confirming close. No future\n'
    '  trigger at all.\n'
    '- Name the confirming event in every CONFIRM.\n'
    '- A valid thesis with uncertain timing is CONFIRM-with-trigger, not\n'
    '  REJECT — except for B/F (above).\n'
)


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


def parse_machine_block(text: str) -> MachineBlock | None:
    """Extract the MACHINE_BLOCK JSON from LLM output and normalize it.

    Looks for: <!-- MACHINE_BLOCK\n{json}\nMACHINE_BLOCK -->

    Thin text-extraction boundary only: regex + json.loads, then delegates
    ALL normalization/validation to the single canonical normalizer
    `MachineBlock.from_raw` (T-016 Phase 1a, SPEC #20). Returns the typed
    MachineBlock, or None on exactly the inputs it rejected before (missing
    block, malformed JSON, structural ∪ business-rule reject — from_raw
    reproduces the prior contract; proven by the equivalence battery).
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

    return MachineBlock.from_raw(data)


def validate_execution_params(
    params: MachineBlock, max_leverage: int | None = None,
) -> tuple[bool, list[str]]:
    """Validate machine block params for trade execution.

    Returns (valid, errors). Empty errors list = valid. Typed access (T-016
    C3) — behaviour byte-identical to the prior dict version; the required
    fields are always present so the old .get(...) defaults were dead, the
    optional tp2/tp2_exit_pct/trigger reads preserve absent==None/0 exactly.
    """
    errors: list[str] = []
    direction = params.direction.upper()

    if direction not in ("LONG", "SHORT"):
        errors.append(f"invalid direction: {params.direction}")

    entry_low = params.entry_low
    entry_high = params.entry_high
    stop = params.stop_loss
    tp1 = params.tp1
    confidence = params.confidence
    leverage = params.leverage

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
    tp2 = params.tp2
    if tp2 is not None and tp1 > 0:
        if direction == "LONG" and tp2 <= tp1:
            errors.append(f"LONG tp2 ({tp2}) must be above tp1 ({tp1})")
        elif direction == "SHORT" and tp2 >= tp1:
            errors.append(f"SHORT tp2 ({tp2}) must be below tp1 ({tp1})")

    # Exit percentages — tp1_exit_pct required (always present); tp2_exit_pct
    # optional, absent must read as 0 exactly as the prior .get(...,0) did.
    tp1_pct = params.tp1_exit_pct
    tp2_pct = params.tp2_exit_pct if params.tp2_exit_pct is not None else 0
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
    trigger = params.entry_trigger
    trigger_price = params.entry_trigger_price
    inv_price = params.invalidation_price
    has_any = any(v is not None for v in (trigger, trigger_price, inv_price))
    has_all = all(v is not None for v in (trigger, trigger_price, inv_price))

    if has_any and not has_all:
        errors.append("conditional entry: entry_trigger, entry_trigger_price, and invalidation_price must all be present together")
    elif has_all and direction in ("LONG", "SHORT"):
        # Pre-entry invalidation must be at least as cautious as the post-entry
        # stop: inv triggers on an equal-or-smaller adverse move than stop_loss.
        # LONG: inv_price >= stop_loss (equal, or closer to entry).
        # SHORT: inv_price <= stop_loss (equal, or closer to entry).
        if direction == "LONG" and inv_price < stop:
            errors.append(f"LONG invalidation_price ({inv_price}) must be at or above stop_loss ({stop})")
        if direction == "SHORT" and inv_price > stop:
            errors.append(f"SHORT invalidation_price ({inv_price}) must be at or below stop_loss ({stop})")

    return (len(errors) == 0, errors)
