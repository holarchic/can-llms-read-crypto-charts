"""LLM deep analysis — orchestrates chart analysis via Claude vision API."""

PROMPT_VERSION = "9.01"

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import anthropic

log = logging.getLogger(__name__)

from .config import ScannerConfig
from .llm import (
    SYSTEM_PROMPT,
    encode_image,
    load_strategy,
    parse_decision_block,
    parse_machine_block,
)
from .models import MarketContext, ScanResult, SetupSignal


def _build_btc_context(btc_ctx: MarketContext) -> str:
    """Build multi-line BTC context block for LLM prompt."""
    structure = btc_ctx.structure.value.lower()
    price = btc_ctx.current_price

    # Position within range (if ranging)
    position = ""
    if btc_ctx.range_bounds:
        lo, hi = btc_ctx.range_bounds
        if hi > lo:
            pct = (price - lo) / (hi - lo) * 100
            if pct < 30:
                position = f"Near range lows ({lo:,.0f}–{hi:,.0f})."
            elif pct > 70:
                position = f"Near range highs ({lo:,.0f}–{hi:,.0f})."
            else:
                position = f"Mid-range ({lo:,.0f}–{hi:,.0f})."

    # Last swing low for reference
    recent_lows = [s for s in btc_ctx.swings if s.type == "low"]
    last_low = f"Last swing low: ${recent_lows[-1].body_price:,.0f}." if recent_lows else ""

    lines = [
        "Market context — BTC:",
        f"- Daily structure: {structure.capitalize()}. {position} {last_low}".strip(),
        f"- Volatility: ATR percentile {btc_ctx.vol_percentile:.0f}%.",
    ]
    return "\n".join(lines)


def _build_scanner_content(
    chart_paths: list[Path],
    signal: SetupSignal,
    btc_structure: str | None = None,
    btc_context: MarketContext | None = None,
) -> list[dict]:
    """Build user content blocks with chart images. No screener context — LLM analyzes blind."""
    content = []

    for path in chart_paths:
        image_data, media_type = encode_image(str(path))
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data,
            },
        })

    multi = len(chart_paths) > 1
    prefix = (
        "Analyze these charts (ordered from highest to lowest timeframe). "
        "Use the higher timeframe(s) for structure and key levels, and the lowest timeframe for entry timing."
    ) if multi else "Analyze this chart."

    constraint = ""
    if signal.max_leverage:
        constraint = f"\n\nExchange constraint: Maximum leverage for this pair is {signal.max_leverage}x. Do not recommend leverage above this."

    content.append({
        "type": "text",
        "text": f"{prefix}\n\nApply the ruleset. Identify the setup, classify, and evaluate from scratch.{constraint}",
    })

    return content


# Cost per million tokens (USD) — https://platform.claude.com/docs/en/about-claude/pricing
COST_PER_M = {
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    "claude-opus-4-6": {"input": 5.0, "output": 25.0},
}


def _estimate_cost(usage, model: str) -> float:
    rates = COST_PER_M.get(model, {"input": 3.0, "output": 15.0})
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    non_cached = max(0, usage.input_tokens - cache_creation - cache_read)
    input_cost = (non_cached / 1_000_000) * rates["input"]
    cache_write_cost = (cache_creation / 1_000_000) * rates["input"] * 1.25
    cache_read_cost = (cache_read / 1_000_000) * rates["input"] * 0.1
    output_cost = (usage.output_tokens / 1_000_000) * rates["output"]
    return round(input_cost + cache_write_cost + cache_read_cost + output_cost, 4)


def analyze_for_scanner(
    signal: SetupSignal,
    chart_paths: list[Path],
    config: ScannerConfig | None = None,
    btc_structure: str | None = None,
    btc_context: MarketContext | None = None,
) -> ScanResult:
    """Run LLM deep analysis on a screener signal. Returns ScanResult."""
    cfg = config or ScannerConfig()

    strategy = load_strategy()
    system_text = SYSTEM_PROMPT.format(ruleset=strategy) + (
        "\n\n## SCANNER MODE\n"
        "Scans run every 4 hours. Put analysis in thinking tokens; the written\n"
        "response is the decision block only.\n\n"

        "## ENTRY TYPE — pick one\n"
        "Default to CONDITIONAL. Use IMMEDIATE only when the confirming event\n"
        "is already complete on the last closed candle.\n\n"

        "**CONDITIONAL (default):** Setup is valid; a specific future event\n"
        "must print before entry. The monitor watches spot price or 1H closes,\n"
        "enters on trigger, kills on invalidation, or expires.\n\n"
        "Pick the trigger that matches the setup (ruleset §4):\n"
        "- `limit_at` (§4.2 Outright Price) — fires when price touches the\n"
        "  specified level. Best for fresh levels and clean retests where\n"
        "  you want the exact price.\n"
        "- `candle_close_above` / `candle_close_below` (§4.6 Candle Close) —\n"
        "  fires when a 1H closes past the level. Best for directional\n"
        "  confirmation on breakouts and trending continuations.\n"
        "- `deviation_reclaim` (§4.4 Deviation) — fires after price closes\n"
        "  OUTSIDE the level then closes back INSIDE. Best for range-bound\n"
        "  reversals and failed breakouts.\n\n"
        "Fields:\n"
        "- entry_trigger: one of the four names above\n"
        "- entry_trigger_price: the level (limit price, candle-close level,\n"
        "  or deviation boundary)\n"
        "- invalidation_price: kills the setup before the trigger fires\n"
        "  (further from entry than stop_loss)\n"
        "- expiry_candles: 1H candles to wait (1-24; match thesis speed)\n"
        "- entry_low/entry_high: bracket entry_trigger_price (±0.2-0.5%);\n"
        "  execution re-brackets at trigger time\n\n"
        "Confirming-event examples by trigger:\n"
        "- limit_at LONG: price drops to support — fill at the exact level\n"
        "- limit_at SHORT: price rallies into resistance — fill at the level\n"
        "- candle_close_above LONG: 1H close above resistance confirms breakout\n"
        "- candle_close_below SHORT: 1H close below support confirms breakdown\n"
        "- deviation_reclaim LONG: 1H close back above support after wicking\n"
        "  below (failed breakdown)\n"
        "- deviation_reclaim SHORT: 1H close back below resistance after\n"
        "  wicking above (failed breakout)\n\n"

        "**IMMEDIATE (exception):** The confirming event already printed on\n"
        "the last closed candle. Name it in Confirming Event.\n"
        "Examples:\n"
        "- MSB just confirmed on the last 4H close\n"
        "- Deviation reclaim just printed on the last 1H close\n"
        "- Rejection wick off a level with the body closing back on the\n"
        "  working side\n"
        "- Structural break in LTF anchoring a HTF setup\n\n"
        "Fields:\n"
        "- entry_low/entry_high: bracket current price (within 0.5% of last\n"
        "  candle close)\n"
        "- Omit entry_trigger, entry_trigger_price, invalidation_price,\n"
        "  expiry_candles\n\n"

        "Price at a level without confirmation yet = CONDITIONAL. Specify\n"
        "the candle close that confirms it.\n\n"

        "## INVALIDATION FIELDS (CONFIRM only)\n"
        "- **Expected Hold:** trade duration estimate (e.g. '1-2 days')\n"
        "- **Time Invalidation:** abandon point if neither SL nor TP hits\n\n"

        "## MACHINE BLOCK (CONFIRM only)\n"
        "Append JSON after the decision block:\n\n"
        '<!-- MACHINE_BLOCK\n'
        '{"entry_low": 61500.0, "entry_high": 62000.0, '
        '"stop_loss": 59800.0, "tp1": 65000.0, "tp1_exit_pct": 40, '
        '"tp2": 68000.0, "tp2_exit_pct": 40, "direction": "LONG", '
        '"confidence": 7, "leverage": 5, '
        '"setup_type": "C", "confluence_score": 8, '
        '"confirming_event": "1H close above flipped 62k resistance", '
        '"entry_trigger": "candle_close_above", '
        '"entry_trigger_price": 62000.0, '
        '"invalidation_price": 59000.0, "expiry_candles": 8}\n'
        "MACHINE_BLOCK -->\n\n"

        "Field rules:\n"
        "- Prices: plain numbers, no $ or commas\n"
        "- tp1_exit_pct: 30-50 (TP1 is the conservative FTA — §6.4)\n"
        "- tp2_exit_pct: 30-50 (when multi-target)\n"
        "- setup_type: single letter A-G per ruleset §3.1\n"
        "- confluence_score: integer 0-15 — sum of applicable §3.2 factors\n"
        "- confirming_event: 10-200 chars naming the specific event that\n"
        "  confirms entry (for both IMMEDIATE and CONDITIONAL)\n"
        "- IMMEDIATE: omit entry_trigger, entry_trigger_price,\n"
        "  invalidation_price, expiry_candles\n"
        "- Trailing TP: execution trails post-TP1 — provide a concrete TP1\n"
        "  to anchor the trail\n\n"

        "## CORE RULES (recap)\n"
        "- Default to CONDITIONAL. IMMEDIATE requires a confirming event on\n"
        "  the last closed candle.\n"
        "- Name the confirming event in every CONFIRM.\n"
        "- A valid thesis with uncertain timing is CONFIRM-with-trigger, not\n"
        "  REJECT.\n"
    )
    user_content = _build_scanner_content(chart_paths, signal, btc_structure, btc_context)
    messages = [{"role": "user", "content": user_content}]

    # Explicit timeout + retries — SDK defaults are 10min/2 retries, which
    # is longer than our staleness budget. Adaptive thinking runs can still
    # burn ~3min, so 180s is the minimum safe ceiling.
    client = anthropic.Anthropic(timeout=180.0, max_retries=3)
    response = client.messages.create(
        model=cfg.model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=[{
            "type": "text",
            "text": system_text,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=messages,
    )

    # Warn if response was truncated
    if response.stop_reason == "max_tokens":
        print(f"  ⚠ {signal.coin}: Response truncated (max_tokens). Output may be incomplete.")

    # Extract text (skip thinking blocks)
    text_parts = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
    analysis_text = "\n".join(text_parts)

    # Handle empty text (thinking-only response — no decision block to parse)
    if not analysis_text.strip():
        print(f"  ⚠ {signal.coin}: Empty LLM response (stop_reason={response.stop_reason}). "
              f"Model may have put everything in thinking tokens.")

    parsed = parse_decision_block(analysis_text)
    cost = _estimate_cost(response.usage, cfg.model)

    # Try machine block first (more reliable than regex on free text)
    machine_block = parse_machine_block(analysis_text)

    if machine_block and "confidence" in machine_block:
        llm_confidence = max(1, min(10, int(machine_block["confidence"])))
    else:
        # Regex fallback — reject values out of 1-10 rather than silently
        # clamping (LLM drift to percent-form like "85/100" otherwise looks
        # identical to a clean confidence=10 downstream).
        llm_confidence = None
        raw_conf = parsed.get("llm_confidence")
        if raw_conf:
            import re as _re
            m = _re.search(r"(\d+)", raw_conf)
            if m:
                value = int(m.group(1))
                if 1 <= value <= 10:
                    llm_confidence = value
                else:
                    log.warning(
                        "Ignoring out-of-range confidence %s for %s (expected 1-10)",
                        value, signal.coin,
                    )

    return ScanResult(
        scan_id=str(uuid.uuid4())[:8],
        coin=signal.coin,
        timestamp=datetime.now(timezone.utc),
        setup_signal=signal,
        decision=parsed.get("decision"),
        confluence_score=parsed.get("confluence_score"),
        analysis_text=analysis_text,
        parsed_decision=parsed,
        chart_paths=[str(p) for p in chart_paths],
        cost_usd=cost,
        llm_confidence=llm_confidence,
        machine_block=machine_block,
        detected_at=signal.detected_at if signal else None,
    )
