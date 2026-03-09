"""LLM deep analysis — orchestrates chart analysis via Claude vision API."""

PROMPT_VERSION = "3"

import uuid
from datetime import datetime, timezone
from pathlib import Path

import anthropic

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
        "BTC declines propagate to alts with 1.5–3x amplification. Weight bearish BTC context more heavily than bullish.",
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

    market_ctx = ""
    if btc_context and signal.coin != "BTC":
        market_ctx = "\n\n" + _build_btc_context(btc_context)
    elif btc_structure and signal.coin != "BTC":
        # Fallback to simple string if full context unavailable
        market_ctx = (f"\n\nMarket context: BTC Daily structure is {btc_structure}."
                      " Factor this into your directional bias and invalidation conditions.")

    content.append({
        "type": "text",
        "text": f"{prefix}\n\nApply the ruleset. Identify the setup, classify, and evaluate from scratch.{constraint}{market_ctx}",
    })

    return content


# Cost per million tokens (USD) — https://platform.claude.com/docs/en/about-claude/pricing
COST_PER_M = {
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    "claude-opus-4-6": {"input": 5.0, "output": 25.0},
}


def _estimate_cost(usage, model: str) -> float:
    rates = COST_PER_M.get(model, {"input": 3.0, "output": 15.0})
    input_cost = (usage.input_tokens / 1_000_000) * rates["input"]
    cache_write = (getattr(usage, "cache_creation_input_tokens", 0) / 1_000_000) * rates["input"] * 1.25
    cache_read = (getattr(usage, "cache_read_input_tokens", 0) / 1_000_000) * rates["input"] * 0.1
    output_cost = (usage.output_tokens / 1_000_000) * rates["output"]
    return round(input_cost + cache_write + cache_read + output_cost, 4)


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
        "Perform the full 8-step analysis procedure in your thinking. "
        "In your written response, output ONLY the DECISION block "
        "(from //////////// to ////////////). No step-by-step walkthrough.\n\n"
        "## DECISION CLASSIFICATION (scanner: CONFIRM or REJECT only — no WAIT)\n"
        "- CONFIRM: An actionable trade exists RIGHT NOW at current price. "
        "Entry zone, stop loss, and take profit are all defined.\n"
        "- REJECT: Everything else — no valid setup, price not at a level, "
        "conditions unmet, or setup needs a trigger that hasn't fired yet. "
        "If you would not enter this trade right now, REJECT.\n\n"
        "## LLM CONFIDENCE (CONFIRM only)\n"
        "Include the **LLM Confidence:** field only for CONFIRM decisions. "
        "Score 1-10 reflects your conviction in the trade. "
        "1 is the weakest possible CONFIRM; anything weaker should be REJECT.\n\n"
        "## INVALIDATION FIELDS (CONFIRM only)\n"
        "Include these fields in the decision block:\n"
        "- **Expected Hold:** How long this trade should take (e.g. '1-2 days', '3-5 days')\n"
        "- **Time Invalidation:** When to abandon if neither SL nor TP hit "
        "(e.g. '4 days' = 2x expected hold). Stale trades lose their edge.\n\n"
        "## MACHINE-READABLE OUTPUT (CONFIRM only)\n"
        "After the decision block, include a JSON machine block for automated parsing.\n"
        "Use this exact format — HTML comment wrapper, one line of JSON, no markdown:\n\n"
        '<!-- MACHINE_BLOCK\n'
        '{"entry_low": 61500.0, "entry_high": 62000.0, "stop_loss": 59800.0, '
        '"tp1": 65000.0, "tp1_exit_pct": 50, "tp2": 68000.0, "tp2_exit_pct": 50, '
        '"direction": "LONG", "confidence": 7, "leverage": 5}\n'
        "MACHINE_BLOCK -->\n\n"
        "Rules:\n"
        "- Include ONLY for CONFIRM decisions\n"
        "- All prices as plain numbers (no $ signs, no commas)\n"
        "- entry_low/entry_high: the range where a MARKET ORDER should fill right now. "
        "This must bracket the current price (right edge of chart). "
        "Do NOT use ideal limit-order levels — the system executes immediately at market. "
        "A narrow range around current price is correct (e.g. within 0.5% of last candle close).\n"
        "- entry_low/entry_high: if single price, set both equal\n"
        "- tp1_exit_pct: 30-70 (percentage to close at TP1)\n"
        "- tp2/tp2_exit_pct: include when multi-target appropriate; tp2_exit_pct is remainder\n"
        "- If any target is a trailing stop (no fixed price), omit machine block entirely"
    )
    user_content = _build_scanner_content(chart_paths, signal, btc_structure, btc_context)
    messages = [{"role": "user", "content": user_content}]

    client = anthropic.Anthropic()
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
        # Fall back to regex confidence extraction
        llm_confidence = None
        raw_conf = parsed.get("llm_confidence")
        if raw_conf:
            import re as _re
            m = _re.search(r"(\d+)", raw_conf)
            if m:
                llm_confidence = max(1, min(10, int(m.group(1))))

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
    )
