"""LLM deep analysis — imports building blocks from chart_analyzer."""

import uuid
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from chart_analyzer.analyzer import (
    SYSTEM_PROMPT,
    encode_image,
    load_strategy,
    parse_decision_block,
)
from .config import ScannerConfig
from .models import ScanResult, SetupSignal


def _build_scanner_content(
    chart_paths: list[Path],
    signal: SetupSignal,
    btc_structure: str | None = None,
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
    if btc_structure and signal.coin != "BTC":
        market_ctx = f"\n\nMarket context: BTC 4H structure is {btc_structure}. Factor this into your directional bias and invalidation conditions."

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
) -> ScanResult:
    """Run LLM deep analysis on a screener signal. Returns ScanResult."""
    cfg = config or ScannerConfig()

    strategy = load_strategy()
    system_text = SYSTEM_PROMPT.format(ruleset=strategy) + (
        "\n\n## SCANNER MODE\n"
        "Perform the full 8-step analysis procedure in your thinking. "
        "In your written response, output ONLY the DECISION block "
        "(from //////////// to ////////////). No step-by-step walkthrough.\n\n"
        "## DECISION CLASSIFICATION\n"
        "- CONFIRM: An actionable trade exists RIGHT NOW at current price. "
        "Entry zone, stop loss, and take profit are all defined.\n"
        "- REJECT: No valid setup, or price is not at a tradeable level, "
        "or conditions do not meet the ruleset criteria. Use REJECT for any "
        "situation where you would not take the trade.\n"
        "- WAIT: Use ONLY when a specific, high-quality setup exists but needs "
        "a clear trigger that hasn't fired yet (e.g. a candle close confirmation). "
        "WAIT must still include full entry/SL/TP. If there is no concrete plan, use REJECT.\n\n"
        "## INVALIDATION FIELDS (required for CONFIRM and WAIT)\n"
        "Include these fields in the decision block:\n"
        "- **Expected Hold:** How long this trade should take (e.g. '1-2 days', '3-5 days')\n"
        "- **Time Invalidation:** When to abandon if neither SL nor TP hit "
        "(e.g. '4 days' = 2x expected hold). Stale trades lose their edge."
    )
    user_content = _build_scanner_content(chart_paths, signal, btc_structure)
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

    # Extract LLM confidence integer from parsed field
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
    )
