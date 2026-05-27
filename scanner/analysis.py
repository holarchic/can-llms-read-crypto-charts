"""LLM deep analysis — orchestrates chart analysis via Claude vision API."""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import anthropic

log = logging.getLogger(__name__)

from .config import ScannerConfig
from .llm import (
    encode_image,
    parse_decision_block,
    parse_machine_block,
)
from .models import BoundarySignal, ScanResult
from .prompt_select import select_prompt


class LLMOverloaded(Exception):
    """Anthropic 529 overload after SDK retry-exhaustion (SPEC #37). Raised
    by `analyze_for_scanner` so callers consume a scanner-domain type, not
    `anthropic.APIStatusError` — the SDK coupling stays inside this module."""

    def __init__(self, request_id: str | None) -> None:
        super().__init__(f"Anthropic 529 (request_id={request_id})")
        self.request_id = request_id


def _build_scanner_content(
    chart_paths: list[Path],
    signal: BoundarySignal,
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
    "claude-opus-4-7": {"input": 5.0, "output": 25.0},
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
    signal: BoundarySignal,
    chart_paths: list[Path],
    config: ScannerConfig | None = None,
) -> ScanResult:
    """Run LLM deep analysis on a screener signal. Returns ScanResult."""
    cfg = config or ScannerConfig()

    spec = select_prompt(cfg.setup_version)
    system_text = spec.system_text
    user_content = _build_scanner_content(chart_paths, signal)
    messages = [{"role": "user", "content": user_content}]

    # Explicit timeout + retries — SDK defaults are 10min/2 retries, which
    # is longer than our staleness budget. Adaptive thinking runs can still
    # burn ~3min, so 180s is the minimum safe ceiling. max_retries=5 yields
    # ~15s of exponential backoff (0.5/1/2/4/8s) — captures short Anthropic
    # 529 blips; longer outages raise LLMOverloaded to the caller.
    client = anthropic.Anthropic(timeout=180.0, max_retries=5)
    try:
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
    except anthropic.APIStatusError as e:
        # 529 retry-exhaustion translates to a scanner-domain exception so the
        # caller never imports anthropic. Other APIStatusError codes propagate
        # to the caller's generic exception handler unchanged.
        if e.status_code == 529:
            raise LLMOverloaded(request_id=e.request_id) from e
        raise

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

    if machine_block is not None:
        # confidence is a required MachineBlock field — always present.
        llm_confidence = max(1, min(10, int(machine_block.confidence)))
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
        prompt_version=spec.prompt_version,
    )
