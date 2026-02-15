"""Discord webhook alerts — formatted embeds with chart images."""

import requests
from pathlib import Path

from .config import ScannerConfig
from .models import ScanResult, SETUP_NAMES, Direction


# Discord embed color
COLOR_LONG = 0x26A69A   # green
COLOR_SHORT = 0xEF5350  # red
COLOR_NEUTRAL = 0x999999


def _build_embed(result: ScanResult) -> dict:
    """Build a Discord embed dict from a scan result."""
    signal = result.setup_signal
    parsed = result.parsed_decision or {}
    setup_name = SETUP_NAMES.get(signal.setup_type, signal.setup_type.value)

    direction_str = signal.direction.value
    color = COLOR_LONG if signal.direction == Direction.LONG else COLOR_SHORT

    title = f"{direction_str} {result.coin} | Setup {signal.setup_type.value}: {setup_name}"

    fields = []

    if parsed.get("entry_zone"):
        fields.append({"name": "Entry Zone", "value": parsed["entry_zone"], "inline": True})
    if parsed.get("stop_loss"):
        fields.append({"name": "Stop Loss", "value": parsed["stop_loss"], "inline": True})
    if parsed.get("take_profit"):
        fields.append({"name": "Take Profit", "value": parsed["take_profit"], "inline": True})
    if result.confluence_score:
        fields.append({"name": "Confluence", "value": result.confluence_score, "inline": True})
    if parsed.get("leverage"):
        fields.append({"name": "Leverage", "value": parsed["leverage"], "inline": True})
    if parsed.get("risk_guidance"):
        fields.append({"name": "Risk", "value": parsed["risk_guidance"], "inline": True})
    if parsed.get("reasoning"):
        reasoning = parsed["reasoning"][:1000]
        fields.append({"name": "Reasoning", "value": reasoning, "inline": False})
    if parsed.get("what_would_change"):
        fields.append({"name": "What Changes", "value": parsed["what_would_change"][:500], "inline": False})

    embed = {
        "title": title,
        "color": color,
        "fields": fields,
        "footer": {"text": f"Scan {result.scan_id} | Cost: ${result.cost_usd:.4f}"},
    }

    return embed


def send_alert(
    result: ScanResult,
    config: ScannerConfig | None = None,
) -> bool:
    """Post a CONFIRM alert to Discord. Returns True on success.

    Attaches the 4H chart image if available.
    """
    cfg = config or ScannerConfig()
    if not cfg.discord_webhook_url:
        return False

    embed = _build_embed(result)
    payload = {"embeds": [embed]}

    # Find the 4H chart to attach
    chart_file = None
    for path_str in result.chart_paths:
        if "4H" in path_str or "4h" in path_str:
            chart_file = Path(path_str)
            break
    if not chart_file and result.chart_paths:
        chart_file = Path(result.chart_paths[0])

    try:
        if chart_file and chart_file.exists():
            # Multipart: embed as JSON + chart as file
            import json
            resp = requests.post(
                cfg.discord_webhook_url,
                data={"payload_json": json.dumps(payload)},
                files={"file": (chart_file.name, chart_file.read_bytes(), "image/png")},
                timeout=10,
            )
        else:
            resp = requests.post(
                cfg.discord_webhook_url,
                json=payload,
                timeout=10,
            )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Discord alert failed: {e}")
        return False


def format_alert_text(result: ScanResult) -> str:
    """Format a scan result as plain text (for CLI output)."""
    signal = result.setup_signal
    parsed = result.parsed_decision or {}
    setup_name = SETUP_NAMES.get(signal.setup_type, signal.setup_type.value)

    lines = [
        f"\n{'=' * 60}",
        f"  {signal.direction.value} {result.coin} | Setup {signal.setup_type.value}: {setup_name}",
        f"  Decision: {result.decision}",
    ]

    if result.confluence_score:
        lines.append(f"  Confluence: {result.confluence_score}")
    if parsed.get("entry_zone"):
        lines.append(f"  Entry: {parsed['entry_zone']}")
    if parsed.get("stop_loss"):
        lines.append(f"  Stop: {parsed['stop_loss']}")
    if parsed.get("take_profit"):
        lines.append(f"  Target: {parsed['take_profit']}")
    if parsed.get("leverage"):
        lines.append(f"  Leverage: {parsed['leverage']}")
    if parsed.get("reasoning"):
        lines.append(f"  Reasoning: {parsed['reasoning']}")

    lines.append(f"  Cost: ${result.cost_usd:.4f} | ID: {result.scan_id}")
    lines.append("=" * 60)

    return "\n".join(lines)
