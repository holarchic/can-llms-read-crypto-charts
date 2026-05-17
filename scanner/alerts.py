"""Discord webhook alerts — formatted embeds with chart images."""

from datetime import datetime, timezone
from pathlib import Path

import requests

from .config import ScannerConfig
from .models import SETUP_NAMES, ScanResult

# Discord embed color
COLOR_LONG = 0x26A69A   # green
COLOR_SHORT = 0xEF5350  # red
COLOR_NEUTRAL = 0x999999

# Ops log colors
COLOR_OK = 0x26A69A      # green — normal scan
COLOR_WARN = 0xFFA726    # orange — anomaly
COLOR_ERROR = 0xEF5350   # red — crash


def _build_embed(result: ScanResult, exchange_label: str = "HyperLiquid") -> dict:
    """Build a Discord embed dict from a scan result."""
    signal = result.setup_signal
    parsed = result.parsed_decision or {}

    # Prefer LLM's classification over screener's (LLM analyzes blind and may reclassify)
    llm_direction = parsed.get("direction", "").upper()
    if llm_direction in ("LONG", "SHORT"):
        direction_str = llm_direction
    else:
        direction_str = signal.direction.value

    llm_setup = parsed.get("setup", "")
    if llm_setup:
        setup_display = llm_setup
    else:
        setup_display = f"Setup {signal.setup_type.value}: {SETUP_NAMES.get(signal.setup_type, signal.setup_type.value)}"

    color = COLOR_LONG if direction_str == "LONG" else COLOR_SHORT

    title = f"{direction_str} {result.coin} | {setup_display}"
    if len(title) > 256:
        title = title[:253] + "..."

    fields = []
    mb = result.machine_block

    if mb is not None:
        # Structured fields from machine block (T-016 C5: typed access)
        if mb.entry_low == mb.entry_high:
            fields.append({"name": "Entry", "value": f"${mb.entry_low:,.5f}", "inline": True})
        else:
            fields.append({"name": "Entry Zone", "value": f"${mb.entry_low:,.5f} – ${mb.entry_high:,.5f}", "inline": True})
        fields.append({"name": "Stop Loss", "value": f"${mb.stop_loss:,.5f}", "inline": True})
        tp_parts = [f"TP1: ${mb.tp1:,.5f} ({mb.tp1_exit_pct:.0f}%)"]
        if mb.tp2:
            tp_parts.append(f"TP2: ${mb.tp2:,.5f} ({(mb.tp2_exit_pct if mb.tp2_exit_pct is not None else 0):.0f}%)")
        fields.append({"name": "Take Profit", "value": " | ".join(tp_parts), "inline": True})
    else:
        if parsed.get("entry_zone"):
            fields.append({"name": "Entry Zone", "value": parsed["entry_zone"], "inline": True})
        if parsed.get("stop_loss"):
            fields.append({"name": "Stop Loss", "value": parsed["stop_loss"], "inline": True})
        if parsed.get("take_profit"):
            fields.append({"name": "Take Profit", "value": parsed["take_profit"], "inline": True})
    if result.confluence_score:
        fields.append({"name": "Confluence", "value": result.confluence_score, "inline": True})
    if result.llm_confidence is not None:
        fields.append({"name": "Confidence", "value": f"{result.llm_confidence}/10", "inline": True})
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
        "footer": {"text": f"{exchange_label} | Scan {result.scan_id} | Cost: ${result.cost_usd:.4f}"},
    }

    return embed


def send_alert(
    result: ScanResult,
    config: ScannerConfig | None = None,
    exchange_label: str = "HyperLiquid",
) -> bool:
    """Post a CONFIRM alert to Discord. Returns True on success.

    Attaches the 4H chart image if available.
    """
    cfg = config or ScannerConfig()
    if not cfg.discord_ops_webhook_url:
        return False

    embed = _build_embed(result, exchange_label=exchange_label)
    payload = {"embeds": [embed]}

    # Find the 4H chart to attach
    chart_file = None
    for path_str in result.chart_paths:
        if "4H" in path_str or "4h" in path_str:
            chart_file = Path(path_str)
            break
    if not chart_file and result.chart_paths:
        chart_file = Path(result.chart_paths[0])

    import json

    try:
        if chart_file and chart_file.exists():
            # Multipart: embed as JSON + chart as file
            resp = requests.post(
                cfg.discord_ops_webhook_url,
                data={"payload_json": json.dumps(payload)},
                files={"file": (chart_file.name, chart_file.read_bytes(), "image/png")},
                timeout=10,
            )
            if resp.status_code == 400:
                # Log why Discord rejected the upload, then retry without chart
                print(f"Discord multipart rejected (400): {resp.text[:500]}")
                print(f"  Chart: {chart_file} ({chart_file.stat().st_size} bytes). Retrying without chart...")
                resp = requests.post(
                    cfg.discord_ops_webhook_url,
                    json=payload,
                    timeout=10,
                )
        else:
            resp = requests.post(
                cfg.discord_ops_webhook_url,
                json=payload,
                timeout=10,
            )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        body = ""
        if hasattr(e, "response") and e.response is not None:
            body = f" | Response: {e.response.text[:500]}"
        print(f"Discord alert failed: {e}{body}")
        return False


def format_alert_text(result: ScanResult) -> str:
    """Format a scan result as plain text (for CLI output)."""
    signal = result.setup_signal
    parsed = result.parsed_decision or {}

    # Prefer LLM's classification over screener's
    llm_direction = parsed.get("direction", "").upper()
    direction_str = llm_direction if llm_direction in ("LONG", "SHORT") else signal.direction.value
    llm_setup = parsed.get("setup", "")
    setup_display = llm_setup or f"Setup {signal.setup_type.value}: {SETUP_NAMES.get(signal.setup_type, signal.setup_type.value)}"

    lines = [
        f"\n{'=' * 60}",
        f"  {direction_str} {result.coin} | {setup_display}",
        f"  Decision: {result.decision}",
    ]

    if result.confluence_score:
        lines.append(f"  Confluence: {result.confluence_score}")
    if result.llm_confidence is not None:
        lines.append(f"  LLM Confidence: {result.llm_confidence}/10")
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


def send_report_embeds(
    embeds: list[dict],
    config: ScannerConfig | None = None,
) -> bool:
    """Post a list of embeds to the reports Discord channel."""
    cfg = config or ScannerConfig()
    webhook_url = cfg.discord_reports_webhook_url or cfg.discord_ops_webhook_url
    if not webhook_url:
        return False

    try:
        resp = requests.post(webhook_url, json={"embeds": embeds}, timeout=10)
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Discord report embeds failed: {e}")
        return False


def send_ops_log(
    title: str,
    fields: dict[str, str],
    color: int = COLOR_OK,
    config: ScannerConfig | None = None,
) -> bool:
    """Post an operational log embed to Discord. Must never crash the scanner."""
    try:
        cfg = config or ScannerConfig()
        webhook_url = cfg.discord_ops_webhook_url or cfg.discord_reports_webhook_url
        if not webhook_url:
            return False

        embed_fields = [
            {"name": k, "value": str(v), "inline": True}
            for k, v in fields.items()
        ]
        embed = {
            "title": title,
            "color": color,
            "fields": embed_fields,
            "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
        }
        resp = requests.post(webhook_url, json={"embeds": [embed]}, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"Ops log failed: {e}")
        return False


def send_errors_alert(
    message: str, config: ScannerConfig | None = None,
) -> bool:
    """Post a scanner-side error to the errors Discord channel.

    Falls back to ops webhook if errors-specific one isn't configured.
    Stage2 LLM failures and scanner crashes otherwise only surface in logs.
    """
    try:
        cfg = config or ScannerConfig()
        webhook_url = cfg.discord_errors_webhook_url or cfg.discord_ops_webhook_url
        if not webhook_url:
            return False
        resp = requests.post(
            webhook_url,
            json={"content": f"**Scanner error**: {message}"},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"Scanner errors alert failed: {e}")
        return False
