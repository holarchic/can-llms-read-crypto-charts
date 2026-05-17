"""Execution reporting — CLI text + Discord embeds."""


def _fmt_pct(v: float) -> str:
    return f"{v:.1%}"


def _row(label: str, v7: str, v28: str) -> str:
    return f"  {label:<18} {v7:>10} {v28:>10}"


def format_execution_report(
    metrics_7d: dict,
    metrics_28d: dict,
    cost_7d: float,
    cost_28d: float,
    mode: str,
) -> str:
    """Format actual-trade performance for both windows as CLI text."""
    lines = [
        f"\nExecution Report ({mode.upper()})",
        "=" * 50,
        f"  Data source:      {metrics_28d.get('data_source', 'db_only').upper()}",
        "",
        f"  {'':<18} {'7d':>10} {'28d':>10}",
        f"  {'-' * 40}",
        _row("Closed trades", str(metrics_7d["count"]), str(metrics_28d["count"])),
        _row("Win rate", _fmt_pct(metrics_7d["win_rate"]), _fmt_pct(metrics_28d["win_rate"])),
        _row("Expectancy", f"{metrics_7d['expectancy_r']:+.4f}R", f"{metrics_28d['expectancy_r']:+.4f}R"),
        _row("Profit factor", f"{metrics_7d['profit_factor']:.2f}", f"{metrics_28d['profit_factor']:.2f}"),
        _row("Total PnL", f"${metrics_7d['total_pnl']:+,.2f}", f"${metrics_28d['total_pnl']:+,.2f}"),
        _row("Max drawdown", f"{metrics_7d['max_drawdown_pct']:.1f}%", f"{metrics_28d['max_drawdown_pct']:.1f}%"),
        _row("Avg holding", f"{metrics_7d['avg_holding_hours']:.1f}h", f"{metrics_28d['avg_holding_hours']:.1f}h"),
        _row("Avg slippage", f"{metrics_7d['avg_slippage_r']:.4f}R", f"{metrics_28d['avg_slippage_r']:.4f}R"),
        "",
        "LLM Cost",
        f"  7d:   ${cost_7d:,.2f}",
        f"  28d:  ${cost_28d:,.2f}",
    ]

    # Quality warnings come from the broader (28d) window — wider sample is more informative.
    warnings = metrics_28d.get("quality_warnings") or []
    if warnings:
        lines.append("")
        lines.append("  Data quality:")
        for w in warnings:
            lines.append(f"    * {w}")

    return "\n".join(lines)


def _window_embed(metrics: dict, cost: float, label: str, mode: str) -> dict:
    """Single-window Discord embed."""
    color = 0x26A69A if metrics.get("expectancy_r", 0) > 0 else 0xEF5350
    pf = f"{metrics['profit_factor']:.2f}" if metrics["profit_factor"] != float("inf") else "inf"

    fields = [
        {"name": "Trades", "value": str(metrics["count"]), "inline": True},
        {"name": "Win Rate", "value": f"{metrics['win_rate']:.1%}", "inline": True},
        {"name": "Expectancy", "value": f"{metrics['expectancy_r']:+.4f}R", "inline": True},
        {"name": "PnL", "value": f"${metrics['total_pnl']:+,.2f}", "inline": True},
        {"name": "Profit Factor", "value": pf, "inline": True},
        {"name": "Max DD", "value": f"{metrics['max_drawdown_pct']:.1f}%", "inline": True},
        {"name": "Avg Holding", "value": f"{metrics['avg_holding_hours']:.1f}h", "inline": True},
        {"name": "Avg Slippage", "value": f"{metrics['avg_slippage_r']:.4f}R", "inline": True},
        {"name": "LLM Cost", "value": f"${cost:,.2f}", "inline": True},
    ]
    return {
        "title": f"Execution Report — {label} ({mode.upper()})",
        "color": color,
        "fields": fields,
    }


def build_execution_embeds(
    metrics_7d: dict,
    metrics_28d: dict,
    cost_7d: float,
    cost_28d: float,
    mode: str,
) -> list[dict]:
    """Two embeds: 7d window then 28d window. Quality warnings appended to the 28d embed."""
    embed_7d = _window_embed(metrics_7d, cost_7d, "7d", mode)
    embed_28d = _window_embed(metrics_28d, cost_28d, "28d", mode)

    warnings = metrics_28d.get("quality_warnings") or []
    if warnings:
        embed_28d["fields"].append({
            "name": "Data Quality",
            "value": "\n".join(warnings),
            "inline": False,
        })

    return [embed_7d, embed_28d]
