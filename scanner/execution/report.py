"""Execution reporting — CLI text + Discord embeds."""


def format_execution_report(metrics: dict, sim_vs_live: list[dict], mode: str) -> str:
    """Format execution metrics as CLI text report."""
    lines = [
        f"\nExecution Report ({mode.upper()})",
        "=" * 50,
        f"  Closed trades:    {metrics['count']}",
        f"  Win rate:         {metrics['win_rate']:.1%}",
        f"  Expectancy:       {metrics['expectancy_r']:+.4f}R",
        f"  Profit factor:    {metrics['profit_factor']:.2f}",
        f"  Total PnL:        ${metrics['total_pnl']:+,.2f}",
        f"  Max drawdown:     {metrics['max_drawdown_pct']:.1f}%",
        f"  Avg holding:      {metrics['avg_holding_hours']:.1f}h",
        f"  Avg slippage:     {metrics['avg_slippage_r']:.4f}R",
        f"  Total funding:    {metrics['total_funding_r']:+.4f}R",
    ]

    if sim_vs_live:
        lines.append(f"\nSim vs Live ({len(sim_vs_live)} trades):")
        lines.append(f"  {'Coin':>8} {'Sim R':>8} {'Live R':>8} {'Delta':>8} {'Slip':>8}")
        lines.append("  " + "-" * 44)
        for c in sim_vs_live[:20]:
            lines.append(
                f"  {c['coin']:>8} {c['sim_r']:>+8.4f} {c['live_r']:>+8.4f} "
                f"{c['delta_r']:>+8.4f} {c['slippage_r']:>8.4f}"
            )
        if len(sim_vs_live) > 20:
            lines.append(f"  ... and {len(sim_vs_live) - 20} more")

        avg_delta = sum(c["delta_r"] for c in sim_vs_live) / len(sim_vs_live)
        lines.append(f"\n  Avg delta (live - sim): {avg_delta:+.4f}R")

    return "\n".join(lines)


def build_execution_embeds(metrics: dict, sim_vs_live: list[dict], mode: str) -> list[dict]:
    """Build Discord embed dicts for execution report."""
    color = 0x26A69A if metrics.get("expectancy_r", 0) > 0 else 0xEF5350

    fields = [
        {"name": "Trades", "value": str(metrics["count"]), "inline": True},
        {"name": "Win Rate", "value": f"{metrics['win_rate']:.1%}", "inline": True},
        {"name": "Expectancy", "value": f"{metrics['expectancy_r']:+.4f}R", "inline": True},
        {"name": "PnL", "value": f"${metrics['total_pnl']:+,.2f}", "inline": True},
        {"name": "Profit Factor", "value": f"{metrics['profit_factor']:.2f}", "inline": True},
        {"name": "Max DD", "value": f"{metrics['max_drawdown_pct']:.1f}%", "inline": True},
        {"name": "Avg Holding", "value": f"{metrics['avg_holding_hours']:.1f}h", "inline": True},
        {"name": "Avg Slippage", "value": f"{metrics['avg_slippage_r']:.4f}R", "inline": True},
    ]

    embeds = [{
        "title": f"Execution Report ({mode.upper()})",
        "color": color,
        "fields": fields,
    }]

    if sim_vs_live:
        avg_delta = sum(c["delta_r"] for c in sim_vs_live) / len(sim_vs_live)
        embeds.append({
            "title": "Sim vs Live",
            "color": color,
            "description": (
                f"**{len(sim_vs_live)} trades compared**\n"
                f"Avg delta (live - sim): {avg_delta:+.4f}R"
            ),
        })

    return embeds
