"""CLI table formatting for backtest reports."""

from .metrics import MetricsReport


def format_summary(report: MetricsReport) -> str:
    """Format a single metrics report as a text block."""
    lines = [
        f"  Sample:          {report.sample_size}",
        f"  Win/Loss/Exp:    {report.wins}/{report.losses}/{report.expired}",
        f"  Win Rate:        {report.win_rate:.1%}",
        f"  Avg R:           {report.avg_r:+.2f}",
        f"  Expectancy:      {report.expectancy:+.4f}R",
    ]

    ci = report.bootstrap_ci
    if ci:
        marker = "edge" if ci.excludes_zero and ci.ci_low > 0 else "negative" if ci.excludes_zero else "inconclusive"
        lines.append(f"  95% CI:          [{ci.ci_low:+.4f}, {ci.ci_high:+.4f}]R  ({marker})")
    elif report.sample_size > 0:
        lines.append(f"  95% CI:          insufficient data (n < 5)")

    perm = report.permutation
    if perm:
        sig = "SIGNIFICANT" if perm.is_significant else "NOT SIGNIFICANT"
        lines.append(f"  Permutation:     p={perm.p_value:.4f}  ({sig})")

    lines.extend([
        f"  Profit Factor:   {report.profit_factor:.2f}" if report.profit_factor != float("inf") else "  Profit Factor:   inf",
        f"  Total R:         {report.total_r:+.2f}",
        f"  Avg Win R:       {report.avg_win_r:+.2f}",
        f"  Avg Loss R:      {report.avg_loss_r:+.2f}",
        f"  Avg Bars:        {report.avg_bars_to_exit:.0f}",
        f"  Avg MAE:         {report.avg_mae:.2f}R",
        f"  Avg MFE:         {report.avg_mfe:.2f}R",
    ])

    if report.avg_funding_r and report.avg_funding_r > 0:
        lines.append(f"  Avg Funding:     {report.avg_funding_r:.4f}R")
    return "\n".join(lines)


def format_slice_table(dimension: str, reports: list[MetricsReport]) -> str:
    """Format a comparison table for a sliced dimension."""
    if not reports:
        return f"  {dimension}: no data\n"

    # Header
    header = f"  {'Slice':<12} {'N':>5} {'WR':>7} {'ExpR':>8} {'95% CI':>22} {'PF':>7} {'TotR':>8}"
    sep = "  " + "-" * len(header.strip())
    rows = [header, sep]

    for r in reports:
        pf = f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "inf"
        ci = r.bootstrap_ci
        if ci:
            ci_str = f"[{ci.ci_low:+.3f},{ci.ci_high:+.3f}]"
        elif r.sample_size < 5:
            ci_str = "n/a"
        else:
            ci_str = ""
        rows.append(
            f"  {r.label:<12} {r.sample_size:>5} {r.win_rate:>6.1%} {r.expectancy:>+7.3f}R "
            f"{ci_str:>22} {pf:>6} {r.total_r:>+7.2f}"
        )

    return "\n".join(rows)


def format_full_report(
    overall: MetricsReport,
    sliced: dict[str, list[MetricsReport]],
    mode: str,
) -> str:
    """Format the complete backtest report."""
    sections = []

    sections.append(f"{'='*60}")
    sections.append(f"  BACKTEST REPORT — {mode.upper()} MODE")
    sections.append(f"{'='*60}")
    sections.append("")
    sections.append("OVERALL")
    sections.append(format_summary(overall))

    for dim_name, reports in sliced.items():
        sections.append("")
        sections.append(f"BY {dim_name.upper()}")
        sections.append(format_slice_table(dim_name, reports))

    sections.append("")
    sections.append(f"{'='*60}")
    return "\n".join(sections)
