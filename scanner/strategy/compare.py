"""Side-by-side A/B comparison formatting for strategy metrics."""

from ..backtest.metrics import MetricsReport


def _delta_str(a: float, b: float, fmt: str = "+.2f", higher_better: bool = True) -> str:
    """Format a delta with directional indicator."""
    diff = b - a
    if abs(diff) < 1e-6:
        return "="
    arrow = "^" if (diff > 0) == higher_better else "v"
    return f"{diff:{fmt}} {arrow}"


def format_comparison(report_a: MetricsReport, report_b: MetricsReport) -> str:
    """Side-by-side table with deltas highlighted."""
    rows = [
        f"{'':>18}  {'A: ' + report_a.label:>16}  {'B: ' + report_b.label:>16}  {'Delta':>12}",
        "  " + "-" * 66,
    ]

    def _row(label, va, vb, fmt=".2f", higher_better=True):
        delta = _delta_str(va, vb, fmt=fmt, higher_better=higher_better)
        rows.append(f"  {label:>16}  {va:>16{fmt}}  {vb:>16{fmt}}  {delta:>12}")

    _row("Sample", report_a.sample_size, report_b.sample_size, fmt=".0f", higher_better=True)
    _row("Win Rate", report_a.win_rate, report_b.win_rate, fmt=".1%")
    _row("Expectancy", report_a.expectancy, report_b.expectancy, fmt="+.4f")
    _row("Profit Factor", report_a.profit_factor, report_b.profit_factor)
    _row("Total R", report_a.total_r, report_b.total_r, fmt="+.2f")
    _row("Avg Win R", report_a.avg_win_r, report_b.avg_win_r, fmt="+.2f")
    _row("Avg Loss R", report_a.avg_loss_r, report_b.avg_loss_r, fmt="+.2f", higher_better=False)
    _row("Avg Bars", report_a.avg_bars_to_exit, report_b.avg_bars_to_exit, fmt=".0f", higher_better=False)
    _row("Avg MAE", report_a.avg_mae, report_b.avg_mae, fmt=".2f", higher_better=False)
    _row("Avg MFE", report_a.avg_mfe, report_b.avg_mfe, fmt=".2f")

    return "\n".join(rows)
