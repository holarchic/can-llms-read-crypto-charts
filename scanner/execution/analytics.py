"""Execution analytics — pure compute over the storage-owned closed-trade
read model (T-016 Phase 2a, SPEC #22). No DB access here: `get_closed_trades`
/ `llm_cost_summary` live in `scanner.execution.storage`."""

from scanner.expectancy import expectancy_report
from scanner.models import MetricsWindow, RiskedOutcomes, TradeRecord, VersionCohort


def risked_outcomes(trades: list[TradeRecord]) -> RiskedOutcomes:
    """Single owner of the risk+outcome partition (T-016 Phase 3, SPEC #26 b;
    KL#9 b). `risked` = trades with a usable risk anchor (`usable_risk`);
    no/degenerate-SL trades are EXCLUDED, never counted via realized_r==0 as
    break-even losses (which would dilute expectancy, depress win-rate, and —
    for a profitable no-SL trade — contaminate profit factor; SPEC #24, KL#2
    v). The `realized_r > 0` win/loss split is folded in here, not duplicated
    at call sites, because it drives win_rate/profit_factor (not
    presentation). Returns a named `RiskedOutcomes` record, not a bare
    same-typed 3-tuple — a positional seam hides a wins/losses swap."""
    risked = tuple(t for t in trades if t.usable_risk)
    wins = tuple(t for t in risked if t.realized_r > 0)
    losses = tuple(t for t in risked if t.realized_r <= 0)
    return RiskedOutcomes(risked=risked, wins=wins, losses=losses)


def compute_live_metrics(trades: list[TradeRecord]) -> MetricsWindow:
    """Compute aggregate performance metrics from closed trades."""
    if not trades:
        return MetricsWindow(
            count=0, risked_count=0, win_rate=0, expectancy_r=0, profit_factor=0,
            total_pnl=0, max_drawdown_pct=0,
            avg_holding_hours=0, avg_slippage_r=0, avg_fees_r=0, total_funding_r=0,
            fill_rate=0, data_source="db_only",
            quality_warnings=_compute_quality_warnings([]),
        )

    # Risk+outcome partition is single-owned (SPEC #26 b). No-usable-risk
    # trades are excluded from R-normalized metrics but stay in non-R
    # aggregates (count/total_pnl/drawdown/holding).
    ro = risked_outcomes(trades)
    risked, wins, losses = ro.risked, ro.wins, ro.losses

    win_rate = len(wins) / len(risked) if risked else 0
    expectancy_r = sum(t.realized_r for t in risked) / len(risked) if risked else 0
    total_pnl = sum(t.realized_pnl for t in trades)

    # PnL sums are net of taker fees (realized_pnl is net, SPEC #42) — coherent
    # with the realized_r-signed win/loss partition (sign(realized_r) == sign(realized_pnl)).
    profit = sum(t.realized_pnl for t in wins)
    loss = abs(sum(t.realized_pnl for t in losses))
    profit_factor = profit / loss if loss > 0 else float("inf") if profit > 0 else 0

    total_risk = sum(t.risk_amount for t in risked)

    # Max drawdown (simple: track running PnL, find max peak-to-trough)
    running = 0
    peak = 0
    max_dd = 0
    for t in trades:
        running += t.realized_pnl
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd
    # Express as % of peak, or % of total risk if equity never went positive
    if peak > 0:
        max_dd_pct = max_dd / peak * 100
    elif total_risk > 0:
        max_dd_pct = max_dd / total_risk * 100
    else:
        max_dd_pct = 0

    avg_holding = sum(t.holding_hours for t in trades) / len(trades)
    # avg_slippage_r / avg_fees_r / total_funding_r are R-normalized → over `risked`.
    avg_slippage = sum(t.slippage_r for t in risked) / len(risked) if risked else 0
    avg_fees = sum(t.fees_r for t in risked) / len(risked) if risked else 0
    total_funding = sum(t.funding_paid for t in risked)
    total_funding_r = total_funding / total_risk if total_risk > 0 else 0

    return MetricsWindow(
        count=len(trades),
        risked_count=len(risked),
        win_rate=round(win_rate, 4),
        expectancy_r=round(expectancy_r, 4),
        profit_factor=round(profit_factor, 4),
        total_pnl=round(total_pnl, 2),
        max_drawdown_pct=round(max_dd_pct, 2),
        avg_holding_hours=round(avg_holding, 1),
        avg_slippage_r=round(avg_slippage, 4),
        avg_fees_r=round(avg_fees, 4),
        total_funding_r=round(total_funding_r, 4),
        fill_rate=0,  # Placeholder — CONFIRM-context wiring out of Phase-2a scope
        data_source="db_only",
        quality_warnings=_compute_quality_warnings(trades),
    )


def _version_sort_key(item: tuple[str | None, list[float]]) -> tuple:
    """Dotted-numeric version order (`9.11` before `10.00`); `None`/unparseable last.
    The leading group int segregates the three classes so the mixed second-element
    types (int-tuple vs empty tuple) are never compared across groups."""
    v = item[0]
    if v is None:
        return (2, ())
    try:
        return (0, tuple(int(p) for p in v.split(".")))
    except (ValueError, AttributeError):
        return (1, ())


def per_version_expectancy(
    trades: list[TradeRecord], *, min_n: int = 30, seed: int = 0,
) -> tuple[VersionCohort, ...]:
    """Per-prompt-version cohort expectancy (Acceptance #8, SPEC #43).

    Partitions the RISKED trades (`usable_risk`, single-owned per SPEC #26 b — no-risk
    trades have undefined R) by `prompt_version`, and computes the shared per-trade-R
    `expectancy_report` (net of fees per #42) for each cohort. Cohorts are returned
    dotted-numeric-version-sorted, the `(unversioned)` `None` cohort last. Empty input
    (or no risked trades) → empty tuple."""
    risked = risked_outcomes(trades).risked
    by_version: dict[str | None, list[float]] = {}
    for t in risked:
        by_version.setdefault(t.prompt_version, []).append(t.realized_r)
    return tuple(
        VersionCohort(version, expectancy_report(rs, min_n=min_n, seed=seed))
        for version, rs in sorted(by_version.items(), key=_version_sort_key)
    )


def _compute_quality_warnings(trades: list[TradeRecord]) -> tuple[str, ...]:
    """Flag data gaps that affect metric accuracy."""
    warnings = ["All metrics estimated from DB — not reconciled with exchange"]
    if trades:
        no_funding = sum(1 for t in trades if t.funding_paid == 0)
        if 0 < no_funding < len(trades):
            warnings.append(f"{no_funding}/{len(trades)} trades missing funding data")
        elif no_funding == len(trades):
            warnings.append("Funding costs not captured — PnL overstated")
        approx_fills = sum(1 for t in trades if t.entry_fill == 0 or t.exit_fill == 0)
        if approx_fills:
            warnings.append(f"{approx_fills} trades with missing fill prices — using trigger prices")
        no_risk = sum(1 for t in trades if not t.usable_risk)
        if no_risk:
            warnings.append(f"{no_risk} trades with no usable risk anchor — excluded from R metrics")
    return tuple(warnings)


def rolling_expectancy(trades: list[TradeRecord], window: int = 20) -> list[tuple[int, float]]:
    """Sliding window R mean for trend detection.

    Returns list of (trade_index, expectancy) tuples.

    Runtime-dead (C5 audit: "mainly serves tests") — retained unchanged;
    deleting runtime-dead reporting/strategy surface is a separate scope call.
    """
    if len(trades) < window:
        return []

    results = []
    for i in range(window, len(trades) + 1):
        window_trades = trades[i - window:i]
        avg_r = sum(t.realized_r for t in window_trades) / window
        results.append((i, round(avg_r, 4)))

    return results
