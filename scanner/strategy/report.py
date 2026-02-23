"""Strategy validation reports — signal quality, LLM filter value, regime decay."""

import sqlite3
import sys
from dataclasses import dataclass

import pandas as pd

from ..backtest.metrics import MetricsReport, compute_metrics, slice_metrics
from ..backtest.report import format_summary, format_slice_table
from ..backtest.resolver import (
    TradeOutcome,
    TradeSetup,
    build_llm_setup,
    build_screener_setup,
    fetch_and_resolve,
)
from .models import Strategy


# --- Shared helpers ---


def load_strategy_signals(
    strategy_id: str,
    since: str,
    until: str,
    conn: sqlite3.Connection,
    coins: list[str] | None = None,
) -> list[dict]:
    """Load signals from DB for a strategy in a date range."""
    clauses = ["strategy_id = ?", "timestamp >= ?", "timestamp <= ?"]
    params: list = [strategy_id, since, until]

    if coins:
        placeholders = ",".join("?" for _ in coins)
        clauses.append(f"coin IN ({placeholders})")
        params.extend([c.upper() for c in coins])

    where = " AND ".join(clauses)
    rows = conn.execute(
        f"SELECT * FROM signals WHERE {where} ORDER BY timestamp",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def load_llm_scans(
    strategy_id: str,
    since: str,
    until: str,
    conn: sqlite3.Connection,
    coins: list[str] | None = None,
) -> list[dict]:
    """Load LLM scan results from DB for a strategy in a date range."""
    clauses = ["strategy_id = ?", "timestamp >= ?", "timestamp <= ?"]
    params: list = [strategy_id, since, until]

    if coins:
        placeholders = ",".join("?" for _ in coins)
        clauses.append(f"coin IN ({placeholders})")
        params.extend([c.upper() for c in coins])

    where = " AND ".join(clauses)
    rows = conn.execute(
        f"SELECT * FROM scans WHERE {where} ORDER BY timestamp",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def resolve_signals(
    signal_rows: list[dict],
    atr_stop_mult: float,
    atr_target_mult: float,
    conn: sqlite3.Connection,
    timeout_bars: int = 72,
    fee_rate: float = 0.00035,
    verbose: bool = False,
    funding_rate_8h: float = 0.0,
    slippage_rate: float = 0.0,
) -> list[TradeOutcome]:
    """Resolve signal rows into trade outcomes via walk-forward."""
    outcomes: list[TradeOutcome] = []

    n = len(signal_rows)
    for i, row in enumerate(signal_rows):
        setup = build_screener_setup(row, atr_stop_mult, atr_target_mult)
        if not setup:
            continue

        try:
            exchange = row.get("exchange", "hyperliquid")
            outcome = fetch_and_resolve(
                setup, conn, timeout_bars, fee_rate,
                funding_rate_8h=funding_rate_8h,
                slippage_rate=slippage_rate,
                exchange=exchange,
            )
            outcomes.append(outcome)
        except (ValueError, KeyError, IndexError, OSError) as e:
            print(f"  Resolve {setup.coin} {setup.setup_type}: {e}", file=sys.stderr)

        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"  Resolving signals: {i + 1}/{n} ({len(outcomes)} resolved)", file=sys.stderr)

    return outcomes


def resolve_llm_scans(
    scan_rows: list[dict],
    conn: sqlite3.Connection,
    timeout_bars: int = 72,
    fee_rate: float = 0.00035,
    verbose: bool = False,
    funding_rate_8h: float = 0.0,
    slippage_rate: float = 0.0,
) -> list[TradeOutcome]:
    """Resolve LLM scan rows (CONFIRMs only) using LLM's entry/stop/target."""
    outcomes: list[TradeOutcome] = []
    confirms = [r for r in scan_rows if r.get("decision") == "CONFIRM"]

    for i, row in enumerate(confirms):
        setup = build_llm_setup(row)
        if not setup:
            continue

        try:
            exchange = row.get("exchange", "hyperliquid")
            outcome = fetch_and_resolve(
                setup, conn, timeout_bars, fee_rate,
                funding_rate_8h=funding_rate_8h,
                slippage_rate=slippage_rate,
                exchange=exchange,
            )
            outcomes.append(outcome)
        except (ValueError, KeyError, IndexError, OSError) as e:
            print(f"  Resolve LLM {setup.coin}: {e}", file=sys.stderr)

    return outcomes


# --- Shared window iteration ---


def _rolling_window_outcomes(
    strategy: Strategy,
    since: str,
    until: str,
    conn: sqlite3.Connection,
    window_weeks: int = 2,
    coins: list[str] | None = None,
    timeout_bars: int = 72,
    fee_rate: float = 0.00035,
) -> list[tuple[str, str, list[TradeOutcome]]]:
    """Iterate rolling windows, load signals, resolve each.

    Returns [(label, w_end, [TradeOutcome]), ...] per window.
    """
    start = pd.Timestamp(since, tz="UTC")
    end = pd.Timestamp(until, tz="UTC")
    delta = pd.Timedelta(weeks=window_weeks)

    windows: list[tuple[str, str, list[TradeOutcome]]] = []
    cursor = start

    while cursor + delta <= end:
        w_start = cursor.strftime("%Y-%m-%d")
        w_end = (cursor + delta).strftime("%Y-%m-%d")

        signal_rows = load_strategy_signals(
            strategy.strategy_id, w_start, w_end, conn, coins,
        )

        if signal_rows:
            outcomes = resolve_signals(
                signal_rows, strategy.atr_stop_mult, strategy.atr_target_mult,
                conn, timeout_bars, fee_rate, verbose=False,
            )
        else:
            outcomes = []

        windows.append((w_start, w_end, outcomes))
        cursor += delta

    return windows


# --- Loop 1: Signal Quality ---


def signal_quality_report(
    strategy: Strategy,
    since: str,
    until: str,
    conn: sqlite3.Connection,
    coins: list[str] | None = None,
    timeout_bars: int = 72,
    fee_rate: float = 0.00035,
    verbose: bool = False,
    run_permutation: bool = False,
) -> tuple[MetricsReport, dict[str, list[MetricsReport]]]:
    """Per-setup-type breakdown. Which of A-G produce edge?

    Returns (overall_metrics, sliced_metrics_dict).
    """
    signal_rows = load_strategy_signals(
        strategy.strategy_id, since, until, conn, coins,
    )

    if verbose:
        print(f"Signal quality: {len(signal_rows)} signals loaded")

    outcomes = resolve_signals(
        signal_rows, strategy.atr_stop_mult, strategy.atr_target_mult,
        conn, timeout_bars, fee_rate, verbose,
    )

    overall = compute_metrics(outcomes, f"strategy:{strategy.strategy_id}", run_permutation=run_permutation)
    sliced = slice_metrics(outcomes)

    return overall, sliced


# --- Loop 2: LLM Filter Value ---


@dataclass
class LLMFilterResult:
    """Comparison of screener-all vs sent-to-LLM vs CONFIRM-only performance."""

    screener_all: MetricsReport       # All signals, ATR resolution
    screener_sent: MetricsReport      # sent_to_llm=1, ATR resolution
    llm_confirmed: MetricsReport      # CONFIRMs, LLM's own entry/stop/target
    total_signals: int
    sent_to_llm_count: int
    confirm_count: int
    reject_count: int
    wait_count: int
    total_cost: float
    confirm_rate: float
    cost_per_confirm: float


def llm_filter_report(
    strategy: Strategy,
    since: str,
    until: str,
    conn: sqlite3.Connection,
    coins: list[str] | None = None,
    timeout_bars: int = 72,
    fee_rate: float = 0.00035,
    verbose: bool = False,
) -> LLMFilterResult:
    """Is the LLM adding alpha? Compare all vs sent vs CONFIRM expectancy."""
    signal_rows = load_strategy_signals(
        strategy.strategy_id, since, until, conn, coins,
    )
    scan_rows = load_llm_scans(
        strategy.strategy_id, since, until, conn, coins,
    )

    if verbose:
        print(f"LLM filter: {len(signal_rows)} signals, {len(scan_rows)} scans")

    # Bucket 1: All screener signals
    all_outcomes = resolve_signals(
        signal_rows, strategy.atr_stop_mult, strategy.atr_target_mult,
        conn, timeout_bars, fee_rate, verbose,
    )
    screener_all = compute_metrics(all_outcomes, "all_signals")

    # Bucket 2: Only signals sent to LLM
    sent_rows = [r for r in signal_rows if r.get("sent_to_llm")]
    sent_outcomes = resolve_signals(
        sent_rows, strategy.atr_stop_mult, strategy.atr_target_mult,
        conn, timeout_bars, fee_rate, verbose,
    )
    screener_sent = compute_metrics(sent_outcomes, "sent_to_llm")

    # Bucket 3: CONFIRM scans with LLM's own levels
    llm_outcomes = resolve_llm_scans(
        scan_rows, conn, timeout_bars, fee_rate, verbose,
    )
    llm_confirmed = compute_metrics(llm_outcomes, "llm_confirmed")

    # Funnel stats
    total_signals = len(signal_rows)
    sent_count = len(sent_rows)
    confirms = [r for r in scan_rows if r.get("decision") == "CONFIRM"]
    rejects = [r for r in scan_rows if r.get("decision") == "REJECT"]
    waits = [r for r in scan_rows if r.get("decision") == "WAIT"]
    total_cost = sum(r.get("cost_usd", 0) or 0 for r in scan_rows)

    confirm_rate = len(confirms) / sent_count if sent_count > 0 else 0.0
    cost_per_confirm = total_cost / len(confirms) if confirms else 0.0

    return LLMFilterResult(
        screener_all=screener_all,
        screener_sent=screener_sent,
        llm_confirmed=llm_confirmed,
        total_signals=total_signals,
        sent_to_llm_count=sent_count,
        confirm_count=len(confirms),
        reject_count=len(rejects),
        wait_count=len(waits),
        total_cost=round(total_cost, 4),
        confirm_rate=round(confirm_rate, 4),
        cost_per_confirm=round(cost_per_confirm, 4),
    )


# --- Loop 3: Regime Decay ---


def regime_decay_report(
    strategy: Strategy,
    since: str,
    until: str,
    conn: sqlite3.Connection,
    window_weeks: int = 2,
    coins: list[str] | None = None,
    timeout_bars: int = 72,
    fee_rate: float = 0.00035,
    verbose: bool = False,
) -> list[tuple[str, MetricsReport]]:
    """Rolling N-week window expectancy trend.

    Returns [(window_label, MetricsReport), ...] for each window.
    """
    start = pd.Timestamp(since, tz="UTC")
    end = pd.Timestamp(until, tz="UTC")
    delta = pd.Timedelta(weeks=window_weeks)

    windows: list[tuple[str, MetricsReport]] = []
    cursor = start

    while cursor + delta <= end:
        w_start = cursor.strftime("%Y-%m-%d")
        w_end = (cursor + delta).strftime("%Y-%m-%d")
        label = f"{w_start}"

        signal_rows = load_strategy_signals(
            strategy.strategy_id, w_start, w_end, conn, coins,
        )

        if signal_rows:
            outcomes = resolve_signals(
                signal_rows, strategy.atr_stop_mult, strategy.atr_target_mult,
                conn, timeout_bars, fee_rate, verbose=False,
            )
            metrics = compute_metrics(outcomes, label)
        else:
            metrics = compute_metrics([], label)

        windows.append((label, metrics))

        if verbose:
            arrow = "^" if metrics.expectancy > 0 else "v" if metrics.expectancy < 0 else "="
            print(f"  {label}: n={metrics.sample_size} exp={metrics.expectancy:+.3f}R {arrow}")

        cursor += delta

    return windows


# --- Decay metric 1: Regime-conditioned rolling expectancy ---


def regime_conditioned_decay_report(
    strategy: Strategy,
    since: str,
    until: str,
    conn: sqlite3.Connection,
    window_weeks: int = 2,
    coins: list[str] | None = None,
    timeout_bars: int = 72,
    fee_rate: float = 0.00035,
    verbose: bool = False,
) -> list[tuple[str, dict[str, MetricsReport]]]:
    """Rolling expectancy broken out by regime within each window.

    Returns [(window_label, {regime: MetricsReport}), ...].
    Declining expectancy within a *single regime* across windows → alpha decay.
    Declining overall because regime mix shifted → regime change, not decay.
    """
    raw_windows = _rolling_window_outcomes(
        strategy, since, until, conn, window_weeks, coins, timeout_bars, fee_rate,
    )

    result: list[tuple[str, dict[str, MetricsReport]]] = []
    for label, _, outcomes in raw_windows:
        # Group outcomes by regime
        by_regime: dict[str, list[TradeOutcome]] = {}
        for o in outcomes:
            regime = o.setup.regime if (o.setup and o.setup.regime) else "unknown"
            by_regime.setdefault(regime, []).append(o)

        metrics_by_regime = {
            regime: compute_metrics(outs, f"{label}:{regime}")
            for regime, outs in sorted(by_regime.items())
        }

        if verbose:
            parts = [f"{r}={m.expectancy:+.3f}" for r, m in metrics_by_regime.items()]
            print(f"  {label}: {', '.join(parts)}")

        result.append((label, metrics_by_regime))

    return result


# --- Decay metric 2: LLM filter alpha delta ---


@dataclass
class LLMAlphaDeltaWindow:
    """Per-window comparison of all-signal vs CONFIRM expectancy."""

    label: str
    n_all: int
    n_confirmed: int
    exp_all: float
    exp_confirmed: float
    delta: float | None  # None when n_confirmed == 0


def llm_alpha_decay_report(
    strategy: Strategy,
    since: str,
    until: str,
    conn: sqlite3.Connection,
    window_weeks: int = 2,
    coins: list[str] | None = None,
    timeout_bars: int = 72,
    fee_rate: float = 0.00035,
    verbose: bool = False,
) -> list[LLMAlphaDeltaWindow]:
    """Delta = E[CONFIRM] - E[all_signals] in rolling windows.

    Delta declining toward zero → LLM filter losing edge.
    """
    start = pd.Timestamp(since, tz="UTC")
    end = pd.Timestamp(until, tz="UTC")
    delta_td = pd.Timedelta(weeks=window_weeks)

    windows: list[LLMAlphaDeltaWindow] = []
    cursor = start

    while cursor + delta_td <= end:
        w_start = cursor.strftime("%Y-%m-%d")
        w_end = (cursor + delta_td).strftime("%Y-%m-%d")

        signal_rows = load_strategy_signals(
            strategy.strategy_id, w_start, w_end, conn, coins,
        )
        scan_rows = load_llm_scans(
            strategy.strategy_id, w_start, w_end, conn, coins,
        )

        # All signals expectancy
        if signal_rows:
            all_outcomes = resolve_signals(
                signal_rows, strategy.atr_stop_mult, strategy.atr_target_mult,
                conn, timeout_bars, fee_rate, verbose=False,
            )
            all_metrics = compute_metrics(all_outcomes, "all")
        else:
            all_metrics = compute_metrics([], "all")

        # CONFIRM expectancy (using LLM's own levels)
        confirm_outcomes = resolve_llm_scans(
            scan_rows, conn, timeout_bars, fee_rate, verbose=False,
        )
        if confirm_outcomes:
            conf_metrics = compute_metrics(confirm_outcomes, "confirmed")
            delta_val = conf_metrics.expectancy - all_metrics.expectancy
        else:
            conf_metrics = compute_metrics([], "confirmed")
            delta_val = None

        window = LLMAlphaDeltaWindow(
            label=w_start,
            n_all=all_metrics.sample_size,
            n_confirmed=conf_metrics.sample_size,
            exp_all=all_metrics.expectancy,
            exp_confirmed=conf_metrics.expectancy,
            delta=round(delta_val, 4) if delta_val is not None else None,
        )
        windows.append(window)

        if verbose:
            d = f"{delta_val:+.3f}" if delta_val is not None else "n/a"
            print(f"  {w_start}: all={all_metrics.expectancy:+.3f} conf={conf_metrics.expectancy:+.3f} Δ={d}")

        cursor += delta_td

    return windows


# --- Decay metric 3: Time-to-resolution drift ---


def resolution_drift_report(
    strategy: Strategy,
    since: str,
    until: str,
    conn: sqlite3.Connection,
    window_weeks: int = 2,
    coins: list[str] | None = None,
    timeout_bars: int = 72,
    fee_rate: float = 0.00035,
    verbose: bool = False,
) -> list[tuple[str, MetricsReport]]:
    """Track avg_bars_to_exit in rolling windows.

    Rising bars → setups losing urgency (crowding at same levels).
    """
    raw_windows = _rolling_window_outcomes(
        strategy, since, until, conn, window_weeks, coins, timeout_bars, fee_rate,
    )

    result: list[tuple[str, MetricsReport]] = []
    for label, _, outcomes in raw_windows:
        metrics = compute_metrics(outcomes, label)
        result.append((label, metrics))

        if verbose:
            print(f"  {label}: n={metrics.sample_size} avg_bars={metrics.avg_bars_to_exit:.1f}")

    return result


# --- Auto-disable suggestions ---


def suggest_disabled_types(
    setup_type_reports: list[MetricsReport],
    min_sample: int = 30,
) -> list[tuple[str, float, int, str]]:
    """Returns setup types with negative expectancy and statistical confidence.

    Requires both:
    - n >= min_sample (CLT minimum)
    - Bootstrap 95% CI upper bound < 0 (can't be explained by noise)

    Returns [(setup_type, expectancy, sample_size, ci_note), ...] sorted worst-first.
    """
    suggestions = []
    for report in setup_type_reports:
        if report.sample_size < min_sample:
            continue
        if report.expectancy >= 0:
            continue

        ci = report.bootstrap_ci
        if ci and ci.excludes_zero and ci.ci_high < 0:
            ci_note = f"95% CI [{ci.ci_low:+.3f}, {ci.ci_high:+.3f}]"
            suggestions.append((report.label, report.expectancy, report.sample_size, ci_note))

    suggestions.sort(key=lambda x: x[1])  # worst first
    return suggestions


# --- Formatting ---


def format_signal_quality(
    overall: MetricsReport,
    sliced: dict[str, list[MetricsReport]],
) -> str:
    """Format signal quality report (Loop 1)."""
    sections = [
        "=" * 60,
        "  SIGNAL QUALITY REPORT",
        "=" * 60,
        "",
        "OVERALL",
        format_summary(overall),
    ]

    for dim_name, reports in sliced.items():
        sections.append("")
        sections.append(f"BY {dim_name.upper()}")
        sections.append(format_slice_table(dim_name, reports))

    sections.append("")
    sections.append("=" * 60)
    return "\n".join(sections)


def format_llm_filter(result: LLMFilterResult) -> str:
    """Format LLM filter value report (Loop 2)."""
    sections = [
        "=" * 60,
        "  LLM FILTER VALUE REPORT",
        "=" * 60,
        "",
        "FUNNEL",
        f"  Total signals:     {result.total_signals}",
        f"  Sent to LLM:       {result.sent_to_llm_count}",
        f"  CONFIRM:           {result.confirm_count}",
        f"  REJECT:            {result.reject_count}",
        f"  WAIT:              {result.wait_count}",
        f"  Confirm rate:      {result.confirm_rate:.1%}",
        f"  Total cost:        ${result.total_cost:.2f}",
        f"  Cost/confirm:      ${result.cost_per_confirm:.2f}",
    ]

    # 3-column comparison
    sections.append("")
    sections.append("EXPECTANCY COMPARISON")
    header = f"  {'Bucket':<18} {'N':>5} {'WR':>7} {'ExpR':>8} {'PF':>7} {'TotR':>8}"
    sep = "  " + "-" * len(header.strip())
    sections.append(header)
    sections.append(sep)

    for report in [result.screener_all, result.screener_sent, result.llm_confirmed]:
        pf = f"{report.profit_factor:.2f}" if report.profit_factor != float("inf") else "inf"
        sections.append(
            f"  {report.label:<18} {report.sample_size:>5} {report.win_rate:>6.1%} "
            f"{report.expectancy:>+7.3f}R {pf:>6} {report.total_r:>+7.2f}"
        )

    sections.append("")
    sections.append("=" * 60)
    return "\n".join(sections)


def format_regime_decay(windows: list[tuple[str, MetricsReport]]) -> str:
    """Format regime decay report (Loop 3)."""
    sections = [
        "=" * 60,
        "  REGIME DECAY REPORT",
        "=" * 60,
        "",
    ]

    header = f"  {'Window':<12} {'N':>5} {'WR':>7} {'ExpR':>8} {'PF':>7} {'TotR':>8} {'Trend':>5}"
    sep = "  " + "-" * len(header.strip())
    sections.append(header)
    sections.append(sep)

    prev_exp = None
    for label, report in windows:
        pf = f"{report.profit_factor:.2f}" if report.profit_factor != float("inf") else "inf"

        if prev_exp is None:
            arrow = " "
        elif report.expectancy > prev_exp + 0.01:
            arrow = "^"
        elif report.expectancy < prev_exp - 0.01:
            arrow = "v"
        else:
            arrow = "="

        sections.append(
            f"  {label:<12} {report.sample_size:>5} {report.win_rate:>6.1%} "
            f"{report.expectancy:>+7.3f}R {pf:>6} {report.total_r:>+7.2f}   {arrow}"
        )
        prev_exp = report.expectancy

    sections.append("")
    sections.append("=" * 60)
    return "\n".join(sections)


def regime_crosstab_report(
    outcomes: list[TradeOutcome],
) -> dict[str, dict[str, MetricsReport]]:
    """2D grid: rows = setup types, columns = regimes.

    Returns {setup_type: {regime: MetricsReport}}.
    """
    grid: dict[str, dict[str, list[TradeOutcome]]] = {}

    for o in outcomes:
        st = o.setup.setup_type if o.setup else "unknown"
        regime = o.setup.regime if (o.setup and o.setup.regime) else "unknown"
        grid.setdefault(st, {}).setdefault(regime, []).append(o)

    result: dict[str, dict[str, MetricsReport]] = {}
    for st in sorted(grid.keys()):
        result[st] = {}
        for regime in sorted(grid[st].keys()):
            result[st][regime] = compute_metrics(grid[st][regime], f"{st}:{regime}")

    return result


def format_regime_crosstab(crosstab: dict[str, dict[str, MetricsReport]]) -> str:
    """Format setup_type x regime grid showing N, WR, ExpR per cell."""
    if not crosstab:
        return "No data for regime cross-tab."

    # Collect all regimes across all setup types
    all_regimes = sorted({r for regimes in crosstab.values() for r in regimes})

    sections = [
        "=" * 60,
        "  SETUP TYPE x REGIME CROSS-TAB",
        "=" * 60,
        "",
    ]

    # Header
    col_w = 22
    header = f"  {'Type':<6}"
    for regime in all_regimes:
        header += f" {regime:>{col_w}}"
    sections.append(header)
    sections.append("  " + "-" * (6 + (col_w + 1) * len(all_regimes)))

    # Rows
    for st in sorted(crosstab.keys()):
        row = f"  {st:<6}"
        for regime in all_regimes:
            m = crosstab[st].get(regime)
            if m and m.sample_size > 0:
                cell = f"n={m.sample_size} {m.win_rate:.0%} {m.expectancy:+.2f}R"
            else:
                cell = "-"
            row += f" {cell:>{col_w}}"
        sections.append(row)

    sections.append("")
    sections.append("=" * 60)
    return "\n".join(sections)


def format_regime_conditioned_decay(
    windows: list[tuple[str, dict[str, MetricsReport]]],
) -> str:
    """Format regime-conditioned decay report.

    Table: rows = windows, columns = all regimes seen (union across windows).
    """
    sections = [
        "=" * 60,
        "  REGIME-CONDITIONED DECAY",
        "=" * 60,
        "",
    ]

    if not windows:
        sections.append("  No data.")
        sections.append("")
        sections.append("=" * 60)
        return "\n".join(sections)

    # Collect all regimes across all windows
    all_regimes = sorted({r for _, regime_map in windows for r in regime_map})

    col_w = 16
    header = f"  {'Window':<12}"
    for regime in all_regimes:
        header += f" {regime:>{col_w}}"
    sections.append(header)
    sections.append("  " + "-" * (12 + (col_w + 1) * len(all_regimes)))

    for label, regime_map in windows:
        row = f"  {label:<12}"
        for regime in all_regimes:
            m = regime_map.get(regime)
            if m and m.sample_size > 0:
                cell = f"n={m.sample_size} {m.expectancy:+.3f}R"
            else:
                cell = "-"
            row += f" {cell:>{col_w}}"
        sections.append(row)

    sections.append("")
    sections.append("=" * 60)
    return "\n".join(sections)


def format_llm_alpha_decay(windows: list[LLMAlphaDeltaWindow]) -> str:
    """Format LLM alpha delta decay report."""
    sections = [
        "=" * 60,
        "  LLM FILTER ALPHA DECAY",
        "=" * 60,
        "",
    ]

    header = (
        f"  {'Window':<12} {'N_all':>6} {'N_conf':>7} "
        f"{'ExpAll':>8} {'ExpConf':>8} {'Delta':>8} {'Trend':>5}"
    )
    sep = "  " + "-" * len(header.strip())
    sections.append(header)
    sections.append(sep)

    prev_delta = None
    for w in windows:
        if w.delta is not None:
            delta_str = f"{w.delta:>+7.3f}R"
        else:
            delta_str = f"{'n/a':>8}"

        if prev_delta is None or w.delta is None:
            arrow = " "
        elif w.delta > prev_delta + 0.01:
            arrow = "^"
        elif w.delta < prev_delta - 0.01:
            arrow = "v"
        else:
            arrow = "="

        sections.append(
            f"  {w.label:<12} {w.n_all:>6} {w.n_confirmed:>7} "
            f"{w.exp_all:>+7.3f}R {w.exp_confirmed:>+7.3f}R {delta_str}   {arrow}"
        )
        if w.delta is not None:
            prev_delta = w.delta

    sections.append("")
    sections.append("=" * 60)
    return "\n".join(sections)


def format_resolution_drift(windows: list[tuple[str, MetricsReport]]) -> str:
    """Format resolution drift report — avg_bars_to_exit trend."""
    sections = [
        "=" * 60,
        "  RESOLUTION DRIFT",
        "=" * 60,
        "",
    ]

    header = f"  {'Window':<12} {'N':>5} {'WR':>7} {'ExpR':>8} {'AvgBars':>8} {'Trend':>5}"
    sep = "  " + "-" * len(header.strip())
    sections.append(header)
    sections.append(sep)

    prev_bars = None
    for label, report in windows:
        if prev_bars is None:
            arrow = " "
        elif report.avg_bars_to_exit > prev_bars + 1.0:
            arrow = "^"  # increasing = warning
        elif report.avg_bars_to_exit < prev_bars - 1.0:
            arrow = "v"  # decreasing = good
        else:
            arrow = "="

        sections.append(
            f"  {label:<12} {report.sample_size:>5} {report.win_rate:>6.1%} "
            f"{report.expectancy:>+7.3f}R {report.avg_bars_to_exit:>7.1f}   {arrow}"
        )
        prev_bars = report.avg_bars_to_exit

    sections.append("")
    sections.append("=" * 60)
    return "\n".join(sections)


# --- Discord Embeds ---

COLOR_POS = 0x26A69A   # green
COLOR_NEG = 0xEF5350   # red
COLOR_WARN = 0xFFA726  # orange


def _ci_verdict(report: MetricsReport) -> str:
    """One-word CI verdict."""
    ci = report.bootstrap_ci
    if not ci:
        return "n/a" if report.sample_size < 5 else ""
    if ci.excludes_zero and ci.ci_low > 0:
        return "edge"
    if ci.excludes_zero:
        return "negative"
    return "inconclusive"


def build_signal_quality_embeds(
    overall: MetricsReport,
    sliced: dict[str, list[MetricsReport]],
) -> list[dict]:
    """Build Discord embeds for signal quality report."""
    from datetime import datetime, timezone

    color = COLOR_POS if overall.expectancy >= 0 else COLOR_NEG
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    pf = f"{overall.profit_factor:.2f}" if overall.profit_factor != float("inf") else "inf"
    overall_embed = {
        "title": "Signal Quality: Overall",
        "color": color,
        "fields": [
            {"name": "Trades", "value": str(overall.sample_size), "inline": True},
            {"name": "W/L/E", "value": f"{overall.wins}/{overall.losses}/{overall.expired}", "inline": True},
            {"name": "Win Rate", "value": f"{overall.win_rate:.1%}", "inline": True},
            {"name": "Expectancy", "value": f"{overall.expectancy:+.3f}R", "inline": True},
            {"name": "Total R", "value": f"{overall.total_r:+.1f}", "inline": True},
            {"name": "PF", "value": pf, "inline": True},
            {"name": "95% CI", "value": _ci_verdict(overall), "inline": True},
        ],
        "footer": {"text": timestamp},
    }

    embeds = [overall_embed]

    type_reports = sliced.get("setup_type", [])
    if type_reports:
        type_lines = []
        for r in type_reports:
            type_lines.append(
                f"**{r.label}**: {r.sample_size} trades | {r.win_rate:.0%} WR | {r.expectancy:+.2f}R"
            )
        embeds.append({
            "title": "Signal Quality: By Setup Type",
            "color": color,
            "description": "\n".join(type_lines),
            "footer": {"text": timestamp},
        })

    return embeds


def build_llm_filter_embeds(result: "LLMFilterResult") -> list[dict]:
    """Build Discord embeds for LLM filter report."""
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Color based on whether LLM confirms outperform all signals
    llm_better = result.llm_confirmed.expectancy > result.screener_all.expectancy
    color = COLOR_POS if llm_better else COLOR_NEG

    # Comparison lines
    def _bucket_line(label: str, r: MetricsReport) -> str:
        return f"{label:<16} {r.sample_size:>3} trades | {r.win_rate:.0%} WR | {r.expectancy:+.2f}R"

    description = "\n".join([
        f"```",
        _bucket_line("All signals:", result.screener_all),
        _bucket_line("Sent to LLM:", result.screener_sent),
        _bucket_line("LLM Confirmed:", result.llm_confirmed),
        f"```",
    ])

    embed = {
        "title": "LLM Filter Value",
        "color": color,
        "fields": [
            {"name": "Total Signals", "value": str(result.total_signals), "inline": True},
            {"name": "Sent to LLM", "value": str(result.sent_to_llm_count), "inline": True},
            {"name": "CONFIRMs", "value": str(result.confirm_count), "inline": True},
            {"name": "Confirm Rate", "value": f"{result.confirm_rate:.1%}", "inline": True},
            {"name": "Cost", "value": f"${result.total_cost:.2f}", "inline": True},
            {"name": "Cost/Confirm", "value": f"${result.cost_per_confirm:.2f}", "inline": True},
        ],
        "description": description,
        "footer": {"text": timestamp},
    }

    return [embed]


def build_disable_embeds(
    suggestions: list[tuple[str, float, int, str]],
    strategy_id: str = "v0",
) -> list[dict]:
    """Build Discord embed for auto-disable suggestions."""
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if not suggestions:
        return [{
            "title": "Auto-Disable: Clean",
            "color": COLOR_POS,
            "description": "No types qualify for disabling",
            "footer": {"text": timestamp},
        }]

    lines = []
    for setup_type, exp, n, ci_note in suggestions:
        lines.append(f"**{setup_type}**: exp={exp:+.3f}R (n={n}) {ci_note}")

    types_str = ",".join(s[0] for s in suggestions)
    lines.append(f"\n`python -m scanner.strategy disable {strategy_id} {types_str}`")

    return [{
        "title": "Auto-Disable: Suggestions",
        "color": COLOR_WARN,
        "description": "\n".join(lines),
        "footer": {"text": timestamp},
    }]


def format_disable_suggestion(suggestions: list[tuple[str, float, int, str]]) -> str:
    """Format auto-disable suggestions."""
    if not suggestions:
        return "No setup types qualify for disabling (all positive, insufficient sample, or CI includes zero)."

    lines = ["SUGGESTED DISABLES (negative expectancy, n >= 30, 95% CI excludes zero):", ""]
    for setup_type, exp, n, ci_note in suggestions:
        lines.append(f"  {setup_type}: expectancy={exp:+.4f}R  n={n}  {ci_note}")

    types_str = ",".join(s[0] for s in suggestions)
    lines.append("")
    lines.append(f"  Run: python -m scanner.strategy disable <strategy_id> {types_str}")

    return "\n".join(lines)


# --- Roadmap: Future Decay Metrics ---
#
# These metrics require more data or external sources before implementation.
# Documented here with triggers for when to build.
#
# 1. S/R Level Effectiveness Rate
#    Track hit rate of key_level prices — how often does price react at our levels?
#    Declining effectiveness = levels becoming crowded/invalidated.
#    Trigger: 6+ months of level-tagged data in signals table.
#
# 2. Setup Type Half-Life
#    For each setup type, measure time from first appearance to when expectancy
#    decays to zero. Identifies which edges are structural vs ephemeral.
#    Trigger: 6+ months of continuous data per setup type.
#
# 3. External Crowding Signals
#    Monitor open interest concentration, funding rate extremes, or social
#    mention frequency at our identified levels. Rising crowding = edge erosion.
#    Trigger: Manual periodic review; requires external data source integration.
