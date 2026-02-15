"""Strategy validation reports — signal quality, LLM filter value, regime decay."""

import sqlite3
from dataclasses import dataclass

import pandas as pd

from ..backtest.candles import fetch_and_cache
from ..backtest.metrics import MetricsReport, compute_metrics, slice_metrics
from ..backtest.report import format_summary, format_slice_table
from ..backtest.resolver import (
    TradeOutcome,
    TradeSetup,
    build_llm_setup,
    build_screener_setup,
    resolve_signal,
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
    timeout_bars: int = 168,
    fee_rate: float = 0.00035,
    verbose: bool = False,
) -> list[TradeOutcome]:
    """Resolve signal rows into trade outcomes via walk-forward."""
    outcomes: list[TradeOutcome] = []

    for i, row in enumerate(signal_rows):
        setup = build_screener_setup(row, atr_stop_mult, atr_target_mult)
        if not setup:
            continue

        try:
            until_dt = (
                pd.Timestamp(setup.timestamp, tz="UTC")
                + pd.Timedelta(hours=timeout_bars + 1)
            ).isoformat()
            candles = fetch_and_cache(setup.coin, "1h", setup.timestamp, until_dt, conn)
            outcome = resolve_signal(setup, candles, timeout_bars, fee_rate)
            outcomes.append(outcome)
        except Exception as e:
            if verbose:
                print(f"  Resolve {setup.coin} {setup.setup_type}: {e}")

        if verbose and (i + 1) % 20 == 0:
            print(f"  Resolved {i + 1}/{len(signal_rows)}")

    return outcomes


def resolve_llm_scans(
    scan_rows: list[dict],
    conn: sqlite3.Connection,
    timeout_bars: int = 168,
    fee_rate: float = 0.00035,
    verbose: bool = False,
) -> list[TradeOutcome]:
    """Resolve LLM scan rows (CONFIRMs only) using LLM's entry/stop/target."""
    outcomes: list[TradeOutcome] = []
    confirms = [r for r in scan_rows if r.get("decision") == "CONFIRM"]

    for i, row in enumerate(confirms):
        setup = build_llm_setup(row)
        if not setup:
            continue

        try:
            until_dt = (
                pd.Timestamp(setup.timestamp, tz="UTC")
                + pd.Timedelta(hours=timeout_bars + 1)
            ).isoformat()
            candles = fetch_and_cache(setup.coin, "1h", setup.timestamp, until_dt, conn)
            outcome = resolve_signal(setup, candles, timeout_bars, fee_rate)
            outcomes.append(outcome)
        except Exception as e:
            if verbose:
                print(f"  Resolve LLM {setup.coin}: {e}")

    return outcomes


# --- Loop 1: Signal Quality ---


def signal_quality_report(
    strategy: Strategy,
    since: str,
    until: str,
    conn: sqlite3.Connection,
    coins: list[str] | None = None,
    timeout_bars: int = 168,
    fee_rate: float = 0.00035,
    verbose: bool = False,
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

    overall = compute_metrics(outcomes, f"strategy:{strategy.strategy_id}")
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
    timeout_bars: int = 168,
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
    timeout_bars: int = 168,
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


# --- Auto-disable suggestions ---


def suggest_disabled_types(
    setup_type_reports: list[MetricsReport],
    min_sample: int = 10,
) -> list[tuple[str, float, int]]:
    """Returns setup types with negative expectancy and sufficient sample.

    Returns [(setup_type, expectancy, sample_size), ...] sorted worst-first.
    """
    suggestions = []
    for report in setup_type_reports:
        if report.sample_size >= min_sample and report.expectancy < 0:
            suggestions.append((report.label, report.expectancy, report.sample_size))

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


def format_disable_suggestion(suggestions: list[tuple[str, float, int]]) -> str:
    """Format auto-disable suggestions."""
    if not suggestions:
        return "No setup types qualify for disabling (all positive or insufficient sample)."

    lines = ["SUGGESTED DISABLES (negative expectancy, n >= min_sample):", ""]
    for setup_type, exp, n in suggestions:
        lines.append(f"  {setup_type}: expectancy={exp:+.4f}R  n={n}")

    types_str = ",".join(s[0] for s in suggestions)
    lines.append("")
    lines.append(f"  Run: python -m scanner.strategy disable <strategy_id> {types_str}")

    return "\n".join(lines)
