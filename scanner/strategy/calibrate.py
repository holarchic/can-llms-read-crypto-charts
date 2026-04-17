"""Scoring calibration analysis — measures predictive power of confidence components.

Run periodically to track whether scoring factors predict wins.
Designed to accumulate data across market regimes for eventual scoring redesign.
"""

import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import ccxt

from ..backtest.resolver import build_screener_setup, fetch_and_resolve


@dataclass
class CalibrationResult:
    """Result of a calibration analysis run."""
    n_signals: int
    n_resolved: int
    overall_wr: float
    overall_avg_r: float

    # Correlation: composite confidence vs win
    conf_win_r: float
    conf_r_r: float

    # Per-component correlations (None if no component data)
    component_correlations: dict[str, float]  # component_name -> pearson r vs win

    # Quartile stats: list of (label, n, wr, avg_r)
    quartiles: list[tuple[str, int, float, float]]

    # Per-type stats: list of (type, n, wr, avg_r, conf_win_r)
    per_type: list[tuple[str, int, float, float, float]]

    # BTC regime breakdown: list of (regime, n, wr, avg_r)
    by_btc_regime: list[tuple[str, int, float, float]]



def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = sum((xi - mx) ** 2 for xi in x) ** 0.5
    dy = sum((yi - my) ** 2 for yi in y) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _stats(outcomes: list[dict]) -> tuple[float, float]:
    """Return (win_rate, avg_r) for a list of outcomes."""
    if not outcomes:
        return 0.0, 0.0
    n = len(outcomes)
    wr = sum(1 for o in outcomes if o["result"] == "WIN") / n
    avg_r = sum(o["r_achieved"] for o in outcomes) / n
    return wr, avg_r


def _resolve_calibration_signals(
    signals: list,
    conn: sqlite3.Connection,
    timeout_bars: int,
) -> list[dict]:
    """Resolve signals and build outcome dicts with score components."""
    outcomes = []
    for sig in signals:
        sig = dict(sig)  # sqlite3.Row → dict for .get() access
        setup = build_screener_setup(sig)
        if not setup:
            continue

        try:
            exchange = sig.get("exchange", "hyperliquid")
            outcome = fetch_and_resolve(
                setup, conn, timeout_bars, exchange=exchange,
            )
        except (ValueError, KeyError, IndexError, OSError, ccxt.NetworkError, ccxt.ExchangeError) as e:
            print(f"  Resolve {sig['coin']}: {e}", file=sys.stderr)
            continue

        if outcome.result == "EXPIRED" and outcome.bars_to_exit == 0:
            continue
        outcomes.append({
            "coin": sig["coin"],
            "setup_type": sig["setup_type"],
            "direction": sig["direction"],
            "confidence": sig["confidence"],
            "structure": sig["structure"],
            "sent_to_llm": bool(sig["sent_to_llm"]),
            "result": outcome.result,
            "r_achieved": outcome.r_achieved,
            "mfe": outcome.mfe,
            "score_base": sig["score_base"],
            "score_alignment": sig["score_alignment"],
            "score_strength": sig["score_strength"],
            "score_proximity": sig["score_proximity"],
            "score_confluence": sig["score_confluence"],
            "btc_structure": sig["btc_structure"],
        })
    return outcomes


def _analyze_outcomes(outcomes: list[dict]) -> dict:
    """Compute correlations, quartiles, per-type, and regime breakdowns."""
    wr, avg_r = _stats(outcomes)

    # Confidence vs outcomes
    conf_vals = [o["confidence"] for o in outcomes]
    win_vals = [1.0 if o["result"] == "WIN" else 0.0 for o in outcomes]
    r_vals = [o["r_achieved"] for o in outcomes]

    # Per-component correlations
    component_names = ["score_base", "score_alignment", "score_strength",
                       "score_proximity", "score_confluence"]
    component_correlations = {}
    for comp in component_names:
        has_data = [o for o in outcomes if o[comp] is not None]
        if len(has_data) >= 10:
            c_vals = [o[comp] for o in has_data]
            c_wins = [1.0 if o["result"] == "WIN" else 0.0 for o in has_data]
            short_name = comp.replace("score_", "")
            component_correlations[short_name] = round(_pearson(c_vals, c_wins), 4)

    # Quartiles
    confs = sorted(o["confidence"] for o in outcomes)
    n = len(confs)
    q1, q2, q3 = confs[n // 4], confs[n // 2], confs[3 * n // 4]
    quartile_defs = [
        (f"Q1 (≤{q1:.3f})", lambda o: o["confidence"] <= q1),
        (f"Q2 ({q1:.3f}-{q2:.3f}]", lambda o: q1 < o["confidence"] <= q2),
        (f"Q3 ({q2:.3f}-{q3:.3f}]", lambda o: q2 < o["confidence"] <= q3),
        (f"Q4 (>{q3:.3f})", lambda o: o["confidence"] > q3),
    ]
    quartiles = []
    for label, pred in quartile_defs:
        bucket = [o for o in outcomes if pred(o)]
        bwr, bavg = _stats(bucket)
        quartiles.append((label, len(bucket), bwr, bavg))

    # Per-type
    types = sorted(set(o["setup_type"] for o in outcomes))
    per_type = []
    for t in types:
        t_rows = [o for o in outcomes if o["setup_type"] == t]
        if len(t_rows) < 3:
            continue
        twr, tavg = _stats(t_rows)
        t_conf = [o["confidence"] for o in t_rows]
        t_win = [1.0 if o["result"] == "WIN" else 0.0 for o in t_rows]
        t_corr = _pearson(t_conf, t_win) if len(t_rows) >= 5 else 0.0
        per_type.append((t, len(t_rows), twr, tavg, t_corr))

    # By BTC regime
    by_btc_regime = []
    regimes = sorted(set(o["btc_structure"] for o in outcomes if o["btc_structure"]))
    for regime in regimes:
        r_rows = [o for o in outcomes if o["btc_structure"] == regime]
        rwr, ravg = _stats(r_rows)
        by_btc_regime.append((regime, len(r_rows), rwr, ravg))

    return {
        "overall_wr": wr, "overall_avg_r": avg_r,
        "conf_win_r": _pearson(conf_vals, win_vals),
        "conf_r_r": _pearson(conf_vals, r_vals),
        "component_correlations": component_correlations,
        "quartiles": quartiles,
        "per_type": per_type,
        "by_btc_regime": by_btc_regime,
    }


def run_calibration(
    conn: sqlite3.Connection,
    days: int = 30,
    timeout_bars: int = 72,
) -> CalibrationResult:
    """Run full calibration analysis on signals from the last N days."""
    signals = conn.execute(
        f"""SELECT * FROM signals
        WHERE datetime(timestamp) > datetime('now', '-{days} days')
        ORDER BY confidence DESC""",
    ).fetchall()

    outcomes = _resolve_calibration_signals(signals, conn, timeout_bars)

    if not outcomes:
        return CalibrationResult(
            n_signals=len(signals), n_resolved=0,
            overall_wr=0, overall_avg_r=0,
            conf_win_r=0, conf_r_r=0,
            component_correlations={}, quartiles=[], per_type=[],
            by_btc_regime=[],
        )

    analysis = _analyze_outcomes(outcomes)
    return CalibrationResult(
        n_signals=len(signals),
        n_resolved=len(outcomes),
        **analysis,
    )


def build_calibration_embeds(result: CalibrationResult) -> list[dict]:
    """Build Discord embeds for calibration report."""
    COLOR_POS = 0x26A69A
    COLOR_WARN = 0xFFA726

    color = COLOR_WARN if result.conf_win_r < -0.1 else COLOR_POS
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Embed 1: Scoring overview
    embed1 = {
        "title": "Scoring Calibration",
        "color": color,
        "fields": [
            {"name": "Signals", "value": str(result.n_signals), "inline": True},
            {"name": "Resolved", "value": str(result.n_resolved), "inline": True},
            {"name": "WR", "value": f"{result.overall_wr:.0%}", "inline": True},
            {"name": "AvgR", "value": f"{result.overall_avg_r:+.2f}", "inline": True},
            {"name": "Conf->Win r", "value": f"{result.conf_win_r:+.3f}", "inline": True},
            {"name": "Conf->R r", "value": f"{result.conf_r_r:+.3f}", "inline": True},
        ],
        "footer": {"text": timestamp},
    }

    # Component correlations in description
    if result.component_correlations:
        comp_lines = []
        for comp, r in sorted(result.component_correlations.items(), key=lambda x: x[1]):
            marker = "!!" if r < -0.1 else ("ok" if r > 0.1 else "~")
            comp_lines.append(f"{marker} {comp:<12} r={r:+.3f}")
        embed1["description"] = "**Components:**\n```\n" + "\n".join(comp_lines) + "\n```"

    embeds = [embed1]

    # Embed 2: Per-type breakdown
    if result.per_type:
        type_lines = []
        for t, n, wr, avg_r, corr in result.per_type:
            type_lines.append(f"**{t}**: n={n} WR={wr:.0%} AvgR={avg_r:+.2f} conf->win r={corr:+.3f}")
        embeds.append({
            "title": "Calibration: By Type",
            "color": color,
            "description": "\n".join(type_lines),
            "footer": {"text": timestamp},
        })

    return embeds


def format_calibration(result: CalibrationResult) -> str:
    """Format calibration result as a text report."""
    lines = []
    lines.append("=== SCORING CALIBRATION ===")
    lines.append(f"Signals: {result.n_signals} | Resolved: {result.n_resolved}")
    lines.append(f"Overall: WR={result.overall_wr:.0%}  AvgR={result.overall_avg_r:+.2f}")
    lines.append("")

    # Correlation headline
    lines.append("--- Composite Confidence ---")
    lines.append(f"  conf vs win: r={result.conf_win_r:+.3f}  conf vs R: r={result.conf_r_r:+.3f}")
    if result.conf_win_r < -0.1:
        lines.append("  ⚠ INVERTED — higher confidence → fewer wins")
    elif result.conf_win_r > 0.1:
        lines.append("  ✓ Positive — higher confidence → more wins")
    else:
        lines.append("  ~ Non-predictive")
    lines.append("")

    # Component correlations
    if result.component_correlations:
        lines.append("--- Component → Win Correlations ---")
        for comp, r in sorted(result.component_correlations.items(), key=lambda x: x[1]):
            marker = "⚠" if r < -0.1 else ("✓" if r > 0.1 else "~")
            lines.append(f"  {marker} {comp:<12} r={r:+.3f}")
        lines.append("")

    # Quartiles
    if result.quartiles:
        lines.append("--- Confidence Quartiles ---")
        lines.append(f"  {'Bucket':<25} {'N':>4} {'WR':>6} {'AvgR':>7}")
        lines.append(f"  {'-' * 45}")
        for label, n, wr, avg_r in result.quartiles:
            lines.append(f"  {label:<25} {n:>4} {wr:>5.0%} {avg_r:>+7.2f}")
        lines.append("")

    # Per-type
    if result.per_type:
        lines.append("--- Per-Type ---")
        for t, n, wr, avg_r, corr in result.per_type:
            lines.append(f"  Type {t}: n={n:>3} WR={wr:>4.0%} AvgR={avg_r:>+.2f} conf→win r={corr:+.3f}")
        lines.append("")

    # BTC regime
    if result.by_btc_regime:
        lines.append("--- By BTC Regime ---")
        for regime, n, wr, avg_r in result.by_btc_regime:
            lines.append(f"  {regime:<10} n={n:>3} WR={wr:>4.0%} AvgR={avg_r:>+.2f}")
        lines.append("")

    return "\n".join(lines)
