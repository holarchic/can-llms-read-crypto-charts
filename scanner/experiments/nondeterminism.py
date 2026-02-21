"""Experiment: LLM decision stability test.

Replays past signals through the LLM multiple times to measure
decision consistency. Uses cached candles (4H/1H) and fetches 1D
from exchange, all truncated to the signal's original timestamp.

Usage:
    python -m scanner.experiments.nondeterminism [options]

Options:
    --runs N          Number of LLM calls per signal (default: 3)
    --confirms N      Number of CONFIRM signals to test (default: 10)
    --rejects N       Number of REJECT signals to test (default: 10)
    --waits N         Number of WAIT signals to test (default: 0)
    --dry-run         Show selected signals without running LLM
    --output FILE     Save results to JSON file
    --cost-limit N    Max total cost in USD (default: 15.0)
    --cached-only     Use only cached candles (4H/1H) — no exchange fetches.
                      Guarantees identical inputs to original analysis.
"""

import argparse
import json
import random
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scanner.config import ScannerConfig, DB_PATH
from scanner.data import fetch_candles
from scanner.backtest.candles import get_cached_candles
from scanner.charts import generate_charts_for_signal
from scanner.analysis import analyze_for_scanner
from scanner.models import (
    SetupSignal, SetupType, Direction, MarketContext,
    Structure, SwingPoint, SRLevel, MSB, FibLevels, ScoreComponents,
)
from scanner.indicators import build_context
from scanner.storage import get_connection


def load_sample_scans(
    conn: sqlite3.Connection,
    n_confirms: int = 10,
    n_rejects: int = 10,
    n_waits: int = 0,
) -> list[dict]:
    """Load a stratified sample of past scans with their signal data."""
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    samples = []
    for decision, n in [("CONFIRM", n_confirms), ("REJECT", n_rejects), ("WAIT", n_waits)]:
        if n <= 0:
            continue
        # Join scans with signals to get btc_structure and full signal context
        cur.execute("""
            SELECT s.scan_id, s.coin, s.setup_type, s.direction,
                   s.screener_confidence, s.key_level, s.decision,
                   s.created_at, s.analysis_text,
                   sig.btc_structure, sig.atr, sig.current_price,
                   sig.structure, sig.description,
                   sig.range_low, sig.range_high,
                   sig.fib_618, sig.fib_786
            FROM scans s
            LEFT JOIN signals sig ON s.coin = sig.coin
                AND s.setup_type = sig.setup_type
                AND s.direction = sig.direction
                AND abs(julianday(s.created_at) - julianday(sig.created_at)) < 0.01
            WHERE s.decision = ?
            ORDER BY s.created_at DESC
        """, (decision,))
        rows = [dict(r) for r in cur.fetchall()]

        # Deduplicate by scan_id (join may produce multiples)
        seen = set()
        unique = []
        for r in rows:
            if r["scan_id"] not in seen:
                seen.add(r["scan_id"])
                unique.append(r)

        if len(unique) > n:
            # Take most recent — they have best candle cache coverage
            unique = unique[:n]
        samples.extend(unique)

    return samples


def get_candles_as_of(
    coin: str,
    interval: str,
    as_of: str,
    count: int,
    conn: sqlite3.Connection,
    config: ScannerConfig,
) -> pd.DataFrame | None:
    """Get candles for a coin/interval as they existed at a point in time.

    Uses cache for 4h/1h, fetches from exchange for 1d.
    Truncates to candles with timestamp <= as_of.
    """
    if interval in ("4h", "1h"):
        # Use cache — go back far enough
        since = "2026-01-01T00:00:00+00:00"
        df = get_cached_candles(coin, interval, since, as_of, conn)
        if df is not None and len(df) > 0:
            # Truncate to as_of and take last N
            df = df[df.index <= pd.Timestamp(as_of)]
            if len(df) > count:
                df = df.iloc[-count:]
            return df if len(df) > 0 else None
        return None
    elif interval == "1d":
        # Fetch from exchange — historical candles are immutable
        try:
            df = fetch_candles(coin, interval, count, config)
            if df is not None and len(df) > 0:
                df = df[df.index <= pd.Timestamp(as_of)]
                if len(df) > count:
                    df = df.iloc[-count:]
                return df if len(df) > 0 else None
        except Exception as e:
            print(f"  Failed to fetch 1D candles for {coin}: {e}")
            return None
    return None


def reconstruct_signal(scan: dict, config: ScannerConfig) -> SetupSignal:
    """Reconstruct a SetupSignal from DB scan+signal data.

    The signal doesn't need to be perfect — the LLM only sees the charts
    and the text prompt. The signal is used to generate chart overlays
    (S/R levels, swing points) which we regenerate from candle data anyway.
    """
    # We'll build a minimal MarketContext — the chart generation will
    # recompute indicators from candle data
    context = MarketContext(
        structure=Structure(scan.get("structure", "RANGING")),
        swings=[],
        sr_levels=[],
        msbs=[],
        range_bounds=None,
        atr=scan.get("atr") or 0.0,
        current_price=scan.get("current_price") or scan.get("key_level") or 0.0,
        fib_levels=None,
    )

    return SetupSignal(
        coin=scan["coin"],
        setup_type=SetupType(scan["setup_type"]),
        direction=Direction(scan["direction"]),
        confidence=scan.get("screener_confidence") or 0.5,
        key_level=scan.get("key_level") or 0.0,
        context=context,
        description=scan.get("description") or "",
        max_leverage=None,  # Will be fetched if needed
    )


def rebuild_context_from_candles(
    signal: SetupSignal,
    candle_4h: pd.DataFrame,
    config: ScannerConfig,
) -> SetupSignal:
    """Rebuild market context (S/R, swings, structure) from 4H candle data.

    This ensures chart overlays match what was shown originally.
    """
    try:
        ctx = build_context(
            candle_4h,
            lookback=config.swing_lookback,
            sr_pct=config.sr_cluster_pct,
        )
        signal.context = ctx
    except Exception as e:
        print(f"  Warning: could not rebuild context for {signal.coin}: {e}")
    return signal


def run_experiment(
    samples: list[dict],
    n_runs: int,
    config: ScannerConfig,
    conn: sqlite3.Connection,
    cached_only: bool = False,
    cost_limit: float = 15.0,
    dry_run: bool = False,
) -> list[dict]:
    """Run the non-determinism experiment.

    For each sample signal, regenerate charts and run LLM analysis n_runs times.
    Returns list of result dicts with agreement metrics.
    """
    results = []
    total_cost = 0.0

    for i, scan in enumerate(samples, 1):
        coin = scan["coin"]
        original_decision = scan["decision"]
        as_of = scan["created_at"]
        btc_structure = scan.get("btc_structure")

        print(f"\n[{i}/{len(samples)}] {coin} {scan['setup_type']} "
              f"{scan['direction']} — original: {original_decision}")

        if dry_run:
            results.append({
                "scan_id": scan["scan_id"],
                "coin": coin,
                "setup_type": scan["setup_type"],
                "direction": scan["direction"],
                "original_decision": original_decision,
                "runs": [],
                "dry_run": True,
            })
            continue

        # Check cost limit
        estimated_remaining = (len(samples) - i + 1) * n_runs * 0.15
        if total_cost + estimated_remaining > cost_limit * 1.5:
            print(f"  Cost limit approaching (${total_cost:.2f} spent, "
                  f"~${estimated_remaining:.2f} remaining). Stopping.")
            break

        # Fetch candles as of signal time
        candle_sets = {}
        timeframes = [("4h", 200), ("1h", 120)]
        if not cached_only:
            timeframes = [("1d", 120)] + timeframes
        for tf, count in timeframes:
            df = get_candles_as_of(coin, tf, as_of, count, conn, config)
            if df is not None and len(df) >= 10:
                candle_sets[tf] = df

        if "4h" not in candle_sets:
            print(f"  Skipping {coin} — no 4H candle data available")
            continue

        # Reconstruct signal with proper context from candle data
        signal = reconstruct_signal(scan, config)
        signal = rebuild_context_from_candles(signal, candle_sets["4h"], config)

        # Generate charts (deterministic from candle data)
        chart_paths = generate_charts_for_signal(signal, candle_sets)
        # Verify all chart files actually exist (encode_image calls sys.exit on missing)
        chart_paths = [p for p in chart_paths if p.exists()]
        if not chart_paths:
            print(f"  Skipping {coin} — chart generation failed")
            continue
        print(f"  Charts: {[p.name for p in chart_paths]}"
              f" | S/R levels: {len(signal.context.sr_levels)}"
              f" | Swings: {len(signal.context.swings)}")

        # Run LLM N times
        runs = []
        for run_idx in range(n_runs):
            if total_cost > cost_limit:
                print(f"  Cost limit reached (${total_cost:.2f}). Stopping runs.")
                break

            try:
                result = analyze_for_scanner(signal, chart_paths, config, btc_structure)
                decision = result.decision
                cost = result.cost_usd
                total_cost += cost

                runs.append({
                    "run": run_idx + 1,
                    "decision": decision,
                    "confluence_score": result.confluence_score,
                    "cost_usd": cost,
                    "entry_zone": (result.parsed_decision or {}).get("entry_zone"),
                    "stop_loss": (result.parsed_decision or {}).get("stop_loss"),
                    "direction": (result.parsed_decision or {}).get("direction"),
                })
                print(f"  Run {run_idx + 1}: {decision} "
                      f"(${cost:.4f}) — total ${total_cost:.2f}")

                # Brief pause to avoid rate limits
                if run_idx < n_runs - 1:
                    time.sleep(1)

            except Exception as e:
                print(f"  Run {run_idx + 1} failed: {e}")
                runs.append({
                    "run": run_idx + 1,
                    "decision": "ERROR",
                    "error": str(e),
                    "cost_usd": 0,
                })

        # Compute agreement metrics
        decisions = [r["decision"] for r in runs if r["decision"] != "ERROR"]
        decision_counts = Counter(decisions)
        most_common = decision_counts.most_common(1)[0] if decision_counts else ("NONE", 0)

        results.append({
            "scan_id": scan["scan_id"],
            "coin": coin,
            "setup_type": scan["setup_type"],
            "direction": scan["direction"],
            "original_decision": original_decision,
            "btc_structure": btc_structure,
            "runs": runs,
            "decisions": decisions,
            "agreement_rate": most_common[1] / len(decisions) if decisions else 0,
            "majority_decision": most_common[0] if decisions else None,
            "original_reproduced": original_decision in decisions,
            "all_agree": len(set(decisions)) == 1 if decisions else False,
            "flipped": original_decision != most_common[0] if decisions else None,
        })

        # Cleanup chart files
        for p in chart_paths:
            try:
                p.unlink()
            except Exception:
                pass

    return results


def print_report(results: list[dict]) -> None:
    """Print a summary report of experiment results."""
    if not results:
        print("\nNo results to report.")
        return

    # Filter out dry runs
    real = [r for r in results if not r.get("dry_run")]
    if not real:
        print("\nDry run — no LLM calls made.")
        for r in results:
            print(f"  {r['coin']:8s} {r['setup_type']} {r['direction']:5s} — {r['original_decision']}")
        return

    print("\n" + "=" * 70)
    print("DECISION STABILITY REPORT")
    print("=" * 70)

    # Per-signal results
    print(f"\n{'Coin':8s} {'Type':4s} {'Dir':5s} {'Original':8s} → {'Runs':20s} {'Agree':6s} {'Flip':4s}")
    print("-" * 65)
    for r in real:
        runs_str = "/".join(r["decisions"]) if r["decisions"] else "n/a"
        agree = f"{r['agreement_rate']:.0%}" if r["decisions"] else "n/a"
        flip = "YES" if r.get("flipped") else "no"
        print(f"{r['coin']:8s} {r['setup_type']:4s} {r['direction']:5s} "
              f"{r['original_decision']:8s} → {runs_str:20s} {agree:>6s} {flip:>4s}")

    # Aggregate by original decision
    print(f"\n{'─' * 40}")
    print("AGGREGATE BY ORIGINAL DECISION")
    print(f"{'─' * 40}")

    for decision in ["CONFIRM", "REJECT", "WAIT"]:
        subset = [r for r in real if r["original_decision"] == decision]
        if not subset:
            continue

        n = len(subset)
        all_agree = sum(1 for r in subset if r.get("all_agree"))
        reproduced = sum(1 for r in subset if r.get("original_reproduced"))
        flipped = sum(1 for r in subset if r.get("flipped"))
        mean_agreement = sum(r.get("agreement_rate", 0) for r in subset) / n

        print(f"\n  {decision} (n={n}):")
        print(f"    Full agreement (all runs same):  {all_agree}/{n} ({all_agree/n:.0%})")
        print(f"    Original reproduced (any run):   {reproduced}/{n} ({reproduced/n:.0%})")
        print(f"    Majority flipped to different:    {flipped}/{n} ({flipped/n:.0%})")
        print(f"    Mean agreement rate:              {mean_agreement:.0%}")

    # Flip matrix
    print(f"\n{'─' * 40}")
    print("FLIP MATRIX (original → majority)")
    print(f"{'─' * 40}")

    matrix = Counter()
    for r in real:
        if r.get("majority_decision"):
            matrix[(r["original_decision"], r["majority_decision"])] += 1

    decisions_seen = sorted(set(
        [k[0] for k in matrix] + [k[1] for k in matrix]
    ))
    if decisions_seen:
        header = f"{'':10s}" + "".join(f"{d:>10s}" for d in decisions_seen)
        print(f"\n{header}")
        for orig in decisions_seen:
            row = f"{orig:10s}"
            for dest in decisions_seen:
                count = matrix.get((orig, dest), 0)
                row += f"{count:>10d}"
            print(row)

    # Cost summary
    total_cost = sum(
        sum(run.get("cost_usd", 0) for run in r.get("runs", []))
        for r in real
    )
    total_calls = sum(len(r.get("runs", [])) for r in real)
    print(f"\n{'─' * 40}")
    print(f"Total LLM calls: {total_calls}")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"Avg cost/call: ${total_cost/total_calls:.4f}" if total_calls else "")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="LLM decision stability experiment")
    parser.add_argument("--runs", type=int, default=3, help="LLM calls per signal (default: 3)")
    parser.add_argument("--confirms", type=int, default=10, help="CONFIRM signals to test")
    parser.add_argument("--rejects", type=int, default=10, help="REJECT signals to test")
    parser.add_argument("--waits", type=int, default=0, help="WAIT signals to test")
    parser.add_argument("--dry-run", action="store_true", help="Show selection without running LLM")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--cost-limit", type=float, default=15.0, help="Max total cost USD")
    parser.add_argument("--cached-only", action="store_true",
                        help="Use only cached candles (4H/1H) — identical inputs guaranteed")
    args = parser.parse_args()

    config = ScannerConfig()
    conn = get_connection()

    print("Loading sample signals...")
    samples = load_sample_scans(conn, args.confirms, args.rejects, args.waits)
    print(f"Selected {len(samples)} signals:")
    for s in samples:
        print(f"  {s['coin']:8s} {s['setup_type']} {s['direction']:5s} — {s['decision']}")

    if not samples:
        print("No signals found. Exiting.")
        sys.exit(1)

    print(f"\nExperiment: {len(samples)} signals × {args.runs} runs = "
          f"{len(samples) * args.runs} LLM calls")
    print(f"Estimated cost: ${len(samples) * args.runs * 0.12:.2f}")
    print(f"Cost limit: ${args.cost_limit:.2f}")

    if args.cached_only:
        print("Mode: CACHED-ONLY (4H/1H from cache, no 1D fetch)")

    results = run_experiment(
        samples, args.runs, config, conn,
        cost_limit=args.cost_limit, dry_run=args.dry_run,
        cached_only=args.cached_only,
    )

    print_report(results)

    if args.output:
        output_path = Path(args.output)
        # Make results JSON-serializable
        serializable = []
        for r in results:
            sr = {k: v for k, v in r.items()}
            serializable.append(sr)
        output_path.write_text(json.dumps(serializable, indent=2, default=str))
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
