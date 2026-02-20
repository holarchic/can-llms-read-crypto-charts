"""Strategy CLI — create, list, activate, replay, compare, sweep, backfill, report, disable."""

import argparse
import sys
from datetime import datetime, timezone, timedelta

import pandas as pd

from ..storage import get_connection
from ..backtest.candles import cache_candles, fetch_and_cache
from ..backtest.report import format_summary
from .models import Strategy
from .store import (
    save_strategy,
    get_strategy,
    list_strategies,
    activate_strategy,
    ensure_baseline,
)
from .replay import replay
from .compare import format_comparison
from ..backtest.metrics import deflated_sharpe_ratio
from .sweep import sweep_resolution, sweep_screener, _parse_range, walk_forward_sweep
from .report import (
    signal_quality_report,
    llm_filter_report,
    regime_decay_report,
    regime_crosstab_report,
    regime_conditioned_decay_report,
    llm_alpha_decay_report,
    resolution_drift_report,
    LLMAlphaDeltaWindow,
    suggest_disabled_types,
    format_signal_quality,
    format_llm_filter,
    format_regime_decay,
    format_regime_crosstab,
    format_regime_conditioned_decay,
    format_llm_alpha_decay,
    format_resolution_drift,
    format_disable_suggestion,
    build_signal_quality_embeds,
    build_llm_filter_embeds,
    build_disable_embeds,
)
from .calibrate import run_calibration, format_calibration, build_calibration_embeds


def cmd_create(args):
    conn = get_connection()
    sid = args.id or f"v{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"

    strategy = Strategy(
        strategy_id=sid,
        label=args.label,
        swing_lookback=args.swing_lookback,
        sr_cluster_pct=args.sr_cluster_pct,
        proximity_atr_mult=args.proximity_mult,
        max_setups_per_coin=args.max_setups,
        atr_stop_mult=args.atr_stop,
        atr_target_mult=args.atr_target,
        ruleset_version=args.ruleset_version or "",
        notes=args.notes or "",
    )
    save_strategy(strategy, conn)
    conn.close()
    print(f"Created strategy: {sid} ({args.label})")


def cmd_list(args):
    conn = get_connection()
    ensure_baseline(conn)
    strategies = list_strategies(conn)
    conn.close()

    if not strategies:
        print("No strategies found.")
        return

    print(f"{'ID':<20} {'Label':<16} {'Active':>6} {'Stop':>5} {'Target':>6} {'Prox':>5} {'Disabled':<8} {'Created':<20}")
    print("-" * 100)
    for s in strategies:
        active = " *" if s.is_active else ""
        created = s.created_at[:19] if s.created_at else ""
        disabled = s.disabled_types or "-"
        print(
            f"{s.strategy_id:<20} {s.label:<16} {active:>6} "
            f"{s.atr_stop_mult:>5.1f} {s.atr_target_mult:>6.1f} "
            f"{s.proximity_atr_mult:>5.1f} {disabled:<8} {created:<20}"
        )


def cmd_activate(args):
    conn = get_connection()
    if not get_strategy(args.strategy_id, conn):
        print(f"Strategy '{args.strategy_id}' not found.")
        conn.close()
        sys.exit(1)

    activate_strategy(args.strategy_id, conn)
    conn.close()
    print(f"Activated strategy: {args.strategy_id}")


def cmd_replay(args):
    conn = get_connection()
    strategy = get_strategy(args.strategy_id, conn)
    if not strategy:
        print(f"Strategy '{args.strategy_id}' not found.")
        conn.close()
        sys.exit(1)

    coins = [c.upper() for c in args.coin] if args.coin else None

    signals, metrics = replay(
        strategy,
        since=args.since,
        until=args.until,
        conn=conn,
        coins=coins,
        resolve=args.resolve,
        verbose=args.verbose,
    )

    print(f"\nReplay results for strategy '{strategy.strategy_id}' ({strategy.label}):")
    print(f"  Signals: {len(signals)}")

    if metrics:
        print(f"\n{format_summary(metrics)}")

    conn.close()


def cmd_compare(args):
    conn = get_connection()
    strat_a = get_strategy(args.id_a, conn)
    strat_b = get_strategy(args.id_b, conn)

    if not strat_a or not strat_b:
        missing = args.id_a if not strat_a else args.id_b
        print(f"Strategy '{missing}' not found.")
        conn.close()
        sys.exit(1)

    coins = [c.upper() for c in args.coin] if args.coin else None

    print(f"Replaying strategy A: {strat_a.strategy_id}...")
    _, metrics_a = replay(
        strat_a, args.since, args.until, conn,
        coins=coins, resolve=True, verbose=args.verbose,
    )

    print(f"Replaying strategy B: {strat_b.strategy_id}...")
    _, metrics_b = replay(
        strat_b, args.since, args.until, conn,
        coins=coins, resolve=True, verbose=args.verbose,
    )

    if not metrics_a or not metrics_b:
        print("Could not resolve signals for one or both strategies.")
        conn.close()
        sys.exit(1)

    print(f"\n{format_comparison(metrics_a, metrics_b)}")
    conn.close()


def cmd_sweep(args):
    conn = get_connection()

    stop_values = _parse_range(args.atr_stop)
    target_values = _parse_range(args.atr_target)

    # Load signals from a base strategy replay
    base_id = args.base or "v0"
    base = get_strategy(base_id, conn)
    if not base:
        ensure_baseline(conn)
        base = get_strategy("v0", conn)

    since = args.since or (datetime.now(timezone.utc) - timedelta(days=args.days)).strftime("%Y-%m-%d")
    until = args.until or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"Replaying base strategy '{base.strategy_id}' to get signals...")
    replay(base, since, until, conn, verbose=False)

    # Load the replayed signals
    rows = conn.execute(
        "SELECT * FROM signals WHERE strategy_id = ? ORDER BY timestamp",
        (base.strategy_id,),
    ).fetchall()
    signal_rows = [dict(r) for r in rows]

    if not signal_rows:
        print("No signals found. Run backfill first.")
        conn.close()
        sys.exit(1)

    if args.walk_forward:
        print(f"Walk-forward sweep: {args.folds} folds, "
              f"{len(stop_values)}x{len(target_values)} combos, {len(signal_rows)} signals...")
        wf_result = walk_forward_sweep(
            signal_rows, stop_values, target_values, conn,
            n_folds=args.folds, verbose=args.verbose,
        )

        if not wf_result:
            print("Insufficient data for walk-forward sweep.")
            conn.close()
            sys.exit(0)

        print(f"\n{'Fold':>6} {'IS_window':<20} {'OOS_window':<20} {'IS_best':<12} "
              f"{'IS_exp':>8} {'OOS_exp':>8} {'WFE':>6}")
        print("-" * 85)
        for fold in wf_result.folds:
            s, t = fold.is_best_params
            wfe_str = f"{fold.wfe:.2f}" if fold.wfe is not None else "n/a"
            print(
                f"{fold.fold_index:>6} {fold.is_start+'-'+fold.is_end:<20} "
                f"{fold.oos_start+'-'+fold.oos_end:<20} "
                f"{'s='+str(s)+' t='+str(t):<12} "
                f"{fold.is_best_metrics.expectancy:>+7.3f}R "
                f"{fold.oos_metrics.expectancy:>+7.3f}R "
                f"{wfe_str:>6}"
            )

        oos = wf_result.overall_oos_metrics
        avg_wfe = f"{wf_result.avg_wfe:.2f}" if wf_result.avg_wfe is not None else "n/a"
        print(f"\n  Combined OOS: n={oos.sample_size} Exp={oos.expectancy:+.3f}R  Avg WFE: {avg_wfe}")

        if wf_result.avg_wfe is not None:
            if wf_result.avg_wfe > 0.5:
                print("  Assessment: ACCEPTABLE (WFE > 0.5)")
            elif wf_result.avg_wfe > 0.3:
                print("  Assessment: CAUTION (WFE 0.3-0.5)")
            else:
                print("  Assessment: LIKELY OVERFITTING (WFE < 0.3)")

    else:
        print(f"Sweeping {len(stop_values)}x{len(target_values)} combos over {len(signal_rows)} signals...")
        results = sweep_resolution(
            signal_rows, stop_values, target_values, conn,
            verbose=args.verbose,
        )

        # Print results sorted by expectancy
        results.sort(key=lambda r: r[2].expectancy, reverse=True)

        print(f"\n{'Stop':>6} {'Target':>7} {'N':>5} {'WR':>7} {'ExpR':>8} {'PF':>7} {'TotR':>8}")
        print("-" * 55)
        for stop, target, m in results:
            pf = f"{m.profit_factor:.2f}" if m.profit_factor != float("inf") else "inf"
            print(
                f"{stop:>6.1f} {target:>7.1f} {m.sample_size:>5} "
                f"{m.win_rate:>6.1%} {m.expectancy:>+7.3f}R "
                f"{pf:>6} {m.total_r:>+7.2f}"
            )

        # DSR on best result
        if len(results) > 1:
            best_stop, best_target, best_m = results[0]
            # Collect outcomes for best params to compute DSR
            from ..backtest.resolver import build_screener_setup
            print(f"\n  Computing DSR (re-resolving {len(signal_rows)} signals for best params)...",
                  file=sys.stderr)
            best_outcomes = []
            for row in signal_rows:
                setup = build_screener_setup(row, best_stop, best_target)
                if not setup:
                    continue
                try:
                    candles = fetch_and_cache(
                        setup.coin, "1h", setup.timestamp,
                        (pd.Timestamp(setup.timestamp, tz="UTC")
                         + pd.Timedelta(hours=72 + 1)).isoformat(),
                        conn,
                    )
                    from ..backtest.resolver import resolve_signal
                    outcome = resolve_signal(setup, candles, 72, 0.00035)
                    best_outcomes.append(outcome)
                except Exception:
                    continue

            n_trials = len(results)
            dsr = deflated_sharpe_ratio(best_outcomes, n_trials)
            if dsr:
                sig = "SIGNIFICANT" if dsr.is_significant else "LIKELY NOISE"
                print(f"\n  DSR: p={dsr.p_value:.4f} (N={n_trials} trials) — {sig}")
                print(f"  Observed SR: {dsr.observed_sharpe:.4f}  Expected max SR: {dsr.expected_max_sharpe:.4f}")

    conn.close()


def cmd_report(args):
    conn = get_connection()
    strategy = get_strategy(args.strategy_id, conn)
    if not strategy:
        print(f"Strategy '{args.strategy_id}' not found.")
        conn.close()
        sys.exit(1)

    coins = [c.upper() for c in args.coin] if args.coin else None

    # Default: run all loops unless specific ones requested
    run_all = not (
        args.signal_quality or args.llm_filter or args.regime
        or args.suggest_disable or args.regime_crosstab
        or args.decay_conditioned or args.llm_decay or args.resolution_drift
    )

    sections: list[str] = []
    embeds: list[dict] = []

    if run_all or args.signal_quality:
        overall, sliced = signal_quality_report(
            strategy, args.since, args.until, conn,
            coins=coins, verbose=args.verbose,
            run_permutation=args.permutation,
        )
        sections.append(format_signal_quality(overall, sliced))
        embeds.extend(build_signal_quality_embeds(overall, sliced))

    if run_all or args.llm_filter:
        result = llm_filter_report(
            strategy, args.since, args.until, conn,
            coins=coins, verbose=args.verbose,
        )
        sections.append(format_llm_filter(result))
        embeds.extend(build_llm_filter_embeds(result))

    if run_all or args.regime:
        windows = regime_decay_report(
            strategy, args.since, args.until, conn,
            window_weeks=args.window_weeks,
            coins=coins, verbose=args.verbose,
        )
        sections.append(format_regime_decay(windows))

    if run_all or args.regime_crosstab:
        from .report import load_strategy_signals, resolve_signals
        signal_rows = load_strategy_signals(
            strategy.strategy_id, args.since, args.until, conn, coins,
        )
        outcomes = resolve_signals(
            signal_rows, strategy.atr_stop_mult, strategy.atr_target_mult,
            conn, verbose=args.verbose,
        )
        crosstab = regime_crosstab_report(outcomes)
        sections.append(format_regime_crosstab(crosstab))

    if args.decay_conditioned:
        dc_windows = regime_conditioned_decay_report(
            strategy, args.since, args.until, conn,
            window_weeks=args.window_weeks,
            coins=coins, verbose=args.verbose,
        )
        sections.append(format_regime_conditioned_decay(dc_windows))

    if args.llm_decay:
        llm_windows = llm_alpha_decay_report(
            strategy, args.since, args.until, conn,
            window_weeks=args.window_weeks,
            coins=coins, verbose=args.verbose,
        )
        sections.append(format_llm_alpha_decay(llm_windows))

    if args.resolution_drift:
        rd_windows = resolution_drift_report(
            strategy, args.since, args.until, conn,
            window_weeks=args.window_weeks,
            coins=coins, verbose=args.verbose,
        )
        sections.append(format_resolution_drift(rd_windows))

    if run_all or args.suggest_disable:
        # Need signal quality data for suggestions
        if not (run_all or args.signal_quality):
            _, sliced = signal_quality_report(
                strategy, args.since, args.until, conn,
                coins=coins, verbose=args.verbose,
                run_permutation=args.permutation,
            )
        setup_type_reports = sliced.get("setup_type", [])
        suggestions = suggest_disabled_types(setup_type_reports, args.min_sample)
        sections.append(format_disable_suggestion(suggestions))
        embeds.extend(build_disable_embeds(suggestions, strategy.strategy_id))

    report_text = "\n\n".join(sections)
    print(report_text)

    if args.discord:
        from ..alerts import send_report_embeds
        ok = send_report_embeds(embeds)
        if ok:
            print("\n  Sent to Discord.")
        else:
            print("\n  Discord send failed (no webhook configured?).")

    conn.close()


def cmd_disable(args):
    conn = get_connection()
    strategy = get_strategy(args.strategy_id, conn)
    if not strategy:
        print(f"Strategy '{args.strategy_id}' not found.")
        conn.close()
        sys.exit(1)

    if args.types.lower() == "none":
        strategy.disabled_types = ""
        print(f"Cleared disabled types for strategy '{args.strategy_id}'")
    else:
        types = ",".join(t.strip().upper() for t in args.types.split(",") if t.strip())
        strategy.disabled_types = types
        print(f"Disabled types for strategy '{args.strategy_id}': {types}")

    save_strategy(strategy, conn)
    conn.close()


def cmd_calibrate(args):
    conn = get_connection()
    result = run_calibration(conn, days=args.days, timeout_bars=args.timeout)
    report_text = format_calibration(result)
    print(report_text)

    if args.discord:
        from ..alerts import send_report_embeds
        embeds = build_calibration_embeds(result)
        ok = send_report_embeds(embeds)
        if ok:
            print("\n  Sent to Discord.")
        else:
            print("\n  Discord send failed (no webhook configured?).")

    conn.close()


def cmd_backfill(args):
    from ..data import fetch_pairs_with_volume, get_exchange
    from ..config import ScannerConfig
    import time

    conn = get_connection()
    config = ScannerConfig()

    if args.coin:
        coins = [args.coin.upper()]
    else:
        pairs = fetch_pairs_with_volume(config)
        coins = [p["coin"] for p in pairs]
        print(f"Backfilling {len(coins)} coins")

    exchange = get_exchange()
    since = args.since
    until = args.until or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

    total_cached = 0
    for i, coin in enumerate(coins):
        try:
            df = fetch_and_cache(coin, "4h", since, until, conn, config)
            n = len(df) if df is not None else 0
            total_cached += n
            if args.verbose:
                print(f"  [{i+1}/{len(coins)}] {coin}: {n} candles")
        except Exception as e:
            if args.verbose:
                print(f"  [{i+1}/{len(coins)}] {coin}: error — {e}")
        time.sleep(config.api_sleep_ms / 1000)

    conn.close()
    print(f"Backfill complete: {total_cached} candles cached for {len(coins)} coins")


def main():
    parser = argparse.ArgumentParser(
        prog="scanner.strategy",
        description="Strategy versioning + replay system",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # create
    p_create = sub.add_parser("create", help="Create a new strategy")
    p_create.add_argument("--id", type=str, default=None, help="Strategy ID (auto-generated if omitted)")
    p_create.add_argument("--label", type=str, required=True, help="Human-readable label")
    p_create.add_argument("--swing-lookback", type=int, default=5)
    p_create.add_argument("--sr-cluster-pct", type=float, default=0.015)
    p_create.add_argument("--proximity-mult", type=float, default=1.5)
    p_create.add_argument("--max-setups", type=int, default=3)
    p_create.add_argument("--atr-stop", type=float, default=1.5)
    p_create.add_argument("--atr-target", type=float, default=3.0)
    p_create.add_argument("--ruleset-version", type=str, default=None, help="UNIFIED_TA_RULESET version (e.g. v1.2.0)")
    p_create.add_argument("--notes", type=str, default=None)
    p_create.set_defaults(func=cmd_create)

    # list
    p_list = sub.add_parser("list", help="List all strategies")
    p_list.set_defaults(func=cmd_list)

    # activate
    p_activate = sub.add_parser("activate", help="Set active strategy")
    p_activate.add_argument("strategy_id", help="Strategy ID to activate")
    p_activate.set_defaults(func=cmd_activate)

    # replay
    p_replay = sub.add_parser("replay", help="Replay screener on cached candles")
    p_replay.add_argument("strategy_id", help="Strategy ID to replay")
    p_replay.add_argument("--since", required=True, help="Start date (YYYY-MM-DD)")
    p_replay.add_argument("--until", required=True, help="End date (YYYY-MM-DD)")
    p_replay.add_argument("--coin", nargs="*", help="Filter by coin(s)")
    p_replay.add_argument("--resolve", action="store_true", help="Also resolve and compute metrics")
    p_replay.add_argument("-v", "--verbose", action="store_true")
    p_replay.set_defaults(func=cmd_replay)

    # compare
    p_compare = sub.add_parser("compare", help="A/B compare two strategies")
    p_compare.add_argument("id_a", help="Strategy A")
    p_compare.add_argument("id_b", help="Strategy B")
    p_compare.add_argument("--since", required=True, help="Start date")
    p_compare.add_argument("--until", required=True, help="End date")
    p_compare.add_argument("--coin", nargs="*", help="Filter by coin(s)")
    p_compare.add_argument("-v", "--verbose", action="store_true")
    p_compare.set_defaults(func=cmd_compare)

    # sweep
    p_sweep = sub.add_parser("sweep", help="Grid search over resolution params")
    p_sweep.add_argument("--atr-stop", type=str, default="1.0:0.5:3.0", help="Range: start:step:end")
    p_sweep.add_argument("--atr-target", type=str, default="2.0:0.5:5.0", help="Range: start:step:end")
    p_sweep.add_argument("--base", type=str, default=None, help="Base strategy ID for signals")
    p_sweep.add_argument("--since", type=str, default=None, help="Start date")
    p_sweep.add_argument("--until", type=str, default=None, help="End date")
    p_sweep.add_argument("--days", type=int, default=30, help="Lookback days (if --since not set)")
    p_sweep.add_argument("--walk-forward", action="store_true", help="Run anchored walk-forward sweep")
    p_sweep.add_argument("--folds", type=int, default=3, help="Number of walk-forward folds (default: 3)")
    p_sweep.add_argument("-v", "--verbose", action="store_true")
    p_sweep.set_defaults(func=cmd_sweep)

    # report
    p_report = sub.add_parser("report", help="Validation reports (signal quality, LLM filter, regime decay)")
    p_report.add_argument("strategy_id", help="Strategy ID to report on")
    p_report.add_argument("--since", required=True, help="Start date (YYYY-MM-DD)")
    p_report.add_argument("--until", required=True, help="End date (YYYY-MM-DD)")
    p_report.add_argument("--coin", nargs="*", help="Filter by coin(s)")
    p_report.add_argument("--signal-quality", action="store_true", help="Run signal quality report only")
    p_report.add_argument("--llm-filter", action="store_true", help="Run LLM filter value report only")
    p_report.add_argument("--regime", action="store_true", help="Run regime decay report only")
    p_report.add_argument("--suggest-disable", action="store_true", help="Suggest setup types to disable")
    p_report.add_argument("--regime-crosstab", action="store_true", help="Show setup_type x regime cross-tab")
    p_report.add_argument("--decay-conditioned", action="store_true", help="Regime-conditioned rolling expectancy")
    p_report.add_argument("--llm-decay", action="store_true", help="LLM filter alpha delta over time")
    p_report.add_argument("--resolution-drift", action="store_true", help="Time-to-resolution drift")
    p_report.add_argument("--window-weeks", type=int, default=2, help="Window size for rolling reports (default: 2)")
    p_report.add_argument("--min-sample", type=int, default=10, help="Min sample for disable suggestions (default: 10)")
    p_report.add_argument("--permutation", action="store_true", help="Run Monte Carlo permutation test on expectancy")
    p_report.add_argument("-v", "--verbose", action="store_true")
    p_report.add_argument("--discord", action="store_true", help="Post results to Discord reports channel")
    p_report.set_defaults(func=cmd_report)

    # disable
    p_disable = sub.add_parser("disable", help="Disable setup types for a strategy")
    p_disable.add_argument("strategy_id", help="Strategy ID")
    p_disable.add_argument("types", help="Comma-separated types to disable (e.g. 'A,E'), or 'none' to clear")
    p_disable.set_defaults(func=cmd_disable)

    # calibrate
    p_cal = sub.add_parser("calibrate", help="Scoring calibration analysis")
    p_cal.add_argument("--days", type=int, default=30, help="Lookback days (default: 30)")
    p_cal.add_argument("--timeout", type=int, default=72, help="Timeout bars (default: 72 = 3d)")
    p_cal.add_argument("--discord", action="store_true", help="Post results to Discord reports channel")
    p_cal.set_defaults(func=cmd_calibrate)

    # backfill
    p_backfill = sub.add_parser("backfill", help="Fetch and cache 4H candles")
    p_backfill.add_argument("--since", required=True, help="Start date (YYYY-MM-DD)")
    p_backfill.add_argument("--until", type=str, default=None, help="End date")
    p_backfill.add_argument("--coin", type=str, default=None, help="Single coin filter")
    p_backfill.add_argument("-v", "--verbose", action="store_true")
    p_backfill.set_defaults(func=cmd_backfill)

    args = parser.parse_args()
    args.func(args)
