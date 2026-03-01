"""Grid search over strategy parameters."""

import sqlite3
import sys
from dataclasses import dataclass
from itertools import product

import pandas as pd

from ..backtest.metrics import MetricsReport, compute_metrics
from ..backtest.resolver import TradeOutcome, build_screener_setup, fetch_and_resolve
from .models import Strategy
from .replay import replay


def _parse_range(spec: str) -> list[float]:
    """Parse 'start:step:end' into a list of floats."""
    parts = spec.split(":")
    if len(parts) == 3:
        start, step, end = float(parts[0]), float(parts[1]), float(parts[2])
        if step <= 0:
            raise ValueError(f"step must be positive, got {step} in '{spec}'")
        values = []
        v = start
        while v <= end + 1e-9:
            values.append(round(v, 4))
            v += step
        return values
    return [float(parts[0])]


def sweep_resolution(
    signal_rows: list[dict],
    stop_range: list[float],
    target_range: list[float],
    conn: sqlite3.Connection,
    timeout_bars: int = 72,
    fee_rate: float = 0.00035,
    verbose: bool = False,
    funding_rate_8h: float = 0.0,
    slippage_rate: float = 0.0,
) -> list[tuple[float, float, MetricsReport]]:
    """Grid over atr_stop_mult x atr_target_mult for existing signals.

    Cheap — reuses cached signals, only re-resolves with different stops/targets.
    """
    results = []
    combos = list(product(stop_range, target_range))

    n_combos = len(combos)
    n_signals = len(signal_rows)

    for i, (stop_mult, target_mult) in enumerate(combos):
        outcomes: list[TradeOutcome] = []

        for j, row in enumerate(signal_rows):
            setup = build_screener_setup(row, stop_mult, target_mult)
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
                print(f"  Resolve {setup.coin}: {e}", file=sys.stderr)
                continue

        if outcomes:
            label = f"stop={stop_mult:.1f} target={target_mult:.1f}"
            metrics = compute_metrics(outcomes, label)
            results.append((stop_mult, target_mult, metrics))

        print(f"  [{i+1}/{n_combos}] s={stop_mult:.1f} t={target_mult:.1f}: "
              f"{len(outcomes)}/{n_signals} resolved", file=sys.stderr)

    return results


def sweep_screener(
    param_grid: list[dict],
    since: str,
    until: str,
    conn: sqlite3.Connection,
    timeout_bars: int = 72,
    fee_rate: float = 0.00035,
    verbose: bool = False,
) -> list[tuple[dict, int, MetricsReport | None]]:
    """Grid over screener params. Re-runs detection per combo. Heavier.

    Each dict in param_grid should have keys matching Strategy fields:
    swing_lookback, sr_cluster_pct, proximity_atr_mult, max_setups_per_coin,
    atr_stop_mult, atr_target_mult.
    """
    results = []

    for i, params in enumerate(param_grid):
        strategy = Strategy(
            strategy_id=f"sweep-{i}",
            label=f"sweep-{i}",
            swing_lookback=params.get("swing_lookback", 5),
            sr_cluster_pct=params.get("sr_cluster_pct", 0.015),
            proximity_atr_mult=params.get("proximity_atr_mult", 1.5),
            max_setups_per_coin=params.get("max_setups_per_coin", 3),
            atr_stop_mult=params.get("atr_stop_mult", 1.5),
            atr_target_mult=params.get("atr_target_mult", 3.0),
        )

        signals, metrics = replay(
            strategy, since, until, conn,
            resolve=True,
            timeout_bars=timeout_bars,
            fee_rate=fee_rate,
            verbose=False,
        )

        results.append((params, len(signals), metrics))

        if verbose:
            wr = f"{metrics.win_rate:.1%}" if metrics else "N/A"
            exp = f"{metrics.expectancy:+.3f}R" if metrics else "N/A"
            print(f"  [{i+1}/{len(param_grid)}] {params}: "
                  f"{len(signals)} signals, WR={wr}, Exp={exp}")

    return results


# --- Walk-Forward Sweep ---


@dataclass
class WalkForwardFold:
    fold_index: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    is_best_params: tuple[float, float]
    is_best_metrics: MetricsReport
    oos_metrics: MetricsReport
    wfe: float | None  # None if IS expectancy <= 0


@dataclass
class WalkForwardResult:
    folds: list[WalkForwardFold]
    overall_oos_metrics: MetricsReport  # all OOS outcomes combined
    avg_wfe: float | None
    n_signals_total: int


def _run_fold(
    fold_idx: int,
    n_folds: int,
    sorted_rows: list[dict],
    boundaries: list,
    stop_range: list[float],
    target_range: list[float],
    conn: sqlite3.Connection,
    timeout_bars: int,
    fee_rate: float,
    funding_rate_8h: float,
    slippage_rate: float,
    verbose: bool,
) -> tuple[WalkForwardFold, list[TradeOutcome]] | None:
    """Run one fold of anchored walk-forward: IS sweep → OOS evaluate."""
    is_start = boundaries[0]
    is_end = boundaries[fold_idx + 1]
    oos_start = boundaries[fold_idx + 1]
    oos_end = boundaries[fold_idx + 2]

    is_rows = [r for r in sorted_rows
                if is_start <= pd.Timestamp(r["timestamp"], tz="UTC") < is_end]
    oos_rows = [r for r in sorted_rows
                 if oos_start <= pd.Timestamp(r["timestamp"], tz="UTC") < oos_end]

    if not is_rows or not oos_rows:
        return None

    print(f"  Fold {fold_idx + 1}/{n_folds}: IS {len(is_rows)} signals, OOS {len(oos_rows)} signals",
          file=sys.stderr)

    # Find best params on IS
    is_results = sweep_resolution(
        is_rows, stop_range, target_range, conn,
        timeout_bars=timeout_bars, fee_rate=fee_rate,
        funding_rate_8h=funding_rate_8h, slippage_rate=slippage_rate,
    )
    if not is_results:
        return None

    is_results.sort(key=lambda r: r[2].expectancy, reverse=True)
    best_stop, best_target, best_is_metrics = is_results[0]

    # Evaluate best params on OOS
    oos_outcomes: list[TradeOutcome] = []
    for row in oos_rows:
        setup = build_screener_setup(row, best_stop, best_target)
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
            oos_outcomes.append(outcome)
        except (ValueError, KeyError, IndexError, OSError) as e:
            print(f"  OOS resolve {setup.coin}: {e}", file=sys.stderr)

    oos_metrics = compute_metrics(oos_outcomes, f"fold-{fold_idx + 1}-oos")

    wfe = None
    if best_is_metrics.expectancy > 0 and oos_metrics.sample_size > 0:
        wfe = round(oos_metrics.expectancy / best_is_metrics.expectancy, 4)

    fold = WalkForwardFold(
        fold_index=fold_idx + 1,
        is_start=is_start.strftime("%Y-%m-%d"),
        is_end=is_end.strftime("%Y-%m-%d"),
        oos_start=oos_start.strftime("%Y-%m-%d"),
        oos_end=oos_end.strftime("%Y-%m-%d"),
        is_best_params=(best_stop, best_target),
        is_best_metrics=best_is_metrics,
        oos_metrics=oos_metrics,
        wfe=wfe,
    )

    if verbose:
        wfe_str = f"{wfe:.2f}" if wfe is not None else "n/a"
        print(f"  Fold {fold_idx + 1}: IS [{is_start.strftime('%m/%d')}-{is_end.strftime('%m/%d')}] "
              f"→ OOS [{oos_start.strftime('%m/%d')}-{oos_end.strftime('%m/%d')}] "
              f"best=s{best_stop:.1f}/t{best_target:.1f} "
              f"IS_exp={best_is_metrics.expectancy:+.3f} "
              f"OOS_exp={oos_metrics.expectancy:+.3f} WFE={wfe_str}")

    return fold, oos_outcomes


def walk_forward_sweep(
    signal_rows: list[dict],
    stop_range: list[float],
    target_range: list[float],
    conn: sqlite3.Connection,
    n_folds: int = 3,
    timeout_bars: int = 72,
    fee_rate: float = 0.00035,
    funding_rate_8h: float = 0.0,
    slippage_rate: float = 0.0,
    verbose: bool = False,
) -> WalkForwardResult | None:
    """Anchored walk-forward sweep.

    Splits signals into n_folds+1 equal-time segments.
    Fold i: IS = segments 0..i, OOS = segment i+1.
    Find best params on IS, evaluate on OOS.
    WFE = OOS_exp / IS_exp.
    """
    if not signal_rows:
        return None

    sorted_rows = sorted(signal_rows, key=lambda r: r.get("timestamp", ""))
    ts_first = pd.Timestamp(sorted_rows[0]["timestamp"], tz="UTC")
    ts_last = pd.Timestamp(sorted_rows[-1]["timestamp"], tz="UTC")
    total_duration = ts_last - ts_first

    if total_duration.total_seconds() <= 0:
        return None

    n_segments = n_folds + 1
    segment_duration = total_duration / n_segments
    boundaries = [ts_first + segment_duration * i for i in range(n_segments + 1)]

    folds: list[WalkForwardFold] = []
    all_oos_outcomes: list[TradeOutcome] = []

    for fold_idx in range(n_folds):
        result = _run_fold(
            fold_idx, n_folds, sorted_rows, boundaries,
            stop_range, target_range, conn,
            timeout_bars, fee_rate, funding_rate_8h, slippage_rate, verbose,
        )
        if result is None:
            continue
        fold, oos_outcomes = result
        folds.append(fold)
        all_oos_outcomes.extend(oos_outcomes)

    if not folds:
        return None

    overall_oos = compute_metrics(all_oos_outcomes, "combined-oos")
    wfe_values = [f.wfe for f in folds if f.wfe is not None]
    avg_wfe = round(sum(wfe_values) / len(wfe_values), 4) if wfe_values else None

    return WalkForwardResult(
        folds=folds,
        overall_oos_metrics=overall_oos,
        avg_wfe=avg_wfe,
        n_signals_total=len(signal_rows),
    )
