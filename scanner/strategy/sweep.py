"""Grid search over strategy parameters."""

import sqlite3
from itertools import product

from ..backtest.resolver import build_screener_setup, resolve_signal, TradeOutcome
from ..backtest.candles import fetch_and_cache
from ..backtest.metrics import compute_metrics, MetricsReport
from .models import Strategy
from .replay import replay

import pandas as pd


def _parse_range(spec: str) -> list[float]:
    """Parse 'start:step:end' into a list of floats."""
    parts = spec.split(":")
    if len(parts) == 3:
        start, step, end = float(parts[0]), float(parts[1]), float(parts[2])
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
    timeout_bars: int = 168,
    fee_rate: float = 0.00035,
    verbose: bool = False,
) -> list[tuple[float, float, MetricsReport]]:
    """Grid over atr_stop_mult x atr_target_mult for existing signals.

    Cheap — reuses cached signals, only re-resolves with different stops/targets.
    """
    results = []
    combos = list(product(stop_range, target_range))

    for i, (stop_mult, target_mult) in enumerate(combos):
        outcomes: list[TradeOutcome] = []

        for row in signal_rows:
            setup = build_screener_setup(row, stop_mult, target_mult)
            if not setup:
                continue

            try:
                candles = fetch_and_cache(
                    setup.coin, "1h",
                    setup.timestamp,
                    pd.Timestamp(setup.timestamp, tz="UTC")
                    .to_pydatetime()
                    .__add__(pd.Timedelta(hours=timeout_bars + 1))
                    .isoformat(),
                    conn,
                )
                outcome = resolve_signal(setup, candles, timeout_bars, fee_rate)
                outcomes.append(outcome)
            except Exception:
                continue

        if outcomes:
            label = f"stop={stop_mult:.1f} target={target_mult:.1f}"
            metrics = compute_metrics(outcomes, label)
            results.append((stop_mult, target_mult, metrics))

        if verbose:
            print(f"  [{i+1}/{len(combos)}] stop={stop_mult:.1f} target={target_mult:.1f}: "
                  f"{len(outcomes)} resolved")

    return results


def sweep_screener(
    param_grid: list[dict],
    since: str,
    until: str,
    conn: sqlite3.Connection,
    timeout_bars: int = 168,
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
