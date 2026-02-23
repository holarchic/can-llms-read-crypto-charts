"""Replay engine — re-run screener on cached 4H candles for any strategy config."""

import sqlite3
import sys
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from ..indicators import build_context
from ..screener import detect_all_setups
from ..models import SetupSignal, SETUP_NAMES
from ..backtest.resolver import build_screener_setup, fetch_and_resolve, TradeOutcome
from ..backtest.metrics import compute_metrics, MetricsReport
from .models import Strategy


def _get_replay_timestamps(
    conn: sqlite3.Connection,
    since: str,
    until: str,
    coins: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Get distinct (coin, timestamp) pairs from cached 4H candles in range.

    Returns sorted list of (coin, timestamp) tuples.
    """
    clauses = ["interval = '4h'", "timestamp >= ?", "timestamp <= ?"]
    params: list = [since, until]

    if coins:
        placeholders = ",".join("?" for _ in coins)
        clauses.append(f"coin IN ({placeholders})")
        params.extend(coins)

    where = " AND ".join(clauses)
    rows = conn.execute(
        f"""SELECT DISTINCT coin, timestamp FROM candles
        WHERE {where} ORDER BY timestamp, coin""",
        params,
    ).fetchall()
    return [(r["coin"], r["timestamp"]) for r in rows]


def _get_trailing_candles(
    coin: str,
    timestamp: str,
    lookback: int,
    conn: sqlite3.Connection,
    exchange: str = "hyperliquid",
) -> pd.DataFrame | None:
    """Load trailing `lookback` 4H candles ending at `timestamp` from cache."""
    rows = conn.execute(
        """SELECT timestamp, open, high, low, close, volume
        FROM candles
        WHERE coin = ? AND interval = '4h' AND timestamp <= ?
        AND exchange = ?
        ORDER BY timestamp DESC LIMIT ?""",
        (coin, timestamp, exchange, lookback),
    ).fetchall()

    if len(rows) < 50:  # Need minimum bars for indicators
        return None

    df = pd.DataFrame(
        [dict(r) for r in reversed(rows)],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def _detect_setups(
    strategy: Strategy,
    ts_groups: dict[str, list[str]],
    conn: sqlite3.Connection,
    verbose: bool = False,
) -> list[tuple[SetupSignal, str]]:
    """Run screener on cached 4H candles for each timestamp group."""
    all_pairs: list[tuple[SetupSignal, str]] = []

    for ts_idx, (timestamp, ts_coins) in enumerate(sorted(ts_groups.items())):
        for coin in ts_coins:
            df = _get_trailing_candles(coin, timestamp, 200, conn)
            if df is None:
                continue

            try:
                ctx = build_context(df, strategy.swing_lookback, strategy.sr_cluster_pct)
                signals = detect_all_setups(
                    coin, df, ctx,
                    strategy.max_setups_per_coin,
                    strategy.proximity_atr_mult,
                    disabled_types=strategy.disabled_types_set,
                )
                for sig in signals:
                    all_pairs.append((sig, timestamp))
            except (ValueError, KeyError, IndexError) as e:
                print(f"  {coin}@{timestamp}: {e}", file=sys.stderr)

        if verbose and (ts_idx + 1) % 10 == 0:
            print(f"  Processed {ts_idx + 1}/{len(ts_groups)} timestamps, "
                  f"{len(all_pairs)} signals so far")

    return all_pairs


def _resolve_replay_signals(
    all_pairs: list[tuple[SetupSignal, str]],
    strategy: Strategy,
    conn: sqlite3.Connection,
    timeout_bars: int = 72,
    fee_rate: float = 0.00035,
    verbose: bool = False,
) -> MetricsReport | None:
    """Walk-forward resolve replay signals and compute metrics."""
    if verbose:
        print(f"Resolving {len(all_pairs)} signals...")

    outcomes: list[TradeOutcome] = []
    for i, (sig, ts) in enumerate(all_pairs):
        ctx = sig.context
        row = {
            "key_level": sig.key_level,
            "atr": ctx.atr,
            "direction": sig.direction.value,
            "coin": sig.coin,
            "timestamp": ts,
            "setup_type": sig.setup_type.value,
            "confidence": sig.confidence,
            "structure": ctx.structure.value,
        }
        setup = build_screener_setup(
            row, strategy.atr_stop_mult, strategy.atr_target_mult,
        )
        if not setup:
            continue

        try:
            outcome = fetch_and_resolve(
                setup, conn, timeout_bars, fee_rate,
            )
            outcomes.append(outcome)
        except (ValueError, KeyError, IndexError, OSError) as e:
            print(f"  Resolve {setup.coin} {setup.setup_type}: {e}", file=sys.stderr)

        if verbose and (i + 1) % 20 == 0:
            print(f"  Resolved {i + 1}/{len(all_pairs)}")

    if not outcomes:
        return None

    metrics = compute_metrics(outcomes, f"strategy:{strategy.strategy_id}")
    if verbose:
        print(f"Resolved {len(outcomes)}/{len(all_pairs)} signals")
    return metrics


def replay(
    strategy: Strategy,
    since: str,
    until: str,
    conn: sqlite3.Connection,
    coins: list[str] | None = None,
    resolve: bool = False,
    timeout_bars: int = 72,
    fee_rate: float = 0.00035,
    verbose: bool = False,
) -> tuple[list[dict], MetricsReport | None]:
    """Re-run screener on cached 4H candles for a given strategy.

    Args:
        strategy: Strategy config to use for screening.
        since/until: ISO date strings for the replay window.
        conn: DB connection with cached candles.
        coins: Optional coin filter.
        resolve: If True, also walk-forward resolve signals and compute metrics.
        timeout_bars: Bars before EXPIRED (for resolution).
        fee_rate: Taker fee per side.
        verbose: Print progress.

    Returns:
        (signals_list, MetricsReport or None)
    """
    # Dedup: clear any existing replay signals for this strategy
    conn.execute(
        "DELETE FROM signals WHERE strategy_id = ?",
        (strategy.strategy_id,),
    )
    conn.commit()

    # Get unique timestamps per coin in the range
    points = _get_replay_timestamps(conn, since, until, coins)
    if verbose:
        unique_coins = len(set(c for c, _ in points))
        unique_ts = len(set(t for _, t in points))
        print(f"Replay: {len(points)} data points ({unique_coins} coins x {unique_ts} timestamps)")

    # Group by timestamp for batch processing
    ts_groups: dict[str, list[str]] = {}
    for coin, ts in points:
        ts_groups.setdefault(ts, []).append(coin)

    # Detect setups
    all_pairs = _detect_setups(strategy, ts_groups, conn, verbose)
    batch_id = uuid4().hex[:8]

    # Persist signals
    if all_pairs:
        _save_replay_signals(all_pairs, batch_id, strategy.strategy_id, conn)
    if verbose:
        print(f"Replay complete: {len(all_pairs)} signals saved")

    # Build return dicts
    signal_rows = []
    for sig, ts in all_pairs:
        ctx = sig.context
        signal_rows.append({
            "coin": sig.coin,
            "setup_type": sig.setup_type.value,
            "setup_name": SETUP_NAMES.get(sig.setup_type, ""),
            "direction": sig.direction.value,
            "confidence": sig.confidence,
            "key_level": sig.key_level,
            "current_price": ctx.current_price,
            "atr": ctx.atr,
            "structure": ctx.structure.value,
            "timestamp": ts,
        })

    # Walk-forward resolution
    metrics = None
    if resolve and all_pairs:
        metrics = _resolve_replay_signals(
            all_pairs, strategy, conn, timeout_bars, fee_rate, verbose,
        )

    return signal_rows, metrics


def _save_replay_signals(
    pairs: list[tuple[SetupSignal, str]],
    batch_id: str,
    strategy_id: str,
    conn: sqlite3.Connection,
) -> None:
    """Insert replay signals with their candle timestamps (not wall-clock time)."""
    now = datetime.now(timezone.utc).isoformat()
    total = len(pairs)

    rows = []
    for rank, (signal, candle_ts) in enumerate(pairs, 1):
        ctx = signal.context
        fib = ctx.fib_levels
        range_low, range_high = ctx.range_bounds if ctx.range_bounds else (None, None)

        rows.append((
            uuid4().hex[:8],
            batch_id,
            signal.coin,
            signal.setup_type.value,
            SETUP_NAMES.get(signal.setup_type, ""),
            signal.direction.value,
            signal.confidence,
            signal.key_level,
            signal.description,
            ctx.current_price,
            ctx.atr,
            ctx.structure.value,
            range_low,
            range_high,
            fib.fib_618 if fib else None,
            fib.fib_786 if fib else None,
            0,  # sent_to_llm
            rank,
            total,
            candle_ts,  # timestamp = when signal was historically detected
            now,         # created_at = when replay was run
            strategy_id,
        ))

    conn.executemany(
        """INSERT INTO signals
        (signal_id, batch_id, coin, setup_type, setup_name, direction,
         confidence, key_level, description, current_price, atr, structure,
         range_low, range_high, fib_618, fib_786,
         sent_to_llm, rank_in_batch, total_in_batch, timestamp, created_at,
         strategy_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
