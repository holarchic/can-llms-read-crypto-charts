"""Scan/signal/universe writes (T-016 Phase 2b, SPEC #23).

The write half of the former `scanner/storage.py` kitchen-sink. Reads live in
`scanner.scan_queries`; schema/migrations in `scanner.schema`. The `conn=None`
convenience path calls `scanner.db.get_connection` directly (the 2a
`_default_conn` shim is gone — the `storage↔db` module-load cycle it guarded
is structurally impossible now that `_run_migrations` lives in `scanner.schema`;
matches the post-2a convention in `cli.py`/`pipeline.py`).
"""

import json as _json
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

from .db import get_connection
from .models import SETUP_NAMES, ScanResult, SetupSignal

__all__ = ["save_result", "mark_alerted", "save_signals", "save_scan_universe"]


def save_result(
    result: ScanResult,
    conn: sqlite3.Connection | None = None,
    strategy_id: str = "v0",
    exchange: str = "hyperliquid",
    prompt_version: str | None = None,
) -> None:
    """Save a scan result to the database."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    parsed = result.parsed_decision or {}
    from .models import SETUP_NAMES
    setup_name = SETUP_NAMES.get(result.setup_signal.setup_type, "")

    # Extract LLM direction from machine_block if available.
    # T-016: machine_block is a typed MachineBlock | None (direction is a
    # required field, always present on a constructed block).
    llm_direction = None
    if result.machine_block is not None:
        llm_direction = result.machine_block.direction

    scan_id = result.scan_id
    values = (
        scan_id,
        result.coin,
        result.timestamp.isoformat(),
        result.setup_signal.setup_type.value,
        setup_name,
        result.setup_signal.direction.value,
        llm_direction,
        result.setup_signal.confidence,
        result.setup_signal.key_level,
        result.decision,
        result.confluence_score,
        parsed.get("entry_zone"),
        parsed.get("stop_loss"),
        parsed.get("take_profit"),
        parsed.get("reasoning"),
        result.cost_usd,
        1 if result.alerted else 0,
        strategy_id,
        result.analysis_text,
        parsed.get("leverage"),
        exchange,
        result.llm_confidence,
        _json.dumps(result.machine_block.to_raw()) if result.machine_block else None,
        prompt_version,
        datetime.now(timezone.utc).isoformat(),
    )

    sql = """INSERT INTO scans
        (scan_id, coin, timestamp, setup_type, setup_name, screener_direction,
         llm_direction, screener_confidence, key_level, decision, confluence_score,
         entry_zone, stop_loss, take_profit, reasoning, cost_usd, alerted,
         strategy_id, analysis_text, leverage, exchange, llm_confidence,
         machine_block, prompt_version,
         created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

    # scan_id is set on the ScanResult at construction time (analysis.py).
    # Collision probability of uuid4().hex[:8] is ~1 in 4 billion — if we hit
    # one, it's almost certainly a real reuse, not hash birthday paradox.
    # Treat it as an error rather than silently rewriting result.scan_id
    # (callers cache the scan_id into other rows before save_result returns).
    try:
        conn.execute(sql, values)
    except sqlite3.IntegrityError as exc:
        raise sqlite3.IntegrityError(
            f"scan_id {scan_id} collision — reusing an existing scan_id "
            f"(caller should regenerate at ScanResult creation, not here)"
        ) from exc

    conn.commit()

    if own_conn:
        conn.close()


def mark_alerted(scan_id: str, conn: sqlite3.Connection | None = None) -> None:
    """Mark a scan result as alerted."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    conn.execute("UPDATE scans SET alerted = 1 WHERE scan_id = ?", (scan_id,))
    conn.commit()

    if own_conn:
        conn.close()


def save_signals(
    signals: list[SetupSignal],
    batch_id: str,
    sent_to_llm_count: int,
    conn: sqlite3.Connection | None = None,
    strategy_id: str = "v0",
    btc_structure: str | None = None,
    exchange: str = "hyperliquid",
) -> None:
    """Batch insert all screener signals from a single run."""
    if not signals:
        return

    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    now = datetime.now(timezone.utc).isoformat()
    total = len(signals)

    rows = []
    for rank, signal in enumerate(signals, 1):
        ctx = signal.context
        fib = ctx.fib_levels
        range_low, range_high = ctx.range_bounds if ctx.range_bounds else (None, None)
        sc = signal.score_components

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
            1 if rank <= sent_to_llm_count else 0,
            rank,
            total,
            now,
            now,
            strategy_id,
            sc.base if sc else None,
            sc.alignment if sc else None,
            sc.strength if sc else None,
            sc.proximity if sc else None,
            sc.confluence if sc else None,
            btc_structure,
            ctx.vol_percentile,
            ctx.efficiency_ratio,
            ctx.regime,
            exchange,
        ))

    conn.executemany(
        """INSERT INTO signals
        (signal_id, batch_id, coin, setup_type, setup_name, direction,
         confidence, key_level, description, current_price, atr, structure,
         range_low, range_high, fib_618, fib_786,
         sent_to_llm, rank_in_batch, total_in_batch, timestamp, created_at,
         strategy_id,
         score_base, score_alignment, score_strength, score_proximity,
         score_confluence, btc_structure,
         vol_percentile, efficiency_ratio, regime,
         exchange)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()

    if own_conn:
        conn.close()


def save_scan_universe(
    batch_id: str,
    pairs: list[dict],
    timestamp: str,
    conn: sqlite3.Connection,
    exchange: str = "hyperliquid",
) -> int:
    """Log the full universe of pairs scanned in this batch.

    Each pair dict should have: coin, volume_24h (optional), max_leverage (optional).
    Returns number of rows inserted.
    """
    if not pairs:
        return 0

    rows = [
        (batch_id, p["coin"], p.get("volume_24h"), p.get("max_leverage"), timestamp, exchange)
        for p in pairs
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO scan_universe (batch_id, coin, volume_24h, max_leverage, timestamp, exchange) VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)
