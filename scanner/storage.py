"""SQLite storage for scan results — deduplication, cost tracking, stats."""

import json as _json
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

from .models import SETUP_NAMES, ScanResult, SetupSignal

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS scans (
    scan_id TEXT PRIMARY KEY,
    coin TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    setup_type TEXT NOT NULL,
    setup_name TEXT,
    screener_direction TEXT NOT NULL,
    llm_direction TEXT,
    screener_confidence REAL,
    key_level REAL,
    decision TEXT,
    confluence_score TEXT,
    entry_zone TEXT,
    stop_loss TEXT,
    take_profit TEXT,
    reasoning TEXT,
    cost_usd REAL DEFAULT 0,
    alerted INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
)
"""

CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_scans_dedup
ON scans (coin, setup_type, screener_direction, timestamp)
"""

CREATE_SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS signals (
    signal_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    coin TEXT NOT NULL,
    setup_type TEXT NOT NULL,
    setup_name TEXT,
    direction TEXT NOT NULL,
    confidence REAL,
    key_level REAL,
    description TEXT,
    current_price REAL,
    atr REAL,
    structure TEXT,
    range_low REAL,
    range_high REAL,
    fib_618 REAL,
    fib_786 REAL,
    sent_to_llm INTEGER DEFAULT 0,
    rank_in_batch INTEGER,
    total_in_batch INTEGER,
    timestamp TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""

CREATE_SIGNALS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_signals_lookup
ON signals (coin, setup_type, timestamp)
"""

CREATE_STRATEGIES_TABLE = """
CREATE TABLE IF NOT EXISTS strategies (
    strategy_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    swing_lookback INTEGER NOT NULL DEFAULT 5,
    sr_cluster_pct REAL NOT NULL DEFAULT 0.015,
    proximity_atr_mult REAL NOT NULL DEFAULT 1.5,
    max_setups_per_coin INTEGER NOT NULL DEFAULT 3,
    atr_stop_mult REAL NOT NULL DEFAULT 1.5,
    atr_target_mult REAL NOT NULL DEFAULT 3.0,
    ruleset_version TEXT DEFAULT '',
    is_active INTEGER NOT NULL DEFAULT 0,
    disabled_types TEXT DEFAULT '',
    notes TEXT,
    created_at TEXT NOT NULL
)
"""

CREATE_SCAN_UNIVERSE_TABLE = """
CREATE TABLE IF NOT EXISTS scan_universe (
    batch_id TEXT NOT NULL,
    coin TEXT NOT NULL,
    volume_24h REAL,
    max_leverage INTEGER,
    timestamp TEXT NOT NULL,
    PRIMARY KEY (batch_id, coin)
)
"""

CREATE_STRATEGIES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_signals_strategy
ON signals (strategy_id)
"""


_VALID_TABLES = {"signals", "scans", "strategies", "scan_universe"}
_VALID_COL_TYPES = {"TEXT", "REAL", "INTEGER"}


def _ensure_column(
    conn: sqlite3.Connection, table: str, column: str, col_type: str, default: str,
) -> None:
    """Add a column to an existing table if it doesn't exist yet."""
    if table not in _VALID_TABLES:
        raise ValueError(f"unknown table: {table}")
    if col_type not in _VALID_COL_TYPES:
        raise ValueError(f"unknown column type: {col_type}")
    if not column.isidentifier():
        raise ValueError(f"invalid column name: {column}")
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type} DEFAULT {default}")


def _migrate_scans_direction(conn: sqlite3.Connection) -> None:
    """Rename scans.direction → screener_direction, add llm_direction column."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(scans)").fetchall()}
    if "direction" in cols and "screener_direction" not in cols:
        conn.execute("ALTER TABLE scans RENAME COLUMN direction TO screener_direction")
    if "llm_direction" not in cols:
        conn.execute("ALTER TABLE scans ADD COLUMN llm_direction TEXT DEFAULT NULL")
        # Backfill from machine_block JSON — skip if machine_block column hasn't
        # been added yet (fresh DB bootstrap adds it later in _run_migrations).
        cols_after = {row[1] for row in conn.execute("PRAGMA table_info(scans)").fetchall()}
        if "machine_block" in cols_after:
            conn.execute(
                "UPDATE scans SET llm_direction = json_extract(machine_block, '$.direction') "
                "WHERE machine_block IS NOT NULL AND llm_direction IS NULL"
            )
    conn.commit()


def _default_conn() -> sqlite3.Connection:
    """Connection for the `conn=None` convenience path. Function-local import
    of the assembly root (`scanner.db`) — keeps scanner-core free of a
    module-load dependency on the connection factory / execution schema
    (T-016 Phase 2a, SPEC #22)."""
    from .db import get_connection

    return get_connection()


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Run scanner-owned schema migrations — tables, indexes, columns.

    Execution-owned tables (`orders`/`pending_entries`) are bootstrapped by
    the assembly root `scanner.db.get_connection`, NOT here — scanner-core
    must not import `scanner.execution.storage` (C2#2, SPEC #22).
    """
    conn.execute(CREATE_TABLE)
    # Migrate scans.direction → screener_direction BEFORE CREATE_INDEX, which
    # references screener_direction. Legacy DBs with the old 'direction' column
    # otherwise fail with "no such column: screener_direction".
    _migrate_scans_direction(conn)
    conn.execute(CREATE_INDEX)
    conn.execute(CREATE_SIGNALS_TABLE)
    conn.execute(CREATE_SIGNALS_INDEX)
    conn.execute(CREATE_STRATEGIES_TABLE)
    conn.execute(CREATE_SCAN_UNIVERSE_TABLE)
    # Migrate: add strategy_id to existing tables
    _ensure_column(conn, "signals", "strategy_id", "TEXT", "'v0'")
    _ensure_column(conn, "scans", "strategy_id", "TEXT", "'v0'")
    _ensure_column(conn, "strategies", "ruleset_version", "TEXT", "''")
    _ensure_column(conn, "strategies", "disabled_types", "TEXT", "''")
    _ensure_column(conn, "scans", "analysis_text", "TEXT", "NULL")
    _ensure_column(conn, "scans", "leverage", "TEXT", "NULL")
    # Scoring component columns for calibration experiments
    _ensure_column(conn, "signals", "score_base", "REAL", "NULL")
    _ensure_column(conn, "signals", "score_alignment", "REAL", "NULL")
    _ensure_column(conn, "signals", "score_strength", "REAL", "NULL")
    _ensure_column(conn, "signals", "score_proximity", "REAL", "NULL")
    _ensure_column(conn, "signals", "score_confluence", "REAL", "NULL")
    _ensure_column(conn, "signals", "btc_structure", "TEXT", "NULL")
    # Regime classification columns
    _ensure_column(conn, "signals", "vol_percentile", "REAL", "NULL")
    _ensure_column(conn, "signals", "efficiency_ratio", "REAL", "NULL")
    _ensure_column(conn, "signals", "regime", "TEXT", "NULL")
    # Exchange columns
    _ensure_column(conn, "signals", "exchange", "TEXT", "'hyperliquid'")
    _ensure_column(conn, "scans", "exchange", "TEXT", "'hyperliquid'")
    _ensure_column(conn, "scan_universe", "exchange", "TEXT", "'hyperliquid'")
    _ensure_column(conn, "scans", "llm_confidence", "INTEGER", "NULL")
    _ensure_column(conn, "scans", "machine_block", "TEXT", "NULL")
    _ensure_column(conn, "scans", "prompt_version", "TEXT", "NULL")
    conn.execute(CREATE_STRATEGIES_INDEX)
    conn.commit()


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
        conn = _default_conn()

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


def is_duplicate(
    coin: str,
    setup_type: str,
    direction: str,
    hours: int = 12,
    conn: sqlite3.Connection | None = None,
    exchange: str = "hyperliquid",
) -> bool:
    """Check if the same setup was already detected within `hours` hours."""
    own_conn = conn is None
    if own_conn:
        conn = _default_conn()

    row = conn.execute(
        """SELECT COUNT(*) as cnt FROM scans
        WHERE coin = ? AND setup_type = ? AND screener_direction = ?
        AND exchange = ?
        AND datetime(timestamp) > datetime('now', ?)""",
        (coin, setup_type, direction, exchange, f"-{hours} hours"),
    ).fetchone()

    if own_conn:
        conn.close()

    return row["cnt"] > 0


def mark_alerted(scan_id: str, conn: sqlite3.Connection | None = None) -> None:
    """Mark a scan result as alerted."""
    own_conn = conn is None
    if own_conn:
        conn = _default_conn()

    conn.execute("UPDATE scans SET alerted = 1 WHERE scan_id = ?", (scan_id,))
    conn.commit()

    if own_conn:
        conn.close()


def daily_cost(conn: sqlite3.Connection | None = None) -> float:
    """Sum cost_usd for today (UTC)."""
    own_conn = conn is None
    if own_conn:
        conn = _default_conn()

    row = conn.execute(
        """SELECT COALESCE(SUM(cost_usd), 0) as total FROM scans
        WHERE date(created_at) = date('now')"""
    ).fetchone()

    if own_conn:
        conn.close()

    return float(row["total"])


def stats(conn: sqlite3.Connection | None = None) -> dict:
    """Return scan summary stats."""
    own_conn = conn is None
    if own_conn:
        conn = _default_conn()

    total = conn.execute("SELECT COUNT(*) as cnt FROM scans").fetchone()["cnt"]
    confirms = conn.execute("SELECT COUNT(*) as cnt FROM scans WHERE decision = 'CONFIRM'").fetchone()["cnt"]
    rejects = conn.execute("SELECT COUNT(*) as cnt FROM scans WHERE decision = 'REJECT'").fetchone()["cnt"]
    waits = conn.execute("SELECT COUNT(*) as cnt FROM scans WHERE decision = 'WAIT'").fetchone()["cnt"]
    alerted = conn.execute("SELECT COUNT(*) as cnt FROM scans WHERE alerted = 1").fetchone()["cnt"]
    total_cost = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) as total FROM scans").fetchone()["total"]
    today_cost = daily_cost(conn)

    # Recent scans
    recent = conn.execute(
        """SELECT coin, setup_type, screener_direction, decision, cost_usd, timestamp
        FROM scans ORDER BY created_at DESC LIMIT 10"""
    ).fetchall()

    if own_conn:
        conn.close()

    return {
        "total_scans": total,
        "confirms": confirms,
        "rejects": rejects,
        "waits": waits,
        "alerted": alerted,
        "total_cost_usd": round(total_cost, 4),
        "today_cost_usd": round(today_cost, 4),
        "recent": [dict(r) for r in recent],
    }


# --- Signal persistence for backtesting ---


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
        conn = _default_conn()

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


def signal_history(
    coin: str | None = None,
    setup_type: str | None = None,
    days: int = 7,
    sent_only: bool = False,
    conn: sqlite3.Connection | None = None,
    exchange: str | None = None,
) -> list[dict]:
    """Query signal history with optional filters."""
    own_conn = conn is None
    if own_conn:
        conn = _default_conn()

    clauses = ["datetime(timestamp) > datetime('now', ?)"]
    params: list = [f"-{days} days"]

    if coin:
        clauses.append("coin = ?")
        params.append(coin.upper())
    if setup_type:
        clauses.append("setup_type = ?")
        params.append(setup_type.upper())
    if sent_only:
        clauses.append("sent_to_llm = 1")
    if exchange:
        clauses.append("exchange = ?")
        params.append(exchange)

    where = " AND ".join(clauses)
    rows = conn.execute(
        f"""SELECT * FROM signals WHERE {where}
        ORDER BY timestamp DESC, rank_in_batch ASC""",
        params,
    ).fetchall()

    if own_conn:
        conn.close()

    return [dict(r) for r in rows]


def signal_distribution(
    days: int = 7,
    conn: sqlite3.Connection | None = None,
    exchange: str | None = None,
) -> dict:
    """Signal counts grouped by setup type and direction for distribution reports."""
    own_conn = conn is None
    if own_conn:
        conn = _default_conn()

    clauses = ["datetime(timestamp) > datetime('now', ?)"]
    params: list = [f"-{days} days"]
    if exchange:
        clauses.append("exchange = ?")
        params.append(exchange)
    where = " AND ".join(clauses)

    total = conn.execute(
        f"SELECT COUNT(*) as cnt FROM signals WHERE {where}", params
    ).fetchone()["cnt"]

    batches = conn.execute(
        f"SELECT COUNT(DISTINCT batch_id) as cnt FROM signals WHERE {where}", params
    ).fetchone()["cnt"]

    # By setup type with long/short split
    by_type = conn.execute(
        f"""SELECT setup_type, setup_name, COUNT(*) as count,
            SUM(CASE WHEN direction = 'LONG' THEN 1 ELSE 0 END) as long,
            SUM(CASE WHEN direction = 'SHORT' THEN 1 ELSE 0 END) as short
        FROM signals WHERE {where}
        GROUP BY setup_type ORDER BY setup_type""", params
    ).fetchall()

    # By direction
    by_direction = conn.execute(
        f"""SELECT direction as dir, COUNT(*) as count
        FROM signals WHERE {where}
        GROUP BY direction ORDER BY direction""", params
    ).fetchall()

    if own_conn:
        conn.close()

    return {
        "days": days,
        "total": total,
        "batches": batches,
        "by_type": [dict(r) for r in by_type],
        "by_direction": [dict(r) for r in by_direction],
    }


def signal_stats(
    days: int = 7,
    conn: sqlite3.Connection | None = None,
    exchange: str | None = None,
) -> dict:
    """Aggregate signal stats for backtesting review."""
    own_conn = conn is None
    if own_conn:
        conn = _default_conn()

    clauses = ["datetime(timestamp) > datetime('now', ?)"]
    params: list = [f"-{days} days"]
    if exchange:
        clauses.append("exchange = ?")
        params.append(exchange)
    where = " AND ".join(clauses)

    total = conn.execute(
        f"SELECT COUNT(*) as cnt FROM signals WHERE {where}", params
    ).fetchone()["cnt"]
    sent = conn.execute(
        f"SELECT COUNT(*) as cnt FROM signals WHERE {where} AND sent_to_llm = 1", params
    ).fetchone()["cnt"]
    filtered = total - sent

    # By setup type
    by_type = conn.execute(
        f"""SELECT setup_type, setup_name, COUNT(*) as cnt,
            SUM(sent_to_llm) as sent, AVG(confidence) as avg_conf
        FROM signals WHERE {where}
        GROUP BY setup_type ORDER BY cnt DESC""", params
    ).fetchall()

    # Top coins
    top_coins = conn.execute(
        f"""SELECT coin, COUNT(*) as cnt, SUM(sent_to_llm) as sent,
            AVG(confidence) as avg_conf
        FROM signals WHERE {where}
        GROUP BY coin ORDER BY cnt DESC LIMIT 10""", params
    ).fetchall()

    # Batch count
    batches = conn.execute(
        f"SELECT COUNT(DISTINCT batch_id) as cnt FROM signals WHERE {where}", params
    ).fetchone()["cnt"]

    if own_conn:
        conn.close()

    return {
        "days": days,
        "total_signals": total,
        "sent_to_llm": sent,
        "filtered": filtered,
        "batches": batches,
        "by_type": [dict(r) for r in by_type],
        "top_coins": [dict(r) for r in top_coins],
    }
