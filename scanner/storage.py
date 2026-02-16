"""SQLite storage for scan results — deduplication, cost tracking, stats."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import DB_PATH
from .models import ScanResult, SetupSignal, SETUP_NAMES


CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS scans (
    scan_id TEXT PRIMARY KEY,
    coin TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    setup_type TEXT NOT NULL,
    setup_name TEXT,
    direction TEXT NOT NULL,
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
ON scans (coin, setup_type, direction, timestamp)
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

CREATE_CANDLES_TABLE = """
CREATE TABLE IF NOT EXISTS candles (
    coin TEXT NOT NULL,
    interval TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (coin, interval, timestamp)
)
"""

CREATE_CANDLES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_candles_lookup
ON candles (coin, interval, timestamp)
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

CREATE_STRATEGIES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_signals_strategy
ON signals (strategy_id)
"""


def _ensure_column(
    conn: sqlite3.Connection, table: str, column: str, col_type: str, default: str,
) -> None:
    """Add a column to an existing table if it doesn't exist yet."""
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type} DEFAULT {default}")


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Get a database connection, creating the table if needed."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute(CREATE_TABLE)
    conn.execute(CREATE_INDEX)
    conn.execute(CREATE_SIGNALS_TABLE)
    conn.execute(CREATE_SIGNALS_INDEX)
    conn.execute(CREATE_CANDLES_TABLE)
    conn.execute(CREATE_CANDLES_INDEX)
    conn.execute(CREATE_STRATEGIES_TABLE)
    # Migrate: add strategy_id to existing tables
    _ensure_column(conn, "signals", "strategy_id", "TEXT", "'v0'")
    _ensure_column(conn, "scans", "strategy_id", "TEXT", "'v0'")
    _ensure_column(conn, "strategies", "ruleset_version", "TEXT", "''")
    _ensure_column(conn, "strategies", "disabled_types", "TEXT", "''")
    _ensure_column(conn, "scans", "analysis_text", "TEXT", "NULL")
    _ensure_column(conn, "scans", "leverage", "TEXT", "NULL")
    conn.execute(CREATE_STRATEGIES_INDEX)
    conn.commit()
    return conn


def save_result(
    result: ScanResult,
    conn: sqlite3.Connection | None = None,
    strategy_id: str = "v0",
) -> None:
    """Save a scan result to the database."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    parsed = result.parsed_decision or {}
    from .models import SETUP_NAMES
    setup_name = SETUP_NAMES.get(result.setup_signal.setup_type, "")

    conn.execute(
        """INSERT OR REPLACE INTO scans
        (scan_id, coin, timestamp, setup_type, setup_name, direction,
         screener_confidence, key_level, decision, confluence_score,
         entry_zone, stop_loss, take_profit, reasoning, cost_usd, alerted,
         strategy_id, analysis_text, leverage, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            result.scan_id,
            result.coin,
            result.timestamp.isoformat(),
            result.setup_signal.setup_type.value,
            setup_name,
            result.setup_signal.direction.value,
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
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()

    if own_conn:
        conn.close()


def is_duplicate(
    coin: str,
    setup_type: str,
    direction: str,
    hours: int = 12,
    conn: sqlite3.Connection | None = None,
) -> bool:
    """Check if the same setup was already detected within `hours` hours."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    row = conn.execute(
        """SELECT COUNT(*) as cnt FROM scans
        WHERE coin = ? AND setup_type = ? AND direction = ?
        AND datetime(timestamp) > datetime('now', ?)""",
        (coin, setup_type, direction, f"-{hours} hours"),
    ).fetchone()

    if own_conn:
        conn.close()

    return row["cnt"] > 0


def mark_alerted(scan_id: str, conn: sqlite3.Connection | None = None) -> None:
    """Mark a scan result as alerted."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    conn.execute("UPDATE scans SET alerted = 1 WHERE scan_id = ?", (scan_id,))
    conn.commit()

    if own_conn:
        conn.close()


def daily_cost(conn: sqlite3.Connection | None = None) -> float:
    """Sum cost_usd for today (UTC)."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

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
        conn = get_connection()

    total = conn.execute("SELECT COUNT(*) as cnt FROM scans").fetchone()["cnt"]
    confirms = conn.execute("SELECT COUNT(*) as cnt FROM scans WHERE decision = 'CONFIRM'").fetchone()["cnt"]
    rejects = conn.execute("SELECT COUNT(*) as cnt FROM scans WHERE decision = 'REJECT'").fetchone()["cnt"]
    waits = conn.execute("SELECT COUNT(*) as cnt FROM scans WHERE decision = 'WAIT'").fetchone()["cnt"]
    alerted = conn.execute("SELECT COUNT(*) as cnt FROM scans WHERE alerted = 1").fetchone()["cnt"]
    total_cost = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) as total FROM scans").fetchone()["total"]
    today_cost = daily_cost(conn)

    # Recent scans
    recent = conn.execute(
        """SELECT coin, setup_type, direction, decision, cost_usd, timestamp
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

    if own_conn:
        conn.close()


def signal_history(
    coin: str | None = None,
    setup_type: str | None = None,
    days: int = 7,
    sent_only: bool = False,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    """Query signal history with optional filters."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    clauses = [f"datetime(timestamp) > datetime('now', '-{days} days')"]
    params: list = []

    if coin:
        clauses.append("coin = ?")
        params.append(coin.upper())
    if setup_type:
        clauses.append("setup_type = ?")
        params.append(setup_type.upper())
    if sent_only:
        clauses.append("sent_to_llm = 1")

    where = " AND ".join(clauses)
    rows = conn.execute(
        f"""SELECT * FROM signals WHERE {where}
        ORDER BY timestamp DESC, rank_in_batch ASC""",
        params,
    ).fetchall()

    if own_conn:
        conn.close()

    return [dict(r) for r in rows]


def signal_stats(
    days: int = 7,
    conn: sqlite3.Connection | None = None,
) -> dict:
    """Aggregate signal stats for backtesting review."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    time_filter = f"datetime(timestamp) > datetime('now', '-{days} days')"

    total = conn.execute(
        f"SELECT COUNT(*) as cnt FROM signals WHERE {time_filter}"
    ).fetchone()["cnt"]
    sent = conn.execute(
        f"SELECT COUNT(*) as cnt FROM signals WHERE {time_filter} AND sent_to_llm = 1"
    ).fetchone()["cnt"]
    filtered = total - sent

    # By setup type
    by_type = conn.execute(
        f"""SELECT setup_type, setup_name, COUNT(*) as cnt,
            SUM(sent_to_llm) as sent, AVG(confidence) as avg_conf
        FROM signals WHERE {time_filter}
        GROUP BY setup_type ORDER BY cnt DESC"""
    ).fetchall()

    # Top coins
    top_coins = conn.execute(
        f"""SELECT coin, COUNT(*) as cnt, SUM(sent_to_llm) as sent,
            AVG(confidence) as avg_conf
        FROM signals WHERE {time_filter}
        GROUP BY coin ORDER BY cnt DESC LIMIT 10"""
    ).fetchall()

    # Batch count
    batches = conn.execute(
        f"SELECT COUNT(DISTINCT batch_id) as cnt FROM signals WHERE {time_filter}"
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
