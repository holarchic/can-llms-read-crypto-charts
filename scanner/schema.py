"""Scanner-owned SQLite schema — DDL + migrations (T-016 Phase 2b, SPEC #23).

The schema-bootstrap concern, split out of the former `scanner/storage.py`
kitchen-sink. Composed by the assembly root `scanner.db.get_connection`
(`from .schema import _run_migrations`, function-local). Execution-owned
tables (`orders`/`pending_entries`) are bootstrapped separately by
`scanner.execution.storage.ensure_orders_table` — scanner-core does not
import the execution package (C2#2, SPEC #22). Pure DDL: stdlib only.
"""

import sqlite3

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

CREATE_SHADOW_SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS shadow_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT NOT NULL UNIQUE,
    coin TEXT NOT NULL,
    exchange TEXT NOT NULL DEFAULT 'hyperliquid',
    anchor_ts TEXT NOT NULL,
    setup_type TEXT,
    direction TEXT,
    prompt_version TEXT,
    confluence_count INTEGER,
    machine_block TEXT NOT NULL,
    validator_verdict TEXT
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
    conn.execute(CREATE_SHADOW_SIGNALS_TABLE)  # T-020.1 shadow-mode capture (SPEC #41)
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
