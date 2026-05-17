"""DB connection + composed schema-bootstrap assembly root (T-016 Phase 2a,
SPEC #22).

This is the single place that composes BOTH subsystems' schema: scanner-owned
tables (`scanner.schema._run_migrations`) and execution-owned tables
(`scanner.execution.storage.ensure_orders_table`). It exists so scanner-core
schema/write modules no longer import `scanner.execution.storage` (C2#2 — a
scanner-core → execution-app upward dependency that originally "quietly folded
execution-table bootstrap into scan-result storage").

Composition uses **function-local** imports inside `get_connection` so there
is no module-load import cycle between `scanner.db`, `scanner.schema`, and
`scanner.execution.storage`.

The `scanner/storage.py` kitchen-sink was split into `scanner/schema.py`
(DDL + migrations), `scanner/scan_store.py` (writes), and
`scanner/scan_queries.py` (reads) in Phase 2b (SPEC #23); 2a created this
connection/schema-bootstrap assembly root.
"""

import sqlite3
import threading
from pathlib import Path

from .config import DB_PATH

# Process-once migration guard. Lives here (with the connection factory it
# guards), moved out of scanner.storage in Phase 2a.
_migrated: set[str] = set()
_migration_lock = threading.Lock()


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Get a database connection, running the composed schema bootstrap once
    per db path per process (scanner schema + execution orders/pending)."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    # busy_timeout: block up to 10s on SQLITE_BUSY so concurrent writers
    # (scanner + monitor + cron reports) don't silently fail their UPDATEs.
    conn.execute("PRAGMA busy_timeout=10000")
    # synchronous=NORMAL is safe with WAL and ~2x faster than FULL on fsync.
    conn.execute("PRAGMA synchronous=NORMAL")

    resolved = str(path.resolve())
    if resolved not in _migrated:
        with _migration_lock:
            if resolved not in _migrated:
                # Function-local imports — no module-load cycle (scanner.db ↔
                # scanner.schema ↔ scanner.execution.storage).
                from .execution.storage import ensure_orders_table
                from .schema import _run_migrations

                _run_migrations(conn)  # scanner-owned schema
                ensure_orders_table(conn)  # execution-owned schema
                conn.commit()
                _migrated.add(resolved)
    return conn
