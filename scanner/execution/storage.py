"""Orders table — CRUD operations for trade execution tracking."""

import json as _json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import NAMESPACE_OID, uuid4, uuid5

from scanner.models import TradeRecord

log = logging.getLogger(__name__)

# --- C4#6: transaction-scoped application services -------------------------
# `get_connection()` uses sqlite3's DEFAULT isolation (an implicit txn opens on
# the first DML), so `conn.in_transaction` is true after any write and is NOT
# an ownership signal. Track ownership by `id(conn)` in a module map: an entry
# exists only while a `transaction()` owns the connection, during which the
# caller strongly holds the conn object so its id cannot be reused. Standalone
# helper calls commit exactly as before (`_maybe_commit` commits when
# unowned); only a manager-owned cluster defers to the single boundary
# commit/rollback. INVARIANT: never wrap an exchange REST call in a
# transaction() — a fill cannot be rolled back, and a write region must not
# span network I/O.
_TX_DEPTH: dict[int, int] = {}


def _owned(conn: sqlite3.Connection) -> bool:
    return _TX_DEPTH.get(id(conn), 0) > 0


def _maybe_commit(conn: sqlite3.Connection) -> None:
    """Commit unless a `transaction()` context owns this connection's boundary."""
    if not _owned(conn):
        conn.commit()


@contextmanager
def transaction(conn: sqlite3.Connection):
    """Own the commit boundary for a multi-write DB cluster: exactly one
    commit on success, one rollback on `BaseException`. Re-entrant — a nested
    `transaction()` is a no-op scope; the outermost owns commit/rollback so
    the whole nested cluster is one rollback unit.
    """
    cid = id(conn)
    depth = _TX_DEPTH.get(cid, 0)
    if depth > 0:  # re-entrant: an outer transaction() owns the boundary
        _TX_DEPTH[cid] = depth + 1
        try:
            yield
        finally:
            _TX_DEPTH[cid] -= 1
        return
    # Outermost owner. A foreign uncommitted implicit txn (a writer that did
    # not go through _maybe_commit) would be flushed by our commit/rollback —
    # fail fast rather than silently commit or discard its work.
    if conn.in_transaction:
        raise RuntimeError(
            "transaction(): connection has an uncommitted foreign transaction "
            "— a non-_maybe_commit writer left work pending"
        )
    _TX_DEPTH[cid] = 1
    try:
        yield
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        del _TX_DEPTH[cid]

# Valid state transitions — reject anything else.
# FILLED and CANCELLED are terminal. Replacements insert NEW rows instead of
# resurrecting old ones (preserves audit trail, avoids quantity accounting drift).
_VALID_TRANSITIONS: dict[str, set[str]] = {
    "PENDING": {"PLACED", "CANCELLED"},
    "PLACED": {"FILLED", "CANCELLED"},
    "FAILED": {"CANCELLED"},  # Superseded by monitor SL recovery
}

# Minimum qty threshold for "position open" checks.
# Prevents IEEE 754 residuals (e.g. 1e-17) from creating ghost positions.
OPEN_QTY_EPSILON = 1e-9

# All order types that represent position exits (used in "remaining qty" calculations).
EXIT_ORDER_TYPES_SQL = "('stop_loss', 'tp1', 'tp2', 'timeout_close', 'emergency_close', 'drift_close', 'manual_close')"

CREATE_ORDERS_TABLE = """
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    scan_id TEXT NOT NULL,
    coin TEXT NOT NULL,
    direction TEXT NOT NULL,
    order_type TEXT NOT NULL,
    exchange_order_id TEXT,
    status TEXT NOT NULL DEFAULT 'PENDING',
    entry_price REAL,
    fill_price REAL,
    quantity REAL,
    leverage INTEGER,
    trigger_price REAL,
    placed_at TEXT,
    filled_at TEXT,
    cancelled_at TEXT,
    realized_pnl REAL,
    funding_paid REAL,
    exchange TEXT NOT NULL DEFAULT 'hyperliquid',
    execution_mode TEXT NOT NULL DEFAULT 'testnet',
    error_message TEXT,
    created_at TEXT NOT NULL
)
"""

CREATE_ORDERS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_orders_scan
ON orders (scan_id)
"""

# Hot-path query: "open positions" — filters by order_type='entry', status='FILLED',
# exchange, execution_mode. Used by count_open_positions, has_open_position,
# get_open_entry_orders, sum_open_notional, has_unprotected_positions every
# monitor cycle. Grows O(n) with DB size without this composite.
CREATE_ORDERS_OPEN_INDEX = """
CREATE INDEX IF NOT EXISTS idx_orders_open
ON orders (order_type, status, exchange, execution_mode)
"""

# Scoped lookups: "find all orders for this scan of a given type/status".
# Used heavily by sync_order_status, monitor reconciliation paths.
CREATE_ORDERS_SCAN_TYPE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_orders_scan_type
ON orders (scan_id, order_type, status)
"""


CREATE_PENDING_ENTRIES_TABLE = """
CREATE TABLE IF NOT EXISTS pending_entries (
    pending_id TEXT PRIMARY KEY,
    scan_id TEXT NOT NULL UNIQUE,
    coin TEXT NOT NULL,
    direction TEXT NOT NULL,
    machine_block TEXT NOT NULL,
    entry_trigger TEXT NOT NULL,
    entry_trigger_price REAL NOT NULL,
    invalidation_price REAL NOT NULL,
    expiry_candles INTEGER NOT NULL,
    candles_checked INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    exchange TEXT NOT NULL DEFAULT 'hyperliquid',
    execution_mode TEXT NOT NULL DEFAULT 'testnet',
    created_at TEXT NOT NULL,
    resolved_at TEXT
)
"""

CREATE_PENDING_ENTRIES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_pending_status
ON pending_entries (status, exchange, execution_mode)
"""


def _safe_add_column(
    conn: sqlite3.Connection, table: str, column: str, coldef: str,
) -> None:
    """Idempotent ALTER TABLE ADD COLUMN. Tolerates concurrent migrations.

    Two cron jobs racing on a fresh schema change can both see "column
    missing" via PRAGMA table_info and both issue ALTER TABLE. SQLite
    serializes writes but raises OperationalError("duplicate column name")
    on the loser. Swallow that specific error; propagate everything else.
    """
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column in cols:
        return
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coldef}")
    except sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc).lower():
            raise


def ensure_orders_table(conn: sqlite3.Connection) -> None:
    """Create orders + pending_entries tables if they don't exist."""
    conn.execute(CREATE_ORDERS_TABLE)
    conn.execute(CREATE_ORDERS_INDEX)
    conn.execute(CREATE_ORDERS_OPEN_INDEX)
    conn.execute(CREATE_ORDERS_SCAN_TYPE_INDEX)
    conn.execute(CREATE_PENDING_ENTRIES_TABLE)

    # Migration: add risk_mult column (counter-trend R scaling).
    # Guard against race: two cron jobs starting in parallel can both read
    # "column missing" and both issue ALTER — second one raises
    # "duplicate column name". Catch it as a no-op.
    _safe_add_column(conn, "orders", "risk_mult", "REAL DEFAULT 1.0")
    # Migration: add last_checked_hour column (dedup 1H boundary multi-fire)
    _safe_add_column(conn, "pending_entries", "last_checked_hour", "TEXT")
    # Migration: add screener_direction so conditional entries can be flagged
    # counter-trend when the trigger fires (preserves risk scaling + min_rr gate).
    _safe_add_column(conn, "pending_entries", "screener_direction", "TEXT")

    conn.execute(CREATE_PENDING_ENTRIES_INDEX)


def new_order_id() -> str:
    """Generate a short unique order ID."""
    return uuid4().hex[:8]


def make_cloid(order_id: str) -> str:
    """Derive a deterministic 128-bit hex cloid from an internal order ID.

    Same order_id always produces the same cloid, making retries idempotent
    on HyperLiquid (which rejects duplicate cloids).
    """
    return "0x" + uuid5(NAMESPACE_OID, order_id).hex


def save_order(order: dict, conn: sqlite3.Connection) -> None:
    """Insert a new order record."""
    conn.execute(
        """INSERT INTO orders
        (order_id, scan_id, coin, direction, order_type, exchange_order_id,
         status, entry_price, fill_price, quantity, leverage, trigger_price,
         placed_at, filled_at, cancelled_at, realized_pnl, funding_paid,
         exchange, execution_mode, error_message, created_at, risk_mult)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            order["order_id"],
            order["scan_id"],
            order["coin"],
            order["direction"],
            order["order_type"],
            order.get("exchange_order_id"),
            order.get("status", "PENDING"),
            order.get("entry_price"),
            order.get("fill_price"),
            order.get("quantity"),
            order.get("leverage"),
            order.get("trigger_price"),
            order.get("placed_at"),
            order.get("filled_at"),
            order.get("cancelled_at"),
            order.get("realized_pnl"),
            order.get("funding_paid"),
            order.get("exchange", "hyperliquid"),
            order.get("execution_mode", "testnet"),
            order.get("error_message"),
            order.get("created_at", datetime.now(timezone.utc).isoformat()),
            order.get("risk_mult", 1.0),
        ),
    )
    _maybe_commit(conn)


def get_order(order_id: str, conn: sqlite3.Connection) -> dict | None:
    """Fetch a single order by ID."""
    row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    return dict(row) if row else None


def get_orders_for_scan(scan_id: str, conn: sqlite3.Connection) -> list[dict]:
    """Fetch all orders related to a scan."""
    rows = conn.execute(
        """SELECT * FROM orders WHERE scan_id = ?
        ORDER BY created_at,
          CASE order_type
            WHEN 'entry' THEN 0
            WHEN 'stop_loss' THEN 1
            WHEN 'tp1' THEN 2
            WHEN 'tp2' THEN 3
            ELSE 4
          END""",
        (scan_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def update_order_status(
    order_id: str,
    status: str,
    conn: sqlite3.Connection,
    fill_price: float | None = None,
    exchange_order_id: str | None = None,
    error_message: str | None = None,
    realized_pnl: float | None = None,
    funding_paid: float | None = None,
    filled_quantity: float | None = None,
) -> None:
    """Update order status and optional fill/error fields.

    Enforces state machine: PENDING→PLACED→FILLED (terminal), with
    CANCELLED→PLACED allowed for SL replacement flows.
    """
    current = conn.execute(
        "SELECT status FROM orders WHERE order_id = ?", (order_id,),
    ).fetchone()
    if current:
        current_status = current["status"]
        allowed = _VALID_TRANSITIONS.get(current_status, set())
        if status not in allowed:
            log.warning(
                "Invalid transition %s → %s for order %s — skipping",
                current_status, status, order_id,
            )
            return

    now = datetime.now(timezone.utc).isoformat()

    updates = ["status = ?"]
    params: list = [status]

    if fill_price is not None:
        updates.append("fill_price = ?")
        params.append(fill_price)

    if exchange_order_id is not None:
        updates.append("exchange_order_id = ?")
        params.append(exchange_order_id)

    if error_message is not None:
        updates.append("error_message = ?")
        params.append(error_message)

    if realized_pnl is not None:
        updates.append("realized_pnl = ?")
        params.append(realized_pnl)

    if funding_paid is not None:
        updates.append("funding_paid = ?")
        params.append(funding_paid)

    if filled_quantity is not None:
        updates.append("quantity = ?")
        params.append(filled_quantity)

    if status == "FILLED":
        updates.append("filled_at = ?")
        params.append(now)
    elif status == "CANCELLED":
        updates.append("cancelled_at = ?")
        params.append(now)

    params.append(order_id)
    conn.execute(f"UPDATE orders SET {', '.join(updates)} WHERE order_id = ?", params)
    _maybe_commit(conn)


def get_open_entry_orders(
    conn: sqlite3.Connection, exchange: str = "hyperliquid", execution_mode: str = "testnet",
) -> list[dict]:
    """Get filled entry orders with remaining open quantity (partial or full)."""
    rows = conn.execute(
        f"""SELECT e.* FROM orders e
        WHERE e.order_type = 'entry' AND e.status = 'FILLED'
        AND e.exchange = ? AND e.execution_mode = ?
        AND (e.quantity - COALESCE(
            (SELECT SUM(x.quantity) FROM orders x
             WHERE x.scan_id = e.scan_id
             AND x.order_type IN {EXIT_ORDER_TYPES_SQL}
             AND x.status = 'FILLED'),
            0
        )) > {OPEN_QTY_EPSILON}""",
        (exchange, execution_mode),
    ).fetchall()
    return [dict(r) for r in rows]


def count_open_positions(
    conn: sqlite3.Connection, exchange: str = "hyperliquid", execution_mode: str = "testnet",
) -> int:
    """Count positions with remaining open quantity."""
    row = conn.execute(
        f"""SELECT COUNT(*) as cnt FROM orders e
        WHERE e.order_type = 'entry' AND e.status = 'FILLED'
        AND e.exchange = ? AND e.execution_mode = ?
        AND (e.quantity - COALESCE(
            (SELECT SUM(x.quantity) FROM orders x
             WHERE x.scan_id = e.scan_id
             AND x.order_type IN {EXIT_ORDER_TYPES_SQL}
             AND x.status = 'FILLED'),
            0
        )) > {OPEN_QTY_EPSILON}""",
        (exchange, execution_mode),
    ).fetchone()
    return row["cnt"]


def has_open_position(
    coin: str, direction: str | None, conn: sqlite3.Connection,
    exchange: str = "hyperliquid", execution_mode: str = "testnet",
) -> bool:
    """Check if there's already an open position for this coin (optionally filtered by direction)."""
    if direction:
        query = f"""SELECT COUNT(*) as cnt FROM orders e
            WHERE e.order_type = 'entry' AND e.status = 'FILLED'
            AND e.coin = ? AND e.direction = ? AND e.exchange = ?
            AND e.execution_mode = ?
            AND (e.quantity - COALESCE(
                (SELECT SUM(x.quantity) FROM orders x
                 WHERE x.scan_id = e.scan_id
                 AND x.order_type IN {EXIT_ORDER_TYPES_SQL}
                 AND x.status = 'FILLED'),
                0
            )) > {OPEN_QTY_EPSILON}"""
        params = (coin, direction, exchange, execution_mode)
    else:
        query = f"""SELECT COUNT(*) as cnt FROM orders e
            WHERE e.order_type = 'entry' AND e.status = 'FILLED'
            AND e.coin = ? AND e.exchange = ?
            AND e.execution_mode = ?
            AND (e.quantity - COALESCE(
                (SELECT SUM(x.quantity) FROM orders x
                 WHERE x.scan_id = e.scan_id
                 AND x.order_type IN {EXIT_ORDER_TYPES_SQL}
                 AND x.status = 'FILLED'),
                0
            )) > {OPEN_QTY_EPSILON}"""
        params = (coin, exchange, execution_mode)
    row = conn.execute(query, params).fetchone()
    return row["cnt"] > 0


def count_open_longs(
    conn: sqlite3.Connection, exchange: str = "hyperliquid", execution_mode: str = "testnet",
) -> int:
    """Count open LONG positions (for black swan cap)."""
    row = conn.execute(
        f"""SELECT COUNT(*) as cnt FROM orders e
        WHERE e.order_type = 'entry' AND e.status = 'FILLED'
        AND e.direction = 'LONG' AND e.exchange = ?
        AND e.execution_mode = ?
        AND (e.quantity - COALESCE(
            (SELECT SUM(x.quantity) FROM orders x
             WHERE x.scan_id = e.scan_id
             AND x.order_type IN {EXIT_ORDER_TYPES_SQL}
             AND x.status = 'FILLED'),
            0
        )) > {OPEN_QTY_EPSILON}""",
        (exchange, execution_mode),
    ).fetchone()
    return row["cnt"]


def sum_open_notional(
    conn: sqlite3.Connection, exchange: str = "hyperliquid", execution_mode: str = "testnet",
) -> float:
    """Total notional value of remaining open quantity (entry_price * remaining_qty)."""
    row = conn.execute(
        f"""SELECT COALESCE(SUM(e.entry_price * (
            e.quantity - COALESCE(
                (SELECT SUM(x.quantity) FROM orders x
                 WHERE x.scan_id = e.scan_id
                 AND x.order_type IN {EXIT_ORDER_TYPES_SQL}
                 AND x.status = 'FILLED'),
                0)
        )), 0) as total
        FROM orders e
        WHERE e.order_type = 'entry' AND e.status = 'FILLED'
        AND e.exchange = ? AND e.execution_mode = ?
        AND (e.quantity - COALESCE(
            (SELECT SUM(x2.quantity) FROM orders x2
             WHERE x2.scan_id = e.scan_id
             AND x2.order_type IN {EXIT_ORDER_TYPES_SQL}
             AND x2.status = 'FILLED'),
            0
        )) > {OPEN_QTY_EPSILON}""",
        (exchange, execution_mode),
    ).fetchone()
    return float(row["total"])


def get_open_position_coins(
    conn: sqlite3.Connection, exchange: str = "hyperliquid", execution_mode: str = "testnet",
) -> list[dict]:
    """Get coin, direction, remaining_qty, entry_price for open positions."""
    rows = conn.execute(
        f"""SELECT e.coin, e.direction, e.entry_price,
            (e.quantity - COALESCE(
                (SELECT SUM(x.quantity) FROM orders x
                 WHERE x.scan_id = e.scan_id
                 AND x.order_type IN {EXIT_ORDER_TYPES_SQL}
                 AND x.status = 'FILLED'),
                0
            )) as remaining_qty
        FROM orders e
        WHERE e.order_type = 'entry' AND e.status = 'FILLED'
        AND e.exchange = ? AND e.execution_mode = ?
        AND (e.quantity - COALESCE(
            (SELECT SUM(x2.quantity) FROM orders x2
             WHERE x2.scan_id = e.scan_id
             AND x2.order_type IN {EXIT_ORDER_TYPES_SQL}
             AND x2.status = 'FILLED'),
            0
        )) > {OPEN_QTY_EPSILON}""",
        (exchange, execution_mode),
    ).fetchall()
    return [dict(r) for r in rows]


def has_unprotected_positions(
    conn: sqlite3.Connection, exchange: str = "hyperliquid", execution_mode: str = "testnet",
) -> bool:
    """Check if any open position has no PLACED stop_loss — blocks new entries."""
    row = conn.execute(
        f"""SELECT COUNT(*) as cnt FROM orders e
        WHERE e.order_type = 'entry' AND e.status = 'FILLED'
        AND e.exchange = ? AND e.execution_mode = ?
        AND (e.quantity - COALESCE(
            (SELECT SUM(x.quantity) FROM orders x
             WHERE x.scan_id = e.scan_id
             AND x.order_type IN {EXIT_ORDER_TYPES_SQL}
             AND x.status = 'FILLED'), 0)
        ) > {OPEN_QTY_EPSILON}
        AND NOT EXISTS (
            SELECT 1 FROM orders sl
            WHERE sl.scan_id = e.scan_id
            AND sl.order_type = 'stop_loss'
            AND sl.status = 'PLACED'
        )""",
        (exchange, execution_mode),
    ).fetchone()
    return row["cnt"] > 0


# ---------------------------------------------------------------------------
# Closed-trade read model — storage owns the orders/scans read surface
# (T-016 Phase 2a, SPEC #22). analytics.py is pure compute over these records.
# ---------------------------------------------------------------------------

def _exit_price(exit_order: dict) -> float | None:
    """Return an exit fill price or None when both fill_price and trigger_price
    are missing. Explicit None check — 0 is never a valid crypto perp price,
    and `dict.get(k) or fallback` would mask a NULL as 0 and produce phantom PnL
    (see 2026-04-17 DOT incident: manual-close row had no price, silent 0
    fallback computed a fake +$227.50 win on a SHORT).
    """
    for field in ("fill_price", "trigger_price"):
        val = exit_order.get(field)
        if val is not None:
            return float(val)
    return None


def original_stop_price(
    scan_id: str, conn: sqlite3.Connection,
    execution_mode: str, exchange: str,
) -> float | None:
    """First-placed stop_loss trigger price for a scan — the risk anchor for R.

    Single owner of "the original SL for a scan" (T-016 Phase 2a, SPEC #22 /
    KL#2 instance ledger). `ORDER BY created_at ASC` makes this the *original*
    SL: drift `sl_move` and post-TP1 trail insert NEW stop_loss rows (decision
    #13), so a bare `LIMIT 1` (SQLite's undefined row order) can return a
    post-drift/trail SL and mis-state `realized_r`. Returns None when no
    stop_loss row exists (e.g. SL placement failed) — caller clamps to 0.
    """
    row = conn.execute(
        """SELECT trigger_price FROM orders
        WHERE scan_id = ? AND order_type = 'stop_loss'
        AND execution_mode = ? AND exchange = ?
        ORDER BY created_at ASC LIMIT 1""",
        (scan_id, execution_mode, exchange),
    ).fetchone()
    return float(row["trigger_price"]) if row and row["trigger_price"] is not None else None


def _calculate_slippage_r(entry, exit_order, entry_fill, exit_fill, risk_amount, conn) -> float:
    """Calculate slippage in R terms by comparing fills to machine_block expected prices."""
    if risk_amount <= 0:
        return 0

    # Get machine_block from scans table
    scan_row = conn.execute(
        "SELECT machine_block FROM scans WHERE scan_id = ?",
        (entry["scan_id"],),
    ).fetchone()

    if not scan_row or not scan_row["machine_block"]:
        return 0

    try:
        mb = _json.loads(scan_row["machine_block"])
    except (ValueError, TypeError):
        return 0

    expected_entry = (mb.get("entry_low", 0) + mb.get("entry_high", 0)) / 2
    if expected_entry <= 0:
        return 0

    entry_slip = abs(entry_fill - expected_entry) * (entry.get("quantity", 0) or 0)
    return entry_slip / risk_amount if risk_amount > 0 else 0


def get_closed_trades(
    conn, execution_mode: str | None = None, days: int | None = None,
) -> list[TradeRecord]:
    """Join orders table to build trade-level records for closed positions.

    A closed trade = entry FILLED + at least one exit (SL/TP) FILLED.

    The `days` filter is applied to *close time* — the MAX filled_at across
    a scan's FILLED exits — not entry creation time. COALESCE(filled_at,
    created_at) keeps manual-close rows (NULL filled_at) inside the window
    using their insertion timestamp.
    """
    clauses = [
        "e.order_type = 'entry'",
        "e.status = 'FILLED'",
    ]
    params = []

    if execution_mode:
        clauses.append("e.execution_mode = ?")
        params.append(execution_mode)
    if days:
        clauses.append(f"""EXISTS (
            SELECT 1 FROM (
                SELECT scan_id, execution_mode, exchange,
                       MAX(datetime(COALESCE(filled_at, created_at))) AS final_exit_at
                FROM orders
                WHERE order_type IN {EXIT_ORDER_TYPES_SQL} AND status = 'FILLED'
                GROUP BY scan_id, execution_mode, exchange
            ) x
            WHERE x.scan_id = e.scan_id
              AND x.execution_mode = e.execution_mode
              AND x.exchange = e.exchange
              AND datetime(x.final_exit_at) > datetime('now', '-{days} days')
        )""")

    where = " AND ".join(clauses)

    entries = conn.execute(
        f"SELECT e.* FROM orders e WHERE {where} ORDER BY e.created_at",
        params,
    ).fetchall()

    trades = []
    for entry in entries:
        entry = dict(entry)
        # Find ALL filled exits for this scan
        exits = conn.execute(
            f"""SELECT * FROM orders
            WHERE scan_id = ? AND order_type IN {EXIT_ORDER_TYPES_SQL}
            AND status = 'FILLED'
            AND execution_mode = ? AND exchange = ?
            ORDER BY datetime(COALESCE(filled_at, created_at))""",
            (entry["scan_id"], entry["execution_mode"], entry["exchange"]),
        ).fetchall()

        if not exits:
            continue  # Still open

        exits = [dict(x) for x in exits]
        entry_fill = entry.get("fill_price") or entry.get("entry_price") or 0
        entry_qty = entry.get("quantity", 0)
        filled_exit_qty = sum(x.get("quantity", 0) for x in exits)

        # Skip partially open positions (not fully closed yet)
        # Use epsilon to handle IEEE 754 artifacts (e.g. 0.043+0.042 = 0.08499...99)
        if entry_qty - filled_exit_qty > 1e-9:
            continue

        # Sum PnL across all exit legs. Skip the whole trade if any leg lacks
        # a price — silent 0-fallback fabricates PnL (e.g. manual-close rows
        # inserted without a price field).
        realized_pnl = 0
        missing_price = False
        for exit_order in exits:
            exit_fill = _exit_price(exit_order)
            if exit_fill is None:
                log.warning(
                    "Skipping trade %s %s — %s exit has no fill_price or trigger_price",
                    entry["coin"], entry["direction"], exit_order.get("order_type"),
                )
                missing_price = True
                break
            exit_qty = exit_order.get("quantity", 0)
            if entry["direction"] == "LONG":
                realized_pnl += (exit_fill - entry_fill) * exit_qty
            else:
                realized_pnl += (entry_fill - exit_fill) * exit_qty

        if missing_price:
            continue

        # Use last exit for labels and holding hours. `continue` above
        # guarantees every exit has a non-None price, so the assert is a
        # type-narrowing invariant rather than a runtime guard.
        last_exit = exits[-1]
        last_exit_fill = _exit_price(last_exit)
        assert last_exit_fill is not None

        # Funding paid (sum across all orders for this scan)
        funding_row = conn.execute(
            """SELECT COALESCE(SUM(funding_paid), 0) as total FROM orders
            WHERE scan_id = ? AND execution_mode = ? AND exchange = ?""",
            (entry["scan_id"], entry["execution_mode"], entry["exchange"]),
        ).fetchone()
        funding_paid = float(funding_row["total"])

        # Risk amount: uses full entry quantity (that was the original risk).
        # The risk anchor is the ORIGINAL (first-by-created_at) SL — see
        # original_stop_price (T-016 Phase 2a, SPEC #22 / KL#2): a bare
        # LIMIT 1 relied on SQLite's undefined order and could pick a
        # post-drift/trail SL, mis-stating R.
        stop_price = original_stop_price(
            entry["scan_id"], conn, entry["execution_mode"], entry["exchange"],
        )
        if stop_price is None:
            # No usable original SL → risk is UNKNOWN, not full-notional. A
            # `or 0` here synthesised `risk = entry_fill·qty` → near-zero R.
            # risk_amount=0 makes realized_r/slippage_r clamp to 0 and the
            # trade fall out of R-normalized aggregates (SPEC #24, KL#2 v).
            risk_amount = 0
        else:
            stop_distance = abs(entry_fill - stop_price)
            risk_amount = stop_distance * entry_qty if stop_distance > 0 else 0

        # Realized R (scaled by risk_mult for counter-trend position-weighted R)
        realized_r = realized_pnl / risk_amount if risk_amount > 0 else 0
        r_mult = float(entry.get("risk_mult") or 1.0)
        if r_mult < 1.0 - 1e-9:
            realized_r *= r_mult

        # Holding hours (entry to last exit)
        entry_time = entry.get("filled_at") or entry.get("placed_at") or entry.get("created_at")
        exit_time = last_exit.get("filled_at") or last_exit.get("created_at")
        holding_hours = 0
        if entry_time and exit_time:
            try:
                t1 = datetime.fromisoformat(entry_time)
                t2 = datetime.fromisoformat(exit_time)
                holding_hours = (t2 - t1).total_seconds() / 3600
            except (ValueError, TypeError):
                pass

        # Slippage: compare actual fill to machine_block expected
        slippage_r = _calculate_slippage_r(entry, last_exit, entry_fill, last_exit_fill, risk_amount, conn)

        # Map order_type to exit_type label (last exit determines label)
        exit_type_map = {
            "stop_loss": "SL", "tp1": "TP1", "tp2": "TP2",
            "timeout_close": "TIMEOUT", "emergency_close": "EMERGENCY",
            "drift_close": "DRIFT",
        }
        exit_type = exit_type_map.get(last_exit["order_type"], last_exit["order_type"])

        trades.append(TradeRecord(
            scan_id=entry["scan_id"],
            coin=entry["coin"],
            direction=entry["direction"],
            entry_fill=entry_fill,
            exit_fill=last_exit_fill,
            exit_type=exit_type,
            realized_pnl=realized_pnl,
            realized_r=realized_r,
            risk_amount=risk_amount,
            holding_hours=holding_hours,
            funding_paid=funding_paid,
            slippage_r=slippage_r,
            execution_mode=entry.get("execution_mode", "testnet"),
            exchange=entry.get("exchange", "hyperliquid"),
        ))

    return trades


def llm_cost_summary(conn, days: int) -> float:
    """Sum scans.cost_usd over the last N days (by scan created_at)."""
    row = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM scans "
        "WHERE datetime(created_at) > datetime('now', ?)",
        (f"-{days} days",),
    ).fetchone()
    return float(row["total"])


# ---------------------------------------------------------------------------
# Pending entries — conditional entry setups waiting for trigger
# ---------------------------------------------------------------------------

def save_pending_entry(entry: dict, conn: sqlite3.Connection) -> None:
    """Insert a new pending entry record."""
    conn.execute(
        """INSERT INTO pending_entries
        (pending_id, scan_id, coin, direction, machine_block,
         entry_trigger, entry_trigger_price, invalidation_price,
         expiry_candles, exchange, execution_mode, created_at,
         screener_direction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            entry["pending_id"],
            entry["scan_id"],
            entry["coin"],
            entry["direction"],
            entry["machine_block"],
            entry["entry_trigger"],
            entry["entry_trigger_price"],
            entry["invalidation_price"],
            entry["expiry_candles"],
            entry.get("exchange", "hyperliquid"),
            entry.get("execution_mode", "testnet"),
            entry["created_at"],
            entry.get("screener_direction"),
        ),
    )
    _maybe_commit(conn)


def get_active_pending_entries(
    conn: sqlite3.Connection, exchange: str, mode: str,
) -> list[dict]:
    """Fetch all ACTIVE pending entries for a given exchange/mode."""
    rows = conn.execute(
        """SELECT * FROM pending_entries
        WHERE status = 'ACTIVE' AND exchange = ? AND execution_mode = ?
        ORDER BY created_at""",
        (exchange, mode),
    ).fetchall()
    return [dict(r) for r in rows]


def update_pending_status(
    pending_id: str, status: str, conn: sqlite3.Connection,
) -> None:
    """Update pending entry status and set resolved_at timestamp."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE pending_entries SET status = ?, resolved_at = ? WHERE pending_id = ?",
        (status, now, pending_id),
    )
    _maybe_commit(conn)


def claim_pending(
    pending_id: str, new_status: str, conn: sqlite3.Connection,
) -> bool:
    """Atomically flip an ACTIVE pending to `new_status`. Returns True if won.

    Prevents TOCTOU double-execution when scanner's immediate-check races
    with monitor's 1H boundary check — only the winner proceeds past this
    point.
    """
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "UPDATE pending_entries SET status = ?, resolved_at = ? "
        "WHERE pending_id = ? AND status = 'ACTIVE'",
        (new_status, now, pending_id),
    )
    _maybe_commit(conn)
    return cursor.rowcount == 1


def increment_candles_checked(
    pending_id: str, conn: sqlite3.Connection,
) -> int:
    """Atomically increment candles_checked by 1 and return the new value.

    Uses UPDATE ... RETURNING so a concurrent monitor run can't sneak a
    second increment between the UPDATE and the read-back (which would
    report an inflated count and expire the pending prematurely).
    """
    cursor = conn.execute(
        "UPDATE pending_entries SET candles_checked = candles_checked + 1 "
        "WHERE pending_id = ? RETURNING candles_checked",
        (pending_id,),
    )
    row = cursor.fetchone()
    _maybe_commit(conn)
    return int(row["candles_checked"]) if row else 0


def update_last_checked_hour(
    pending_id: str, hour_key: str, conn: sqlite3.Connection,
) -> None:
    """Set last_checked_hour to prevent re-processing the same candle."""
    conn.execute(
        "UPDATE pending_entries SET last_checked_hour = ? WHERE pending_id = ?",
        (hour_key, pending_id),
    )
    _maybe_commit(conn)


def has_active_pending(
    coin: str, conn: sqlite3.Connection, exchange: str, mode: str,
) -> bool:
    """Check if an ACTIVE pending entry already exists for this coin."""
    row = conn.execute(
        """SELECT 1 FROM pending_entries
        WHERE coin = ? AND status = 'ACTIVE' AND exchange = ? AND execution_mode = ?
        LIMIT 1""",
        (coin, exchange, mode),
    ).fetchone()
    return row is not None
