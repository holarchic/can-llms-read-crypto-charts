"""Orders table — CRUD operations for trade execution tracking."""

import logging
import sqlite3
from datetime import datetime, timezone
from uuid import NAMESPACE_OID, uuid4, uuid5

log = logging.getLogger(__name__)

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
EXIT_ORDER_TYPES_SQL = "('stop_loss', 'tp1', 'tp2', 'timeout_close', 'emergency_close', 'drift_close')"

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


def ensure_orders_table(conn: sqlite3.Connection) -> None:
    """Create orders + pending_entries tables if they don't exist."""
    conn.execute(CREATE_ORDERS_TABLE)
    conn.execute(CREATE_ORDERS_INDEX)
    conn.execute(CREATE_ORDERS_OPEN_INDEX)
    conn.execute(CREATE_ORDERS_SCAN_TYPE_INDEX)
    conn.execute(CREATE_PENDING_ENTRIES_TABLE)

    # Migration: add risk_mult column (counter-trend R scaling)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(orders)").fetchall()}
    if "risk_mult" not in cols:
        conn.execute("ALTER TABLE orders ADD COLUMN risk_mult REAL DEFAULT 1.0")

    # Migration: add last_checked_hour column (dedup 1H boundary multi-fire)
    pe_cols = {row[1] for row in conn.execute("PRAGMA table_info(pending_entries)").fetchall()}
    if "last_checked_hour" not in pe_cols:
        conn.execute("ALTER TABLE pending_entries ADD COLUMN last_checked_hour TEXT")

    # Migration: add screener_direction so conditional entries can be flagged
    # counter-trend when the trigger fires (preserves risk scaling + min_rr gate).
    if "screener_direction" not in pe_cols:
        conn.execute("ALTER TABLE pending_entries ADD COLUMN screener_direction TEXT")

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
    conn.commit()


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
    conn.commit()


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


def update_order_quantity(
    order_id: str,
    new_quantity: float,
    new_exchange_order_id: str,
    conn: sqlite3.Connection,
    new_trigger_price: float | None = None,
) -> None:
    """Update order quantity, exchange order ID, and optionally trigger price."""
    if new_trigger_price is not None:
        conn.execute(
            "UPDATE orders SET quantity = ?, exchange_order_id = ?, trigger_price = ? WHERE order_id = ?",
            (new_quantity, new_exchange_order_id, new_trigger_price, order_id),
        )
    else:
        conn.execute(
            "UPDATE orders SET quantity = ?, exchange_order_id = ? WHERE order_id = ?",
            (new_quantity, new_exchange_order_id, order_id),
        )
    conn.commit()


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
    conn.commit()


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
    conn.commit()


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
    conn.commit()
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
    conn.commit()
    return int(row["candles_checked"]) if row else 0


def update_last_checked_hour(
    pending_id: str, hour_key: str, conn: sqlite3.Connection,
) -> None:
    """Set last_checked_hour to prevent re-processing the same candle."""
    conn.execute(
        "UPDATE pending_entries SET last_checked_hour = ? WHERE pending_id = ?",
        (hour_key, pending_id),
    )
    conn.commit()


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
