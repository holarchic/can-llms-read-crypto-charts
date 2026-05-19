"""CRUD for the strategies table — `disabled_types` per `strategy_id`.

`disabled_types` is the only column the live scan reads (via
`get_strategy`). The other `strategies` columns are inert defaulted
carriers (no migration; decision #31): writes here never touch them, so a
`set_disabled_types` leaves `created_at` and the rest of the row intact.
"""

import sqlite3
from datetime import datetime, timezone

from .models import Strategy


def get_strategy(strategy_id: str, conn: sqlite3.Connection) -> Strategy | None:
    """Fetch a strategy's disabled_types by ID."""
    row = conn.execute(
        "SELECT strategy_id, disabled_types FROM strategies WHERE strategy_id = ?",
        (strategy_id,),
    ).fetchone()
    if row is None:
        return None
    return Strategy(strategy_id=row["strategy_id"], disabled_types=row["disabled_types"] or "")


def list_strategies(conn: sqlite3.Connection) -> list[Strategy]:
    """List all strategies (strategy_id + disabled_types), newest first."""
    rows = conn.execute(
        "SELECT strategy_id, disabled_types FROM strategies ORDER BY created_at DESC"
    ).fetchall()
    return [
        Strategy(strategy_id=r["strategy_id"], disabled_types=r["disabled_types"] or "")
        for r in rows
    ]


def set_disabled_types(strategy_id: str, disabled_types: str, conn: sqlite3.Connection) -> None:
    """Set `disabled_types` on an existing strategy row, in place.

    A targeted single-column UPDATE — every other column on the row
    (`created_at`, the inert param columns) is preserved (decision #31; no
    whole-row replace).
    """
    conn.execute(
        "UPDATE strategies SET disabled_types = ? WHERE strategy_id = ?",
        (disabled_types, strategy_id),
    )
    conn.commit()


def ensure_baseline(conn: sqlite3.Connection) -> Strategy:
    """Create the `v0` baseline strategy if absent. Returns it.

    Minimal insert — supplies only the two NOT-NULL-no-default columns
    (`label`, `created_at`); the `NOT NULL DEFAULT` param columns and
    `disabled_types DEFAULT ''` fill themselves. `INSERT OR IGNORE` keeps it
    conflict-tolerant (the prior `INSERT OR REPLACE` was) so a concurrent
    bootstrap of a fresh DB cannot raise a UNIQUE failure.
    """
    conn.execute(
        "INSERT OR IGNORE INTO strategies (strategy_id, label, created_at) VALUES (?, ?, ?)",
        ("v0", "baseline", datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    existing = get_strategy("v0", conn)
    if existing:
        return existing
    return Strategy(strategy_id="v0", disabled_types="")
