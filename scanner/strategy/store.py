"""CRUD operations for the strategies table."""

import sqlite3
from datetime import datetime, timezone

from .models import Strategy


def save_strategy(strategy: Strategy, conn: sqlite3.Connection) -> None:
    """Insert or replace a strategy."""
    conn.execute(
        """INSERT OR REPLACE INTO strategies
        (strategy_id, label, swing_lookback, sr_cluster_pct, proximity_atr_mult,
         max_setups_per_coin, atr_stop_mult, atr_target_mult, ruleset_version,
         is_active, disabled_types, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            strategy.strategy_id,
            strategy.label,
            strategy.swing_lookback,
            strategy.sr_cluster_pct,
            strategy.proximity_atr_mult,
            strategy.max_setups_per_coin,
            strategy.atr_stop_mult,
            strategy.atr_target_mult,
            strategy.ruleset_version,
            1 if strategy.is_active else 0,
            strategy.disabled_types,
            strategy.notes,
            strategy.created_at or datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def _row_to_strategy(row: sqlite3.Row) -> Strategy:
    """Convert a DB row to a Strategy dataclass."""
    return Strategy(
        strategy_id=row["strategy_id"],
        label=row["label"],
        swing_lookback=row["swing_lookback"],
        sr_cluster_pct=row["sr_cluster_pct"],
        proximity_atr_mult=row["proximity_atr_mult"],
        max_setups_per_coin=row["max_setups_per_coin"],
        atr_stop_mult=row["atr_stop_mult"],
        atr_target_mult=row["atr_target_mult"],
        ruleset_version=row["ruleset_version"] or "",
        is_active=bool(row["is_active"]),
        disabled_types=row["disabled_types"] or "",
        notes=row["notes"] or "",
        created_at=row["created_at"],
    )


def get_strategy(strategy_id: str, conn: sqlite3.Connection) -> Strategy | None:
    """Fetch a strategy by ID."""
    row = conn.execute(
        "SELECT * FROM strategies WHERE strategy_id = ?", (strategy_id,)
    ).fetchone()
    return _row_to_strategy(row) if row else None


def get_active_strategy(conn: sqlite3.Connection) -> Strategy | None:
    """Fetch the currently active strategy."""
    row = conn.execute(
        "SELECT * FROM strategies WHERE is_active = 1"
    ).fetchone()
    return _row_to_strategy(row) if row else None


def list_strategies(conn: sqlite3.Connection) -> list[Strategy]:
    """List all strategies, newest first."""
    rows = conn.execute(
        "SELECT * FROM strategies ORDER BY created_at DESC"
    ).fetchall()
    return [_row_to_strategy(r) for r in rows]


def activate_strategy(strategy_id: str, conn: sqlite3.Connection) -> None:
    """Set one strategy as active, deactivating all others."""
    conn.execute("UPDATE strategies SET is_active = 0")
    conn.execute(
        "UPDATE strategies SET is_active = 1 WHERE strategy_id = ?",
        (strategy_id,),
    )
    conn.commit()


def ensure_baseline(conn: sqlite3.Connection) -> Strategy:
    """Create the v0 baseline strategy if it doesn't exist. Returns it."""
    existing = get_strategy("v0", conn)
    if existing:
        return existing

    baseline = Strategy(
        strategy_id="v0",
        label="baseline",
        is_active=True,
    )
    save_strategy(baseline, conn)
    return baseline
