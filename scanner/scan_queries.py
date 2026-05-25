"""Scan/signal read & reporting queries (T-016 Phase 2b, SPEC #23).

The read half of the former `scanner/storage.py` kitchen-sink. Writes live in
`scanner.scan_store`; schema/migrations in `scanner.schema`. The `conn=None`
convenience path calls `scanner.db.get_connection` directly (the 2a
`_default_conn` shim is gone — see `scanner.scan_store` for the rationale).
"""

import sqlite3

from .db import get_connection
from .models import normalize_setup_type

__all__ = [
    "is_duplicate",
    "daily_cost",
    "stats",
    "signal_history",
    "signal_distribution",
    "signal_stats",
]


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
        conn = get_connection()

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
        conn = get_connection()

    clauses = ["datetime(timestamp) > datetime('now', ?)"]
    params: list = [f"-{days} days"]

    if coin:
        clauses.append("coin = ?")
        params.append(coin.upper())
    if setup_type:
        clauses.append("setup_type = ?")
        params.append(normalize_setup_type(setup_type))
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
        conn = get_connection()

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
        conn = get_connection()

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
