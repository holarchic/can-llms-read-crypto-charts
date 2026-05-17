"""T-016 C4#6 — transaction() context manager + _maybe_commit ownership.

Verifies the commit-boundary contract: one commit on success, one rollback on
BaseException, re-entrant single rollback unit, no id-leak after exceptions,
foreign-uncommitted-txn fail-fast, standalone _maybe_commit unchanged.

Uses a file DB + a SEPARATE probe connection so only *committed* rows are
observed (same-connection reads see uncommitted writes — read-your-own-writes).
"""

import sqlite3

import pytest

from scanner.execution.storage import (
    _TX_DEPTH,
    ensure_orders_table,
    save_order,
    transaction,
)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "c46.db"
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    ensure_orders_table(conn)
    conn.commit()
    return conn, str(path)


def _order(oid):
    return {
        "order_id": oid, "scan_id": "s1", "coin": "ABC", "direction": "LONG",
        "order_type": "entry", "status": "FILLED", "fill_price": 100.0,
        "quantity": 1.0, "exchange": "hyperliquid", "execution_mode": "testnet",
        "created_at": "2026-01-01T00:00:00+00:00",
    }


def _committed(path):
    """Rows visible from a fresh connection = only what was COMMITTED."""
    p = sqlite3.connect(path, timeout=5)
    try:
        p.execute("PRAGMA busy_timeout=5000")
        return p.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    finally:
        p.close()


def test_commit_on_success_and_no_leak(db):
    conn, path = db
    with transaction(conn):
        save_order(_order("a"), conn)
        save_order(_order("b"), conn)
    assert _committed(path) == 2
    assert id(conn) not in _TX_DEPTH


def test_rollback_on_exception_conn_usable_after(db):
    conn, path = db
    with pytest.raises(ValueError):
        with transaction(conn):
            save_order(_order("a"), conn)
            raise ValueError("boom")
    assert _committed(path) == 0  # full rollback, partial write discarded
    assert id(conn) not in _TX_DEPTH  # finally-cleaned, no leak
    # conn still usable; subsequent standalone write commits (proves _owned()
    # is False again — _maybe_commit no longer suppressed)
    save_order(_order("c"), conn)
    assert _committed(path) == 1


def test_rollback_on_baseexception(db):
    conn, path = db
    with pytest.raises(KeyboardInterrupt):
        with transaction(conn):
            save_order(_order("a"), conn)
            raise KeyboardInterrupt  # BaseException, not Exception
    assert _committed(path) == 0
    assert id(conn) not in _TX_DEPTH


def test_reentrant_one_commit_boundary(db):
    conn, path = db
    with transaction(conn):
        save_order(_order("a"), conn)
        with transaction(conn):
            save_order(_order("b"), conn)
        assert _committed(path) == 0  # inner did NOT commit independently
    assert _committed(path) == 2  # outer is the single commit boundary
    assert id(conn) not in _TX_DEPTH


def test_reentrant_inner_failure_rolls_back_both(db):
    conn, path = db
    with pytest.raises(ValueError):
        with transaction(conn):
            save_order(_order("a"), conn)
            with transaction(conn):
                save_order(_order("b"), conn)
                raise ValueError
    assert _committed(path) == 0  # one rollback unit
    assert id(conn) not in _TX_DEPTH


def test_foreign_uncommitted_txn_fails_fast(db):
    conn, _ = db
    conn.execute(
        "INSERT INTO orders (order_id, scan_id, coin, direction, order_type, "
        "status, quantity, exchange, execution_mode, created_at) "
        "VALUES ('x','s','C','LONG','entry','FILLED',1,'hyperliquid','testnet','t')"
    )
    assert conn.in_transaction  # default isolation opened an implicit txn
    with pytest.raises(RuntimeError, match="foreign transaction"):
        with transaction(conn):
            pass
    assert id(conn) not in _TX_DEPTH  # guard fired before ownership recorded


def test_standalone_maybe_commit_persists(db):
    conn, path = db
    save_order(_order("a"), conn)  # no transaction() → _maybe_commit commits
    assert _committed(path) == 1
    assert id(conn) not in _TX_DEPTH
