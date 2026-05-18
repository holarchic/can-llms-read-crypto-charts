"""T-016 C4#6 Chunk 2b/2c — atomic full-close boundary.

The load-bearing property: a close-state write and its counterpart CANCELLED
marks commit in ONE DB transaction AFTER all exchange calls, so a mid-cluster
DB failure rolls the close BACK → the position stays in
`get_open_entry_orders()` → the next monitor cycle retries (self-healing,
DB-failure scoped). Naively committing the close before the marks would strand
PLACED counterparts on a position never revisited — the phantom_sl family
C4#6 exists to kill.

Covers all three close clusters (sync filled-exit, timeout, drift), the
happy-path frozen behaviour (close row + counterpart CANCELLED, one unit), and
the 2c unowned-foreign-txn guard at a real close call site.
"""

import sqlite3
from unittest.mock import patch

import pytest

from scanner.execution import drift as D
from scanner.execution import monitor as M
from scanner.execution.config import ExecutionConfig
from scanner.execution.storage import (
    _TX_DEPTH,
    ensure_orders_table,
    get_open_entry_orders,
    save_order,
)


def _cfg():
    return ExecutionConfig(exchange="hyperliquid", mode="testnet")


@pytest.fixture
def db(tmp_path):
    """File DB so a SEPARATE probe connection observes only COMMITTED rows
    (same-conn reads see uncommitted writes — read-your-own-writes)."""
    path = tmp_path / "c46_2b.db"
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    ensure_orders_table(conn)
    conn.commit()
    return conn, str(path)


def _seed_position(conn, *, sl_status="PLACED", sl_exch_oid="SL_OID"):
    """One FILLED entry (qty 1) + one PLACED SL counterpart (qty 1)."""
    save_order({
        "order_id": "e1", "scan_id": "s1", "coin": "ABC", "direction": "LONG",
        "order_type": "entry", "status": "FILLED", "fill_price": 100.0,
        "quantity": 1.0, "exchange": "hyperliquid", "execution_mode": "testnet",
        "created_at": "2020-01-01T00:00:00+00:00",
        "filled_at": "2020-01-01T00:00:00+00:00",
    }, conn)
    save_order({
        "order_id": "sl1", "scan_id": "s1", "coin": "ABC", "direction": "LONG",
        "order_type": "stop_loss", "status": sl_status, "trigger_price": 95.0,
        "exchange_order_id": sl_exch_oid, "quantity": 1.0,
        "exchange": "hyperliquid", "execution_mode": "testnet",
        "created_at": "2020-01-01T00:00:00+00:00",
    }, conn)
    conn.commit()


def _probe(path, sql, params=()):
    p = sqlite3.connect(path, timeout=5)
    try:
        p.row_factory = sqlite3.Row
        p.execute("PRAGMA busy_timeout=5000")
        return p.execute(sql, params).fetchall()
    finally:
        p.close()


def _full_result(order_type):
    return M.CloseResult(
        "full",
        {"order_id": "c1", "scan_id": "s1", "coin": "ABC", "direction": "LONG",
         "order_type": order_type, "status": "FILLED", "fill_price": 99.0,
         "quantity": 1.0, "exchange": "hyperliquid", "execution_mode": "testnet",
         "exchange_order_id": "EXCH1",
         "created_at": "2020-01-02T00:00:00+00:00",
         "filled_at": "2020-01-02T00:00:00+00:00"},
        99.0,
    )


# --- injected mid-cluster failure → full rollback + position stays open -----


def test_sync_full_close_midcluster_failure_rolls_back_keeps_open(db):
    """_handle_exit_fill full route: exit→FILLED staged, counterpart-mark
    raises → the whole cluster rolls back. The SL fill is unrecorded so the
    entry stays in get_open_entry_orders() and the next cycle retries."""
    conn, path = db
    _seed_position(conn)
    sl = {"order_id": "sl1", "scan_id": "s1", "coin": "ABC",
          "direction": "LONG", "order_type": "stop_loss", "quantity": 1.0,
          "status": "PLACED", "trigger_price": 95.0}
    entry = {"scan_id": "s1", "coin": "ABC", "direction": "LONG", "quantity": 1.0}

    with patch.object(M, "_cancel_counterparts_on_exchange", return_value=[]), \
         patch.object(M, "_mark_counterparts_cancelled",
                      side_effect=RuntimeError("db boom")):
        with pytest.raises(RuntimeError, match="db boom"):
            M._handle_exit_fill(sl, entry, [sl], "ABC/USDC:USDC",
                                object(), _cfg(), conn,
                                fill_price=95.0, fill_qty=1.0)

    # Cluster rolled back: the SL is NOT FILLED on disk.
    assert _probe(path, "SELECT status FROM orders WHERE order_id='sl1'")[0]["status"] == "PLACED"
    # Self-healing: entry still in the open set (close unrecorded).
    assert {e["scan_id"] for e in get_open_entry_orders(conn)} == {"s1"}
    assert id(conn) not in _TX_DEPTH  # ownership cleaned despite the exception


class _SyncExch:
    """fetch_order reports the SL as a confirmed full fill; everything else
    is a no-op. amount_to_precision passthrough."""

    def amount_to_precision(self, symbol, qty):
        return qty

    def fetch_order(self, oid, symbol):
        return {"id": oid, "status": "closed", "filled": 1.0, "average": 99.0}


def test_sync_order_status_midcluster_failure_no_counterpart_leak(db):
    """CALLER-LEVEL regression (R2 finding): a rolled-back close cluster must
    NOT let sync_order_status keep iterating the stale scan_orders and commit
    a later counterpart (tp1) CANCELLED OUTSIDE the rolled-back txn. The scan
    is aborted; SL stays PLACED, tp1 stays PLACED, the position stays open."""
    conn, path = db
    _seed_position(conn, sl_exch_oid="SL_OID")
    save_order({
        "order_id": "tp1", "scan_id": "s1", "coin": "ABC", "direction": "LONG",
        "order_type": "tp1", "status": "PLACED", "trigger_price": 110.0,
        "exchange_order_id": "TP_OID", "quantity": 1.0,
        "exchange": "hyperliquid", "execution_mode": "testnet",
        "created_at": "2020-01-01T00:00:00+00:00",
    }, conn)
    conn.commit()
    cfg = ExecutionConfig(exchange="hyperliquid", mode="testnet")

    with patch.object(M, "get_auth_exchange", return_value=_SyncExch()), \
         patch.object(M, "get_exchange_meta",
                      return_value={"symbol_template": "{coin}/USDC:USDC"}), \
         patch.object(M, "_send_fill_alert", return_value=True), \
         patch.object(M, "cancel_order", return_value=True), \
         patch.object(M, "_mark_counterparts_cancelled",
                      side_effect=RuntimeError("db boom")):
        # exchange_open_orders=[] → both SL and tp1 are "disappeared".
        M.sync_order_status(cfg, conn, exchange_open_orders=[])

    # Cluster rolled back AND the scan aborted before tp1 was reached:
    rows = {r["order_id"]: r["status"]
            for r in _probe(path, "SELECT order_id, status FROM orders")}
    assert rows["sl1"] == "PLACED"   # close rolled back
    assert rows["tp1"] == "PLACED"   # NOT leaked to CANCELLED outside the txn
    assert {e["scan_id"] for e in get_open_entry_orders(conn)} == {"s1"}
    assert id(conn) not in _TX_DEPTH


def _seed_tp1(conn, qty=1.0):
    save_order({
        "order_id": "tp1", "scan_id": "s1", "coin": "ABC", "direction": "LONG",
        "order_type": "tp1", "status": "PLACED", "trigger_price": 110.0,
        "exchange_order_id": "", "quantity": qty,
        "exchange": "hyperliquid", "execution_mode": "testnet",
        "created_at": "2020-01-01T00:00:00+00:00",
    }, conn)
    conn.commit()


def test_full_size_tp1_close_midcluster_failure_rolls_back_keeps_open(db):
    """R2-arch HIGH: a full-size TP1 (zero remaining) is a FULL close. The TP1
    FILLED mark MUST commit atomically with the counterpart cancels — a
    mid-cluster failure rolls the FILLED back so the position is NOT silently
    closed with a stranded PLACED SL (the live counter-trend single-TP path)."""
    conn, path = db
    _seed_position(conn, sl_exch_oid="")          # SL counterpart, no exch oid
    _seed_tp1(conn, qty=1.0)                       # full-size TP1 = entry qty
    tp1 = {"order_id": "tp1", "scan_id": "s1", "coin": "ABC",
           "direction": "LONG", "order_type": "tp1", "quantity": 1.0,
           "status": "PLACED", "trigger_price": 110.0}
    entry = {"scan_id": "s1", "coin": "ABC", "direction": "LONG", "quantity": 1.0}

    with patch.object(M, "_get_live_position_qty", return_value=0.0), \
         patch.object(M, "_mark_counterparts_cancelled",
                      side_effect=RuntimeError("db boom")):
        with pytest.raises(RuntimeError, match="db boom"):
            M._handle_exit_fill(tp1, entry, [tp1], "ABC/USDC:USDC",
                                object(), _cfg(), conn,
                                fill_price=110.0, fill_qty=1.0)

    rows = {r["order_id"]: r["status"]
            for r in _probe(path, "SELECT order_id, status FROM orders")}
    assert rows["tp1"] == "PLACED"   # FILLED rolled back — NOT a silent close
    assert rows["sl1"] == "PLACED"   # counterpart not stranded on a closed pos
    assert {e["scan_id"] for e in get_open_entry_orders(conn)} == {"s1"}
    assert id(conn) not in _TX_DEPTH


def test_partial_becomes_full_close_midcluster_failure_rolls_back(db):
    """R2-arch HIGH (sibling): a partial SL fill whose live-qty check resolves
    to zero residual is a FULL close — same atomic guarantee as full-size TP1."""
    conn, path = db
    _seed_position(conn, sl_exch_oid="")           # SL is the filled order
    _seed_tp1(conn, qty=1.0)                        # tp1 counterpart
    sl = {"order_id": "sl1", "scan_id": "s1", "coin": "ABC",
          "direction": "LONG", "order_type": "stop_loss", "quantity": 1.0,
          "status": "PLACED", "trigger_price": 95.0}
    entry = {"scan_id": "s1", "coin": "ABC", "direction": "LONG", "quantity": 1.0}

    with patch.object(M, "_get_live_position_qty", return_value=0.0), \
         patch.object(M, "_mark_counterparts_cancelled",
                      side_effect=RuntimeError("db boom")):
        with pytest.raises(RuntimeError, match="db boom"):
            # fill_qty 0.4 < order qty 1.0 → partial route → residual 0.6 →
            # _get_live_position_qty patched to 0 → resolves to FULL close.
            M._handle_exit_fill(sl, entry, [sl], "ABC/USDC:USDC",
                                object(), _cfg(), conn,
                                fill_price=95.0, fill_qty=0.4)

    rows = {r["order_id"]: r["status"]
            for r in _probe(path, "SELECT order_id, status FROM orders")}
    assert rows["sl1"] == "PLACED"   # FILLED rolled back
    assert rows["tp1"] == "PLACED"   # counterpart not stranded
    assert {e["scan_id"] for e in get_open_entry_orders(conn)} == {"s1"}
    assert id(conn) not in _TX_DEPTH


def test_timeout_full_close_midcluster_failure_rolls_back_keeps_open(db):
    conn, path = db
    _seed_position(conn)
    cfg = ExecutionConfig(exchange="hyperliquid", mode="testnet",
                          position_timeout_hours=1)

    with patch.object(M, "get_auth_exchange", return_value=object()), \
         patch.object(M, "get_exchange_meta",
                      return_value={"symbol_template": "{coin}/USDC:USDC"}), \
         patch.object(M, "market_close_position",
                      return_value=_full_result("timeout_close")), \
         patch.object(M, "_cancel_counterparts_on_exchange", return_value=[]), \
         patch.object(M, "_mark_counterparts_cancelled",
                      side_effect=RuntimeError("db boom")):
        with pytest.raises(RuntimeError, match="db boom"):
            M.enforce_position_timeouts(cfg, conn)

    assert _probe(path, "SELECT * FROM orders WHERE order_type='timeout_close'") == []
    assert {e["scan_id"] for e in get_open_entry_orders(conn)} == {"s1"}
    assert id(conn) not in _TX_DEPTH


def test_drift_full_close_midcluster_failure_rolls_back_keeps_open(db):
    conn, path = db
    _seed_position(conn)
    entry = {"scan_id": "s1", "coin": "ABC", "direction": "LONG",
             "quantity": 1.0, "fill_price": 100.0}

    with patch.object(D, "market_close_position",
                      return_value=_full_result("drift_close")), \
         patch.object(D, "_cancel_counterparts_on_exchange", return_value=[]), \
         patch.object(D, "_mark_counterparts_cancelled",
                      side_effect=RuntimeError("db boom")):
        with pytest.raises(RuntimeError, match="db boom"):
            D._execute_force_close(entry, "reversal", {}, "ABC/USDC:USDC",
                                   object(), _cfg(), conn)

    assert _probe(path, "SELECT * FROM orders WHERE order_type='drift_close'") == []
    assert {e["scan_id"] for e in get_open_entry_orders(conn)} == {"s1"}
    assert id(conn) not in _TX_DEPTH


# --- happy path: close row + counterpart CANCELLED commit as ONE unit -------


def test_drift_full_close_commits_close_and_cancel_atomically(db):
    conn, path = db
    _seed_position(conn, sl_exch_oid="")  # no exch oid → exchange-cancel skipped
    entry = {"scan_id": "s1", "coin": "ABC", "direction": "LONG",
             "quantity": 1.0, "fill_price": 100.0}

    with patch.object(D, "market_close_position",
                      return_value=_full_result("drift_close")), \
         patch.object(D, "_send_drift_close_alert"):
        rc = D._execute_force_close(entry, "reversal", {}, "ABC/USDC:USDC",
                                    object(), _cfg(), conn)

    assert rc == 1
    # Both the close row AND the counterpart cancel are durably committed.
    close = _probe(path, "SELECT status, quantity FROM orders WHERE order_type='drift_close'")
    assert len(close) == 1 and close[0]["status"] == "FILLED"
    assert _probe(path, "SELECT status FROM orders WHERE order_id='sl1'")[0]["status"] == "CANCELLED"
    # Position now closed (close row's filled qty covers the entry).
    assert get_open_entry_orders(conn) == []
    assert id(conn) not in _TX_DEPTH


# --- 2c: a real close call site never trips the foreign-txn guard -----------


def test_2c_full_close_path_clean_ownership_no_foreign_txn(db):
    """The close `transaction()` is the SOLE owner — it acquires on a conn
    with no foreign uncommitted txn and tears ownership down in finally. A
    clean run raises no RuntimeError and leaves _TX_DEPTH empty (2c guard not
    tripped; current call sites are clean)."""
    conn, path = db
    _seed_position(conn, sl_exch_oid="")
    entry = {"scan_id": "s1", "coin": "ABC", "direction": "LONG",
             "quantity": 1.0, "fill_price": 100.0}

    with patch.object(D, "market_close_position",
                      return_value=_full_result("drift_close")), \
         patch.object(D, "_send_drift_close_alert"):
        D._execute_force_close(entry, "reversal", {}, "ABC/USDC:USDC",
                               object(), _cfg(), conn)

    assert id(conn) not in _TX_DEPTH  # acquired + released cleanly, no leak
