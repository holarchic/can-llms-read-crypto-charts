"""T-016 correctness-fix phase equivalence battery (SPEC #24).

Three fixes, each gated by a by-construction adversarial partition (prod-count-
zero today — the oracle is the fixture, not prod divergence; LESSONS
"Green != verified"):

- Fix 3 (`market_close_position`): never fabricate a full-qty priced close.
- Fix 1 (`drift._execute_force_close`): drift-close alert-R anchored on the
  storage-owned original SL, not a bare `LIMIT 1`.
- Fix 2 (`get_closed_trades` + `compute_live_metrics` + execution-log script):
  no-usable-risk trades excluded from R-normalized aggregates, not silently
  counted as break-even losses.
"""

import sqlite3
from unittest.mock import patch

from scanner.execution import drift as D
from scanner.execution import monitor as M
from scanner.execution.config import ExecutionConfig
from scanner.execution.storage import ensure_orders_table

# ---------------------------------------------------------------------------
# Fix 3 — market_close_position: confirmed-fill gating
# ---------------------------------------------------------------------------


class FakeExchange:
    """Minimal ccxt stand-in. `create_resp`/`fetch_resp` are the order dicts
    returned by create_order / fetch_order; `fetch_raises` forces fetch_order
    to throw (degraded-path test)."""

    def __init__(self, create_resp, fetch_resp=None, fetch_raises=False):
        self._create_resp = create_resp
        self._fetch_resp = fetch_resp
        self._fetch_raises = fetch_raises

    def amount_to_precision(self, symbol, qty):
        return qty

    def fetch_ticker(self, symbol):
        return {"last": 100.0}

    def create_order(self, *a, **k):
        return self._create_resp

    def fetch_order(self, oid, symbol):
        if self._fetch_raises:
            raise RuntimeError("fetch_order boom")
        return self._fetch_resp


def _entry():
    return {"scan_id": "s1", "coin": "ABC", "direction": "LONG"}


def _cfg():
    return ExecutionConfig(exchange="hyperliquid", mode="testnet")


def _conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_orders_table(c)
    return c


def _rows(conn):
    return conn.execute(
        "SELECT status, fill_price, quantity, order_type FROM orders"
    ).fetchall()


def _run(create_resp, fetch_resp=None, fetch_raises=False, qty=1.0):
    conn = _conn()
    ex = FakeExchange(create_resp, fetch_resp, fetch_raises)
    ret = M.market_close_position(
        _entry(), "ABC/USDC:USDC", "sell", qty, ex, _cfg(), conn,
        order_type="drift_close",
    )
    return ret, _rows(conn)


def test_closed_zero_fill_writes_no_row():
    ret, rows = _run({"id": "x", "status": "closed", "filled": 0, "average": 99.0})
    assert ret is None
    assert rows == []


def test_closed_partial_writes_truthful_partial_leg():
    ret, rows = _run(
        {"id": "x", "status": "closed", "filled": 0.5, "average": 99.0}, qty=1.0
    )
    assert ret is None  # caller keeps SL, remainder retried next cycle
    assert len(rows) == 1
    assert rows[0]["status"] == "FILLED"
    assert rows[0]["quantity"] == 0.5  # actual leg, NOT rounded_qty
    assert rows[0]["fill_price"] == 99.0


def test_closed_full_average_returns_price():
    ret, rows = _run(
        {"id": "x", "status": "closed", "filled": 1.0, "average": 99.0}, qty=1.0
    )
    assert ret == 99.0
    assert len(rows) == 1
    assert rows[0]["quantity"] == 1.0
    assert rows[0]["fill_price"] == 99.0


def test_closed_full_price_fallback_when_no_average():
    ret, rows = _run(
        {"id": "x", "status": "closed", "filled": 1.0, "average": None, "price": 98.5},
        qty=1.0,
    )
    assert ret == 98.5
    assert rows[0]["fill_price"] == 98.5


def test_closed_filled_absent_is_full():
    # filled key absent (≠ explicit 0) + closed + priced → full close.
    ret, rows = _run({"id": "x", "status": "closed", "average": 97.0}, qty=2.0)
    assert ret == 97.0
    assert len(rows) == 1
    assert rows[0]["quantity"] == 2.0  # rounded_qty (full), not None


def test_nonclosed_with_echoed_price_writes_no_row():
    # status canceled, filled absent, price PRESENT (echoed IOC limit) — must
    # NOT be treated as a fill just because `price` exists.
    ret, rows = _run({"id": "x", "status": "canceled", "price": 100.0})
    assert ret is None
    assert rows == []


def test_fetch_order_fallback_resolves_close():
    # create resp inconclusive (open, no price) → fetch_order yields the close.
    ret, rows = _run(
        {"id": "x", "status": "open"},
        fetch_resp={"id": "x", "status": "closed", "filled": 1.0, "average": 96.0},
        qty=1.0,
    )
    assert ret == 96.0
    assert rows[0]["fill_price"] == 96.0


def test_fetch_order_raises_writes_no_row():
    ret, rows = _run({"id": "x", "status": "open"}, fetch_raises=True)
    assert ret is None
    assert rows == []


def test_no_row_path_logs_durable_ids(caplog):
    import logging
    with caplog.at_level(logging.ERROR, logger="scanner.execution.monitor"):
        _run({"id": "EXCH99", "status": "canceled", "filled": 0})
    msg = "\n".join(r.message for r in caplog.records)
    assert "NOT confirmed" in msg
    assert "exchange_id=EXCH99" in msg
    assert "order_id=" in msg and "cloid=" in msg


# --- by-construction divergence vs the frozen pre-fix behaviour -------------

def _frozen_old(resp, rounded_qty):
    """Verbatim pre-fix tail: average-only price, UNCONDITIONAL FILLED row at
    full rounded_qty, return `fill_price if fill_price else 0.0`."""
    fp = M._extract_fill_price(resp)
    wrote = {"status": "FILLED", "fill_price": fp, "quantity": rounded_qty}
    return wrote, (fp if fp else 0.0)


def test_adversarial_frozen_old_fabricates_where_new_refuses():
    # status canceled, zero fill, no average: the exact audit-hole input.
    resp = {"id": "x", "status": "canceled", "filled": 0}
    old_row, old_ret = _frozen_old(resp, 1.0)
    assert old_row["status"] == "FILLED"      # OLD: fabricated full-qty close
    assert old_row["fill_price"] is None      # OLD: NULL-price FILLED row
    assert old_ret == 0.0                     # OLD: 0.0 success sentinel

    new_ret, new_rows = _run(resp)
    assert new_rows == []                     # NEW: no fabricated row
    assert new_ret is None                    # NEW: explicit failure
    assert new_ret != old_ret                 # anti-vacuity: provably diverges


# --- caller-level: None return is safe in both production callers ----------


def test_drift_force_close_none_keeps_sl_and_alerts():
    conn = _conn()
    conn.execute(
        """INSERT INTO orders (order_id, scan_id, coin, direction, order_type,
        status, trigger_price, quantity, exchange, execution_mode, created_at)
        VALUES ('sl1','s1','ABC','LONG','stop_loss','PLACED',95.0,1.0,
                'hyperliquid','testnet','2026-01-01T00:00:00+00:00')""",
    )
    conn.commit()
    entry = {"scan_id": "s1", "coin": "ABC", "direction": "LONG",
             "quantity": 1.0, "fill_price": 100.0}

    with patch.object(D, "market_close_position", return_value=None), \
         patch.object(D, "_send_urgent_alert") as urgent, \
         patch.object(D, "cancel_remaining_orders") as cancel:
        rc = D._execute_force_close(
            entry, "reversal", {}, "ABC/USDC:USDC", object(), _cfg(), conn,
        )

    assert rc == 0
    urgent.assert_called_once()
    cancel.assert_not_called()  # SL must NOT be cancelled on an unconfirmed close
    sl = conn.execute(
        "SELECT status FROM orders WHERE order_type='stop_loss'"
    ).fetchone()
    assert sl["status"] == "PLACED"  # protective SL retained → no naked position


def test_timeout_none_keeps_sl_and_alerts():
    conn = _conn()
    conn.execute(
        """INSERT INTO orders (order_id, scan_id, coin, direction, order_type,
        status, fill_price, quantity, exchange, execution_mode, created_at,
        filled_at)
        VALUES ('e1','s2','ABC','LONG','entry','FILLED',100.0,1.0,
                'hyperliquid','testnet','2020-01-01T00:00:00+00:00',
                '2020-01-01T00:00:00+00:00')""",
    )
    conn.execute(
        """INSERT INTO orders (order_id, scan_id, coin, direction, order_type,
        status, trigger_price, quantity, exchange, execution_mode, created_at)
        VALUES ('sl2','s2','ABC','LONG','stop_loss','PLACED',95.0,1.0,
                'hyperliquid','testnet','2020-01-01T00:00:00+00:00')""",
    )
    conn.commit()
    cfg = ExecutionConfig(exchange="hyperliquid", mode="testnet",
                          position_timeout_hours=1)

    with patch.object(M, "get_auth_exchange", return_value=object()), \
         patch.object(M, "get_exchange_meta",
                      return_value={"symbol_template": "{coin}/USDC:USDC"}), \
         patch.object(M, "market_close_position", return_value=None), \
         patch.object(M, "_send_urgent_sl_alert") as urgent, \
         patch.object(M, "cancel_remaining_orders") as cancel:
        M.enforce_position_timeouts(cfg, conn)

    urgent.assert_called_once()
    cancel.assert_not_called()
    sl = conn.execute(
        "SELECT status FROM orders WHERE order_type='stop_loss'"
    ).fetchone()
    assert sl["status"] == "PLACED"
