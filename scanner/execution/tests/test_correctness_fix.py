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

import importlib.util as _ilu
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from scanner.execution import drift as D
from scanner.execution import exit_fills as EF
from scanner.execution import monitor as M
from scanner.execution.analytics import compute_live_metrics
from scanner.execution.config import ExecutionConfig
from scanner.execution.storage import (
    ensure_orders_table,
    get_closed_trades,
    make_cloid,
    original_stop_price,
)

# ---------------------------------------------------------------------------
# Fix 3 — market_close_position: confirmed-fill gating
# ---------------------------------------------------------------------------


class FakeExchange:
    """Minimal ccxt stand-in. `create_resp`/`fetch_resp` are the order dicts
    returned by create_order / fetch_order; `fetch_raises` forces fetch_order
    to throw (degraded-path test). `seen_cloid` records the clientOrderId of
    the last create_order — cloid-for-cloid identity assertion."""

    def __init__(self, create_resp, fetch_resp=None, fetch_raises=False):
        self._create_resp = create_resp
        self._fetch_resp = fetch_resp
        self._fetch_raises = fetch_raises
        self.seen_cloid = None

    def amount_to_precision(self, symbol, qty):
        return qty

    def fetch_ticker(self, symbol):
        return {"last": 100.0}

    def create_order(self, symbol, otype, side, qty, price, params=None):
        self.seen_cloid = (params or {}).get("clientOrderId")
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
    # get_closed_trades → _calculate_slippage_r reads scans.machine_block;
    # an empty table is fine (no row → slippage_r 0).
    c.execute(
        "CREATE TABLE IF NOT EXISTS scans (scan_id TEXT PRIMARY KEY, "
        "machine_block TEXT, cost_usd REAL, created_at TEXT, prompt_version TEXT)"
    )
    return c


def _rows(conn):
    return conn.execute(
        "SELECT status, fill_price, quantity, order_type FROM orders"
    ).fetchall()


# C4#6 Chunk 2b re-contracted `market_close_position` → typed
# `CloseResult`; persistence moved to the caller (atomic full-close
# boundary). The Fix-3 confirmed-fill GATING is preserved verbatim — these
# retargeted tests assert (1) the kind/payload mapping per case, (2) that
# market_close_position itself NEVER writes a row, and (3) the caller persists
# a row + cloid IDENTICAL to the Fix-3 baseline (LESSONS "Green != verified":
# the contract moved boundary, not behaviour).

def _close(create_resp, fetch_resp=None, fetch_raises=False, qty=1.0):
    """Direct market_close_position call. Returns (result, rows, exchange).
    `rows` proves the invariant: market_close_position writes NOTHING."""
    conn = _conn()
    ex = FakeExchange(create_resp, fetch_resp, fetch_raises)
    result = M.market_close_position(
        _entry(), "ABC/USDC:USDC", "sell", qty, ex, _cfg(),
        order_type="drift_close",
    )
    return result, _rows(conn), ex


def test_closed_zero_fill_maps_none_no_self_persist():
    result, rows, _ = _close({"id": "x", "status": "closed", "filled": 0, "average": 99.0})
    assert result.kind == "none"
    assert result.payload is None and result.fill_price is None
    assert rows == []  # market_close_position never self-persists


def test_closed_partial_maps_partial_truthful_leg():
    result, rows, _ = _close(
        {"id": "x", "status": "closed", "filled": 0.5, "average": 99.0}, qty=1.0
    )
    assert result.kind == "partial"
    assert rows == []  # caller persists, not market_close_position
    assert result.payload["status"] == "FILLED"
    assert result.payload["quantity"] == 0.5  # actual leg, NOT rounded_qty
    assert result.payload["fill_price"] == 99.0
    assert result.fill_price == 99.0


def test_closed_full_average_maps_full_with_price():
    result, rows, _ = _close(
        {"id": "x", "status": "closed", "filled": 1.0, "average": 99.0}, qty=1.0
    )
    assert result.kind == "full"
    assert result.fill_price == 99.0
    assert result.payload["quantity"] == 1.0
    assert result.payload["fill_price"] == 99.0
    assert rows == []


def test_closed_full_price_fallback_when_no_average():
    result, _, _ = _close(
        {"id": "x", "status": "closed", "filled": 1.0, "average": None, "price": 98.5},
        qty=1.0,
    )
    assert result.kind == "full"
    assert result.fill_price == 98.5
    assert result.payload["fill_price"] == 98.5


def test_closed_filled_absent_is_full():
    # filled key absent (≠ explicit 0) + closed + priced → full close.
    result, _, _ = _close({"id": "x", "status": "closed", "average": 97.0}, qty=2.0)
    assert result.kind == "full"
    assert result.fill_price == 97.0
    assert result.payload["quantity"] == 2.0  # rounded_qty (full), not None


def test_nonclosed_with_echoed_price_maps_none():
    # status canceled, filled absent, price PRESENT (echoed IOC limit) — must
    # NOT be treated as a fill just because `price` exists.
    result, rows, _ = _close({"id": "x", "status": "canceled", "price": 100.0})
    assert result.kind == "none"
    assert rows == []


def test_fetch_order_fallback_resolves_close():
    # create resp inconclusive (open, no price) → fetch_order yields the close.
    result, _, _ = _close(
        {"id": "x", "status": "open"},
        fetch_resp={"id": "x", "status": "closed", "filled": 1.0, "average": 96.0},
        qty=1.0,
    )
    assert result.kind == "full"
    assert result.fill_price == 96.0
    assert result.payload["fill_price"] == 96.0


def test_fetch_order_raises_maps_none():
    result, rows, _ = _close({"id": "x", "status": "open"}, fetch_raises=True)
    assert result.kind == "none"
    assert rows == []


def test_no_row_path_logs_durable_ids(caplog):
    import logging
    with caplog.at_level(logging.ERROR, logger="scanner.execution.monitor"):
        _close({"id": "EXCH99", "status": "canceled", "filled": 0})
    msg = "\n".join(r.message for r in caplog.records)
    assert "NOT confirmed" in msg
    assert "exchange_id=EXCH99" in msg
    assert "order_id=" in msg and "cloid=" in msg


# --- caller-level: row-for-row + cloid-for-cloid identical to Fix-3 baseline -

def test_full_caller_persists_row_and_cloid_identical_to_baseline():
    """Drive the real drift caller end-to-end. The persisted close row and the
    clientOrderId the exchange saw must equal the Fix-3 baseline (the row
    market_close_position used to write itself) — the boundary moved, the
    bytes did not."""
    conn = _conn()
    conn.execute(
        """INSERT INTO orders (order_id, scan_id, coin, direction, order_type,
        status, exchange_order_id, quantity, exchange, execution_mode, created_at)
        VALUES ('sl1','s1','ABC','LONG','stop_loss','PLACED','SLX',1.0,
                'hyperliquid','testnet','2026-01-01T00:00:00+00:00')""",
    )
    conn.commit()
    entry = {"scan_id": "s1", "coin": "ABC", "direction": "LONG",
             "quantity": 1.0, "fill_price": 100.0}
    ex = FakeExchange({"id": "EXCH7", "status": "closed", "filled": 1.0, "average": 99.0})

    with patch.object(D, "_send_drift_close_alert"), \
         patch.object(EF, "cancel_order", return_value=True):
        rc = D._execute_force_close(
            entry, "reversal", {}, "ABC/USDC:USDC", ex, _cfg(), conn,
        )

    assert rc == 1
    close = conn.execute(
        "SELECT * FROM orders WHERE order_type='drift_close'"
    ).fetchone()
    # Fix-3 baseline row: FILLED, full rounded_qty, confirmed price, exch id.
    assert close["status"] == "FILLED"
    assert close["fill_price"] == 99.0
    assert close["quantity"] == 1.0
    assert close["exchange_order_id"] == "EXCH7"
    # cloid-for-cloid: clientOrderId == make_cloid(close order_id), as before.
    assert ex.seen_cloid == make_cloid(close["order_id"])
    # counterpart SL marked CANCELLED inside the SAME txn as the close row.
    sl = conn.execute(
        "SELECT status FROM orders WHERE order_type='stop_loss'"
    ).fetchone()
    assert sl["status"] == "CANCELLED"


def test_partial_caller_persists_only_partial_row_sl_retained():
    """kind=='partial' → caller persists ONLY the truthful partial row and
    RETAINS the SL/counterparts (Fix-3 partial safety not regressed)."""
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
    ex = FakeExchange({"id": "x", "status": "closed", "filled": 0.4, "average": 99.0})

    with patch.object(D, "_send_urgent_alert") as urgent:
        rc = D._execute_force_close(
            entry, "reversal", {}, "ABC/USDC:USDC", ex, _cfg(), conn,
        )

    assert rc == 0
    urgent.assert_called_once()
    closes = conn.execute(
        "SELECT quantity, fill_price FROM orders WHERE order_type='drift_close'"
    ).fetchall()
    assert len(closes) == 1
    assert closes[0]["quantity"] == 0.4  # truthful partial leg only
    sl = conn.execute(
        "SELECT status FROM orders WHERE order_type='stop_loss'"
    ).fetchone()
    assert sl["status"] == "PLACED"  # SL retained — remainder retried next cycle


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

    result, new_rows, _ = _close(resp)
    assert result.kind == "none"              # NEW: explicit non-close
    assert result.payload is None             # NEW: no row to persist
    assert new_rows == []                     # NEW: nothing fabricated
    assert result.fill_price != old_ret       # anti-vacuity: provably diverges


# --- caller-level: kind=='none' is safe in both production callers ----------


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

    with patch.object(D, "market_close_position",
                      return_value=M.CloseResult("none", None, None)), \
         patch.object(D, "_send_urgent_alert") as urgent, \
         patch.object(D, "_cancel_counterparts_on_exchange") as exch_cancel, \
         patch.object(D, "_mark_counterparts_cancelled") as db_cancel:
        rc = D._execute_force_close(
            entry, "reversal", {}, "ABC/USDC:USDC", object(), _cfg(), conn,
        )

    assert rc == 0
    urgent.assert_called_once()
    exch_cancel.assert_not_called()  # SL must NOT be cancelled on a non-close
    db_cancel.assert_not_called()
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
         patch.object(M, "market_close_position",
                      return_value=M.CloseResult("none", None, None)), \
         patch.object(M, "_send_urgent_close_alert") as urgent, \
         patch.object(M, "_cancel_counterparts_on_exchange") as exch_cancel:
        M.enforce_position_timeouts(cfg, conn)

    urgent.assert_called_once()
    exch_cancel.assert_not_called()
    sl = conn.execute(
        "SELECT status FROM orders WHERE order_type='stop_loss'"
    ).fetchone()
    assert sl["status"] == "PLACED"


# ---------------------------------------------------------------------------
# Fix 1 — drift-close alert R anchored on the storage-owned original SL
# ---------------------------------------------------------------------------

def _adversarial_drift_conn():
    """scan with SL_drift (95.0, LATER created_at, inserted FIRST → lower
    rowid) and SL_orig (100.0, EARLIER created_at, inserted SECOND). A bare
    `LIMIT 1` (rowid order) picks the non-original 95.0; original_stop_price
    (ORDER BY created_at ASC) picks 100.0."""
    conn = _conn()
    conn.execute(
        """INSERT INTO orders (order_id, scan_id, coin, direction, order_type,
        status, fill_price, entry_price, quantity, exchange, execution_mode,
        created_at, filled_at)
        VALUES ('e','s1','ABC','LONG','entry','FILLED',105.0,105.0,1.0,
                'hyperliquid','testnet','2026-01-01T00:00:00+00:00',
                '2026-01-01T00:00:00+00:00')""",
    )
    conn.execute(
        """INSERT INTO orders (order_id, scan_id, coin, direction, order_type,
        status, trigger_price, quantity, exchange, execution_mode, created_at)
        VALUES ('sl_drift','s1','ABC','LONG','stop_loss','CANCELLED',95.0,1.0,
                'hyperliquid','testnet','2026-01-01T06:00:00+00:00')""",
    )
    conn.execute(
        """INSERT INTO orders (order_id, scan_id, coin, direction, order_type,
        status, trigger_price, quantity, exchange, execution_mode, created_at)
        VALUES ('sl_orig','s1','ABC','LONG','stop_loss','CANCELLED',100.0,1.0,
                'hyperliquid','testnet','2026-01-01T00:00:00+00:00')""",
    )
    conn.commit()
    return conn


def test_original_stop_price_picks_original_not_rowid_first():
    conn = _adversarial_drift_conn()
    frozen_old = conn.execute(
        "SELECT trigger_price FROM orders WHERE scan_id = ? "
        "AND order_type = 'stop_loss' LIMIT 1",
        ("s1",),
    ).fetchone()
    assert float(frozen_old["trigger_price"]) == 95.0  # bug: rowid-first
    assert original_stop_price("s1", conn, "testnet", "hyperliquid") == 100.0


def test_drift_close_alert_r_anchored_on_original_sl():
    """By-construction divergence: LONG entry 105, drift_close fill 102, qty 1.
      frozen-old (bare LIMIT 1 → SL 95):  R = (102-105)/|105-95|  = -0.3
      new (original_stop_price → SL 100): R = (102-105)/|105-100| = -0.6
    """
    conn = _adversarial_drift_conn()
    entry = {"scan_id": "s1", "coin": "ABC", "direction": "LONG",
             "quantity": 1.0, "fill_price": 105.0}
    captured = {}

    def _capture_alert(e, trig, fp, r_val, cfg):
        captured["r"] = r_val

    with patch.object(D, "market_close_position",
                      return_value=M.CloseResult(
                          "full",
                          {"order_id": "c1", "scan_id": "s1", "coin": "ABC",
                           "direction": "LONG", "order_type": "drift_close",
                           "status": "FILLED", "fill_price": 102.0, "quantity": 1.0,
                           "exchange": "hyperliquid", "execution_mode": "testnet"},
                          102.0)), \
         patch.object(D, "_cancel_counterparts_on_exchange", return_value=[]), \
         patch.object(D, "_send_drift_close_alert", side_effect=_capture_alert):
        rc = D._execute_force_close(
            entry, "reversal", {}, "ABC/USDC:USDC", object(), _cfg(), conn,
        )

    assert rc == 1
    assert captured["r"] == pytest.approx(-0.6)   # fix: original SL anchor
    frozen_old_r = (102.0 - 105.0) / abs(105.0 - 95.0)
    assert frozen_old_r == pytest.approx(-0.3)
    assert captured["r"] != pytest.approx(frozen_old_r)  # anti-vacuity


# ---------------------------------------------------------------------------
# Fix 2 — no-usable-risk trades excluded from R-normalized aggregates
# ---------------------------------------------------------------------------

def _ins(conn, scan_id, otype, status, *, fill=None, trig=None, qty=1.0,
         created="2026-01-01T00:00:00+00:00", filled="2026-01-01T01:00:00+00:00"):
    conn.execute(
        """INSERT INTO orders (order_id, scan_id, coin, direction, order_type,
        status, fill_price, trigger_price, entry_price, quantity, exchange,
        execution_mode, created_at, filled_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (f"{scan_id}_{otype}_{created}", scan_id, "ABC", "LONG", otype, status,
         fill, trig, fill, qty, "hyperliquid", "testnet", created, filled),
    )


def _fix2_corpus():
    """sA: no stop_loss row, +pnl. sB: stop_loss row w/ NULL trigger, -pnl.
    sC: real SL (risked), +pnl. (a)/(b) both → original_stop_price is None."""
    conn = _conn()
    # sA — no SL: entry 100 → tp1 110, LONG → +10
    _ins(conn, "sA", "entry", "FILLED", fill=100.0, qty=1.0)
    _ins(conn, "sA", "tp1", "FILLED", fill=110.0, qty=1.0)
    # sB — SL row present but trigger_price NULL: entry 100 → drift_close 95 → -5
    _ins(conn, "sB", "entry", "FILLED", fill=100.0, qty=1.0)
    _ins(conn, "sB", "stop_loss", "CANCELLED", trig=None, qty=1.0)
    _ins(conn, "sB", "drift_close", "FILLED", fill=95.0, qty=1.0)
    # sC — real original SL 90: entry 100 → tp1 120 → +20, risk |100-90|=10 → R 2
    _ins(conn, "sC", "entry", "FILLED", fill=100.0, qty=1.0)
    _ins(conn, "sC", "stop_loss", "CANCELLED", trig=90.0, qty=1.0)
    _ins(conn, "sC", "tp1", "FILLED", fill=120.0, qty=1.0)
    conn.commit()
    return conn


def test_fix2_no_sl_trades_get_zero_risk():
    trades = {t.scan_id: t for t in get_closed_trades(_fix2_corpus())}
    assert trades["sA"].risk_amount == 0 and trades["sA"].realized_r == 0
    assert trades["sB"].risk_amount == 0 and trades["sB"].realized_r == 0
    # net of taker fees (SPEC #42): gross R 2.0, fee 0.00035*(100+120)/10 = 0.0077 → 1.9923
    assert trades["sC"].risk_amount == 10.0
    assert trades["sC"].realized_r == pytest.approx(1.9923, abs=1e-4)
    assert trades["sC"].fees_r == pytest.approx(0.0077, abs=1e-6)


def _frozen_old_metrics(trades):
    """Pre-fix core: W/L + expectancy + PF over ALL trades (no risked filter)."""
    wins = [t for t in trades if t.realized_r > 0]
    losses = [t for t in trades if t.realized_r <= 0]
    wr = len(wins) / len(trades)
    exp = sum(t.realized_r for t in trades) / len(trades)
    gp = sum(t.realized_pnl for t in wins)
    gl = abs(sum(t.realized_pnl for t in losses))
    pf = gp / gl if gl > 0 else float("inf") if gp > 0 else 0
    return wr, exp, pf


def test_fix2_metrics_exclude_no_risk_and_diverge_from_frozen_old():
    trades = list(get_closed_trades(_fix2_corpus()))
    m = compute_live_metrics(trades)

    # NEW: only sC (risked) drives R-metrics; sA/sB stay in count/pnl.
    assert m.count == 3
    # net of taker fees (SPEC #42): gross (10-5+20)=25 minus per-leg taker fees
    assert m.total_pnl == pytest.approx(24.78, abs=1e-2)
    assert m.win_rate == pytest.approx(1.0)            # 1/1 risked, a win
    assert m.expectancy_r == pytest.approx(1.9923, abs=1e-4)  # sC only, net of fees
    assert m.profit_factor == float("inf")             # no risked losers
    assert any("2 trades with no usable risk anchor" in w
               for w in m.quality_warnings)

    # Frozen-old would fold sA/sB in as break-even losses → divergence.
    old_wr, old_exp, old_pf = _frozen_old_metrics(trades)
    assert old_wr == pytest.approx(1 / 3)              # bug: diluted
    assert old_exp == pytest.approx(1.9923 / 3, abs=1e-4)  # bug: diluted (net of fees)
    assert old_pf == pytest.approx(4.101, abs=1e-3)    # bug: +pnl no-SL in gross_loss
    assert m.win_rate != pytest.approx(old_wr)         # anti-vacuity
    assert m.expectancy_r != pytest.approx(old_exp)
    assert m.profit_factor != old_pf


def _load_script():
    p = Path(__file__).resolve().parents[3] / "scripts" / "execution-log-data.py"
    spec = _ilu.spec_from_file_location("_elog", p)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fix2_script_summary_excludes_no_risk_and_guards_empty():
    elog = _load_script()
    trades = list(get_closed_trades(_fix2_corpus()))
    m = compute_live_metrics(trades)
    out = elog.format_summary([], [], trades, [], m, "2026-01-01")
    assert "1W / 0L — 100% over 1 risked" in out
    assert "| Closed trades | 3 (" in out          # total closed shown

    # all-no-risk window → no ZeroDivisionError, explicit N/A
    no_risk_only = [t for t in trades if t.risk_amount == 0]
    out2 = elog.format_summary([], [], no_risk_only, [],
                               compute_live_metrics(no_risk_only), "2026-01-01")
    assert "0W / 0L — N/A; no usable risk anchor" in out2
