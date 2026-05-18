"""T-016 C4#6 Chunk 3 — the two named, injected-failure-tested orphan windows.

(i) claim→execute→resolve: a WON claim_pending ALWAYS resolves the pending
    in-process (TRIGGERED on success; TRIGGER_FAILED on skip / Exception /
    BaseException) — never a stranded ACTIVE or orphan TRIGGERED. The
    resolution write is guarded so it cannot mask the original error.
(ii) entry-FILLED→SL: attempt_sl_recovery converges to >=1 on-exchange SL
     (never naked). The lost-random-order_id duplicate is the NAMED
     Phase-3b reconcile residual — asserted as deferred, not a failure.
"""

import sqlite3
from unittest.mock import patch

import pytest

from scanner.execution import entry_monitor as EM
from scanner.execution import monitor as M
from scanner.execution.config import ExecutionConfig
from scanner.execution.executor import ExecutionResult
from scanner.execution.storage import (
    ensure_orders_table,
    save_order,
    save_pending_entry,
)

_MB = (
    '{"entry_low":99.0,"entry_high":101.0,"stop_loss":95.0,"tp1":110.0,'
    '"tp1_exit_pct":50,"direction":"LONG","confidence":7,"leverage":5,'
    '"entry_trigger":"limit_at","entry_trigger_price":100.0,'
    '"invalidation_price":90.0,"expiry_candles":12}'
)


def _cfg():
    return ExecutionConfig(exchange="hyperliquid", mode="testnet")


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_orders_table(c)
    return c


def _seed_pending(conn, status="ACTIVE"):
    save_pending_entry({
        "pending_id": "p1", "scan_id": "s1", "coin": "ABC", "direction": "LONG",
        "machine_block": _MB, "entry_trigger": "limit_at",
        "entry_trigger_price": 100.0, "invalidation_price": 90.0,
        "expiry_candles": 12, "exchange": "hyperliquid",
        "execution_mode": "testnet", "created_at": "2026-01-01T00:00:00+00:00",
    }, conn)
    if status != "ACTIVE":
        conn.execute("UPDATE pending_entries SET status=? WHERE pending_id='p1'",
                     (status,))
    conn.commit()


def _pending_status(conn):
    return conn.execute(
        "SELECT status FROM pending_entries WHERE pending_id='p1'"
    ).fetchone()["status"]


def _entry():
    return {"pending_id": "p1", "scan_id": "s1", "coin": "ABC",
            "direction": "LONG", "machine_block": _MB,
            "entry_trigger": "limit_at", "entry_trigger_price": 100.0,
            "invalidation_price": 90.0, "expiry_candles": 12}


# --- (i) claim→execute→resolve invariant -----------------------------------


def _run_triggered(conn, *, execute_side_effect=None, er=None):
    with patch.object(EM, "_send_trigger_alert"), \
         patch.object(EM, "ScannerConfig", lambda: object()), \
         patch("scanner.execution.executor.execute_trade",
               side_effect=execute_side_effect,
               return_value=er) as mock_exec:
        rc = EM._execute_triggered_entry(_entry(), 100.0, _cfg(), conn)
    return rc, mock_exec


def test_success_resolves_to_triggered(conn):
    _seed_pending(conn)
    rc, _ = _run_triggered(conn, er=ExecutionResult(executed=True,
                                                    entry_order_id="e1",
                                                    sl_order_id="sl1"))
    assert rc == 1
    assert _pending_status(conn) == "TRIGGERED"  # resolved-success terminal


def test_skip_resolves_to_trigger_failed(conn):
    _seed_pending(conn)
    rc, _ = _run_triggered(conn, er=ExecutionResult(executed=False,
                                                    skip_reason="duplicate"))
    assert rc == 1
    assert _pending_status(conn) == "TRIGGER_FAILED"


def test_exception_resolves_and_propagates(conn):
    _seed_pending(conn)
    with pytest.raises(RuntimeError, match="exec boom"):
        _run_triggered(conn, execute_side_effect=RuntimeError("exec boom"))
    # Resolved before the exception left the function — no stranded TRIGGERED.
    assert _pending_status(conn) == "TRIGGER_FAILED"


def test_baseexception_resolves_and_propagates(conn):
    _seed_pending(conn)
    with pytest.raises(KeyboardInterrupt):
        _run_triggered(conn, execute_side_effect=KeyboardInterrupt())
    assert _pending_status(conn) == "TRIGGER_FAILED"  # incl. BaseException


def test_lost_claim_bails_without_executing(conn):
    _seed_pending(conn, status="TRIGGERED")  # already claimed by a racer
    rc, mock_exec = _run_triggered(conn, er=ExecutionResult(executed=True))
    assert rc == 0
    mock_exec.assert_not_called()
    assert _pending_status(conn) == "TRIGGERED"  # untouched


def test_guarded_resolution_does_not_mask_original_error(conn, caplog):
    """If the TRIGGER_FAILED write ITSELF fails, the original execute_trade
    error must still propagate (not the resolution error) and the strand is
    logged as the KL#0/Phase-3b boundary — not silently swallowed."""
    import logging
    _seed_pending(conn)
    with caplog.at_level(logging.ERROR, logger="scanner.execution.entry_monitor"):
        with patch.object(EM, "update_pending_status",
                          side_effect=sqlite3.OperationalError("db down")):
            with pytest.raises(RuntimeError, match="original boom"):
                _run_triggered(conn,
                               execute_side_effect=RuntimeError("original boom"))
    assert "resolution retry failed" in caplog.text


def test_skip_path_resolution_failure_propagates_not_silent_success(conn):
    """Skip (executed=False) + the TRIGGER_FAILED write fails: must NOT
    return 1 (silent stranded TRIGGERED). The unguarded skip write raises →
    `except BaseException` retries once → re-raises, so an unresolved skip
    surfaces to the caller loop instead of a false success."""
    _seed_pending(conn)
    with patch.object(EM, "update_pending_status",
                      side_effect=sqlite3.OperationalError("db down")) as upd:
        with pytest.raises(sqlite3.OperationalError, match="db down"):
            _run_triggered(conn, er=ExecutionResult(executed=False,
                                                    skip_reason="duplicate"))
    # Resolution attempted twice: unguarded skip write + guarded except retry.
    assert upd.call_count == 2


# --- (ii) entry-FILLED → >=1 on-exchange SL invariant ----------------------


class _SLExch:
    """create_order raises `fail_times` then succeeds. Records every cloid."""

    def __init__(self, fail_times=0):
        self._fail_times = fail_times
        self._n = 0
        self.cloids = []

    def amount_to_precision(self, symbol, qty):
        return qty

    def create_order(self, symbol, otype, side, qty, price, params=None):
        self.cloids.append((params or {}).get("clientOrderId"))
        self._n += 1
        if self._n <= self._fail_times:
            raise RuntimeError("SL placement boom")
        return {"id": f"X{self._n}"}


def _seed_filled_entry_no_sl(conn):
    save_order({
        "order_id": "e1", "scan_id": "s1", "coin": "ABC", "direction": "LONG",
        "order_type": "entry", "status": "FILLED", "fill_price": 100.0,
        "quantity": 1.0, "exchange": "hyperliquid", "execution_mode": "testnet",
        "created_at": "2026-01-01T00:00:00+00:00",
        "filled_at": "2026-01-01T00:00:00+00:00",
    }, conn)
    # A FAILED SL row carries the trigger_price (the SL write that didn't land).
    save_order({
        "order_id": "sl_failed", "scan_id": "s1", "coin": "ABC",
        "direction": "LONG", "order_type": "stop_loss", "status": "FAILED",
        "trigger_price": 95.0, "quantity": 1.0, "exchange": "hyperliquid",
        "execution_mode": "testnet", "created_at": "2026-01-01T00:00:01+00:00",
    }, conn)
    conn.commit()


def _placed_sls(conn):
    return conn.execute(
        "SELECT order_id FROM orders WHERE order_type='stop_loss' "
        "AND status='PLACED'"
    ).fetchall()


def _recover(conn, exch):
    issue = {"scan_id": "s1", "coin": "ABC", "direction": "LONG",
             "type": "missing_sl"}
    with patch.object(M, "get_auth_exchange", return_value=exch), \
         patch.object(M, "get_exchange_meta",
                      return_value={"symbol_template": "{coin}/USDC:USDC"}), \
         patch.object(M, "get_open_positions",
                      return_value=[{"coin": "ABC", "direction": "LONG",
                                     "size": 1.0}]):
        return M.attempt_sl_recovery(issue, _cfg(), conn)


def test_sl_recovery_converges_to_at_least_one_sl(conn):
    """Cycle 1 placement raises (3 retries) → unprotected, zero PLACED SL.
    Cycle 2 placement succeeds → converges to >=1 PLACED SL (not naked)."""
    _seed_filled_entry_no_sl(conn)

    assert _recover(conn, _SLExch(fail_times=3)) is False
    assert len(_placed_sls(conn)) == 0  # cycle 1: still unprotected

    assert _recover(conn, _SLExch(fail_times=0)) is True
    assert len(_placed_sls(conn)) >= 1  # cycle 2: >=1 SL — never naked


def test_sl_recovery_duplicate_residual_is_named_not_idempotent(conn):
    """Two successful recoveries mint DIFFERENT cloids (no stable scan_id+role
    key). DB converges to exactly one PLACED (supersede), but the lost-
    random-order_id exchange-level duplicate is the accepted KL#0/Phase-3b
    reconcile residual — asserted here as deferred, NOT a failure."""
    _seed_filled_entry_no_sl(conn)

    e1 = _SLExch(fail_times=0)
    assert _recover(conn, e1) is True
    placed_after_1 = {r["order_id"] for r in _placed_sls(conn)}

    e2 = _SLExch(fail_times=0)
    assert _recover(conn, e2) is True
    placed_after_2 = {r["order_id"] for r in _placed_sls(conn)}

    # >=1 invariant holds every cycle (never zero — no naked position).
    assert len(placed_after_1) == 1 and len(placed_after_2) == 1
    # Non-idempotent minting: the second cycle's cloid differs from the first
    # (HL would NOT dedupe → the named exchange-level duplicate residual).
    assert e1.cloids[-1] != e2.cloids[-1]
    # DB self-heals to a single owner (old PLACED superseded → CANCELLED).
    assert placed_after_1 != placed_after_2
