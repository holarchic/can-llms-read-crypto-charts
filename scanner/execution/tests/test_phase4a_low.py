"""T-016 Phase 4a C-LOW — the two C4#6-deferred cosmetic LOWs (SPEC #27).

In `sync_order_status`, the exit-fill `_send_fill_alert` + `updated += 1`
moved to AFTER `_handle_exit_fill` returns successfully. On the rare
close-cluster rollback the close is unrecorded and the next cycle re-detects
the fill — emitting them before the commit double-alerted (LOW a) and
over-counted the non-persisted update (LOW b). Happy path is byte-identical
(exactly one of each, after the commit).
"""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from scanner.execution import monitor as M
from scanner.execution.config import ExecutionConfig
from scanner.execution.storage import ensure_orders_table, save_order


def _cfg():
    return ExecutionConfig(exchange="hyperliquid", mode="testnet")


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_orders_table(c)
    return c


def _seed(conn):
    """FILLED entry (skipped — not PLACED) + one PLACED SL with an exch oid
    that has 'disappeared' from the open set → routed as an exit fill."""
    save_order({
        "order_id": "e1", "scan_id": "s1", "coin": "ABC", "direction": "LONG",
        "order_type": "entry", "status": "FILLED", "fill_price": 100.0,
        "quantity": 1.0, "exchange": "hyperliquid", "execution_mode": "testnet",
        "created_at": "2020-01-01T00:00:00+00:00",
        "filled_at": "2020-01-01T00:00:00+00:00",
    }, conn)
    save_order({
        "order_id": "sl1", "scan_id": "s1", "coin": "ABC", "direction": "LONG",
        "order_type": "stop_loss", "status": "PLACED", "trigger_price": 95.0,
        "exchange_order_id": "SL_OID", "quantity": 1.0,
        "exchange": "hyperliquid", "execution_mode": "testnet",
        "created_at": "2020-01-01T00:00:00+00:00",
    }, conn)
    conn.commit()


class _Exch:
    def amount_to_precision(self, symbol, qty):
        return qty

    def fetch_order(self, oid, symbol):
        return {"id": oid, "status": "closed", "filled": 1.0, "average": 99.0}


def _run(conn):
    return M.sync_order_status(_cfg(), conn, exchange_open_orders=[])


def test_no_duplicate_fill_alert_across_rollback_then_retry(conn):
    """LOW (a): cluster rolls back on cycle 1 (no alert), succeeds cycle 2 →
    exactly ONE fill-alert total. Pre-fix the alert fired before the handler,
    so a rollback+retry double-alerted."""
    _seed(conn)
    with patch.object(M, "get_auth_exchange", return_value=_Exch()), \
         patch.object(M, "get_exchange_meta",
                      return_value={"symbol_template": "{coin}/USDC:USDC"}), \
         patch.object(M, "_send_fill_alert") as sfa, \
         patch.object(M, "_handle_exit_fill",
                      side_effect=[RuntimeError("db boom"), None]):
        _run(conn)   # cycle 1: _handle_exit_fill raises → rollback, NO alert
        _run(conn)   # cycle 2: succeeds → exactly one alert
    assert sfa.call_count == 1   # not 2 (the pre-fix duplicate)


def test_updated_count_excludes_rolled_back_exit(conn):
    """LOW (b): a rolled-back exit close is NOT counted in the returned
    `updated`; a successful one is counted exactly once."""
    _seed(conn)
    common = [
        patch.object(M, "get_auth_exchange", return_value=_Exch()),
        patch.object(M, "get_exchange_meta",
                     return_value={"symbol_template": "{coin}/USDC:USDC"}),
        patch.object(M, "_send_fill_alert", MagicMock()),
    ]
    with common[0], common[1], common[2], \
         patch.object(M, "_handle_exit_fill", side_effect=RuntimeError("db boom")):
        assert _run(conn) == 0   # rolled back → not counted (pre-fix: 1, over-count)

    with patch.object(M, "get_auth_exchange", return_value=_Exch()), \
         patch.object(M, "get_exchange_meta",
                      return_value={"symbol_template": "{coin}/USDC:USDC"}), \
         patch.object(M, "_send_fill_alert", MagicMock()), \
         patch.object(M, "_handle_exit_fill", return_value=None):
        assert _run(conn) == 1   # successful close → counted exactly once
