"""T-016 Phase-3b-(b) — duplicate-SL convergence (SPEC #30, C4#6 window b).

C1: the pure keeper-identity + SL/TP-discriminator helpers, incl. the
anti-vacuity test that pins identity is NOT current-row-derived (the codex
plan-review R1 HIGH). C3: drives the REAL `reconcile_positions` over an
in-memory DB with mocked exchange truth — converges/idempotent, never-naked
(keeper-absent ⇒ zero cancels), fail-closed ×2, SL≠TP (LONG+SHORT+"N/A"),
the replacement-keeper end-to-end case, net-dedup, and the method pin.
Green ≠ verified — every C3 assertion drives a real cancel/non-cancel through
the production `reconcile_positions`.
"""

import sqlite3
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import pytest

from scanner.execution.config import ExecutionConfig
from scanner.execution.reconcile import (
    _SL_CLOID_SUFFIXES,
    _is_reduce_only_sl,
    _keeper_ids,
    reconcile_positions,
)
from scanner.execution.storage import ensure_orders_table, make_cloid, save_order


def _cfg():
    return ExecutionConfig(exchange="hyperliquid", mode="testnet")


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_orders_table(c)
    return c


def _oo(*, coin="ABC", side="sell", reduce_only=True, is_trigger=True,
        condition="Price below 95.0", oid="x1", cloid="0xc1"):
    """A representative ccxt-4.5.50 HL `frontendOpenOrders` parsed order dict
    (unified `reduceOnly`/`side`/`symbol` + raw `info` carrying isTrigger /
    triggerCondition — NOT stopLossPrice/isStopLoss; those are creation
    params, absent on fetched orders)."""
    return {
        "id": oid,
        "clientOrderId": cloid,
        "symbol": f"{coin}/USDC:USDC",
        "side": side,
        "reduceOnly": reduce_only,
        "triggerPrice": 95.0,
        "info": {
            "isPositionTpsl": False,
            "isTrigger": is_trigger,
            "reduceOnly": reduce_only,
            "triggerCondition": condition,
            "triggerPx": "95.0",
        },
    }


def _seed_entry(conn, scan_id="s1", coin="ABC", direction="LONG", qty=1.0):
    save_order({
        "order_id": f"e_{scan_id}", "scan_id": scan_id, "coin": coin,
        "direction": direction, "order_type": "entry", "status": "FILLED",
        "fill_price": 100.0, "entry_price": 100.0, "quantity": qty,
        "exchange": "hyperliquid", "execution_mode": "testnet",
        "created_at": "2026-01-01T00:00:00+00:00",
        "filled_at": "2026-01-01T00:00:00+00:00",
    }, conn)


def _seed_sl(conn, scan_id, order_id, *, status="PLACED", exch_oid="",
             coin="ABC", direction="LONG", created="2026-01-01T00:00:00+00:00"):
    save_order({
        "order_id": order_id, "scan_id": scan_id, "coin": coin,
        "direction": direction, "order_type": "stop_loss", "status": status,
        "exchange_order_id": exch_oid, "trigger_price": 95.0, "quantity": 1.0,
        "exchange": "hyperliquid", "execution_mode": "testnet",
        "created_at": created, "placed_at": created,
    }, conn)


@contextmanager
def _patch_exchange(positions, open_orders, cancel_ok=True):
    """Window (b) needs all three exchange seams mocked at source
    (function-local imports in reconcile.py → patch the owning module)."""
    cancel = patch("scanner.execution.exchange.cancel_order",
                    return_value=cancel_ok)
    with ExitStack() as es:
        es.enter_context(patch(
            "scanner.execution.exchange.get_open_positions",
            return_value=positions))
        es.enter_context(patch(
            "scanner.execution.exchange.get_exchange_open_orders",
            return_value=open_orders))
        m = es.enter_context(cancel)
        yield m


# --- C1: _is_reduce_only_sl discriminator (real frontendOpenOrders shape) ---

class TestIsReduceOnlySL:
    def test_long_sl_below_sell(self):
        assert _is_reduce_only_sl(
            _oo(side="sell", condition="Price below 95.0"), "ABC", "LONG")

    def test_long_tp_above_sell_is_not_sl(self):
        assert not _is_reduce_only_sl(
            _oo(side="sell", condition="Price above 120.0"), "ABC", "LONG")

    def test_short_sl_above_buy(self):
        assert _is_reduce_only_sl(
            _oo(side="buy", condition="Price above 105.0"), "ABC", "SHORT")

    def test_short_tp_below_buy_is_not_sl(self):
        assert not _is_reduce_only_sl(
            _oo(side="buy", condition="Price below 80.0"), "ABC", "SHORT")

    def test_na_condition_is_not_sl_conservative(self):
        assert not _is_reduce_only_sl(
            _oo(side="sell", condition="N/A"), "ABC", "LONG")

    def test_not_reduce_only_is_not_sl(self):
        assert not _is_reduce_only_sl(
            _oo(reduce_only=False, side="sell", condition="Price below 95.0"),
            "ABC", "LONG")

    def test_not_trigger_is_not_sl(self):
        assert not _is_reduce_only_sl(
            _oo(is_trigger=False, side="sell", condition="Price below 95.0"),
            "ABC", "LONG")

    def test_wrong_coin_is_not_sl(self):
        assert not _is_reduce_only_sl(
            _oo(coin="XYZ", side="sell", condition="Price below 95.0"),
            "ABC", "LONG")

    def test_wrong_side_is_not_sl(self):
        # LONG exit is sell; a buy trigger is not this position's SL.
        assert not _is_reduce_only_sl(
            _oo(side="buy", condition="Price below 95.0"), "ABC", "LONG")


# --- C1: _keeper_ids — identity is NOT current-row-derived (R1 HIGH) ---

class TestKeeperIds:
    def test_base_cloid_from_own_row(self):
        rows = [{"order_id": "sl1", "exchange_order_id": ""}]
        assert make_cloid("sl1") in _keeper_ids(rows)

    def test_oid_when_present(self):
        rows = [{"order_id": "slX", "exchange_order_id": "EXCH123"}]
        ids = _keeper_ids(rows)
        assert "EXCH123" in ids                       # OID matched
        assert make_cloid("slX") in ids               # cloid still derived too

    def test_replacement_cloid_is_predecessor_derived_not_current_row(self):
        # codex plan-review R1 HIGH anti-vacuity: a drift-replaced SL's live
        # clientOrderId = make_cloid(PREDECESSOR.order_id + ":drift") while the
        # CURRENT row carries a fresh random order_id. Identity built from the
        # current row alone would MISS the live keeper and cancel it (naked).
        pred = {"order_id": "A", "exchange_order_id": ""}      # cancelled predecessor
        curr = {"order_id": "B", "exchange_order_id": ""}      # PLACED, B != A
        live_cloid = make_cloid("A" + ":drift")

        # current-row-only derivation matches NOTHING:
        assert all(make_cloid("B" + s) != live_cloid for s in _SL_CLOID_SUFFIXES)
        # whole-chain union DOES contain it (via predecessor A + ":drift"):
        assert live_cloid in _keeper_ids([pred, curr])

    def test_every_frozen_suffix_round_trips(self):
        rows = [{"order_id": "z", "exchange_order_id": ""}]
        ids = _keeper_ids(rows)
        for s in _SL_CLOID_SUFFIXES:
            assert make_cloid("z" + s) in ids


# --- C3: end-to-end through the real reconcile_positions ---

# A live exchange position so window (a) is a no-op (isolates window b).
_POS = [{"coin": "ABC", "direction": "LONG", "size": 1.0}]


class TestConverges:
    def test_extra_cancelled_keeper_untouched_idempotent(self, conn):
        _seed_entry(conn, "s1")
        _seed_sl(conn, "s1", "sl1")                       # tracked keeper
        keeper = _oo(oid="K", cloid=make_cloid("sl1"))    # live keeper SL
        orphan = _oo(oid="ORPH", cloid="0xdead")          # untracked dup SL
        with _patch_exchange(_POS, [keeper, orphan]) as cancel:
            n = reconcile_positions(_cfg(), conn)
        assert n == 1
        cancel.assert_called_once()
        assert cancel.call_args[0][0] == "ORPH"           # the orphan's id

        # 2nd run: snapshot no longer lists the orphan → no-op.
        with _patch_exchange(_POS, [keeper]) as cancel2:
            assert reconcile_positions(_cfg(), conn) == 0
        cancel2.assert_not_called()

    def test_keeper_matched_by_exchange_order_id(self, conn):
        _seed_entry(conn, "s1")
        _seed_sl(conn, "s1", "sl1", exch_oid="EXCH9")
        keeper = _oo(oid="EXCH9", cloid="0xwhatever")      # matched by OID
        orphan = _oo(oid="ORPH", cloid="0xdead")
        with _patch_exchange(_POS, [keeper, orphan]) as cancel:
            n = reconcile_positions(_cfg(), conn)
        assert n == 1
        assert cancel.call_args[0][0] == "ORPH"


class TestNeverNaked:
    def test_keeper_absent_zero_cancels(self, conn):
        # The single most important test: NO tracked keeper live in the
        # snapshot → window (b) cancels NOTHING (that lone SL is the
        # missing_sl/attempt_sl_recovery case; cancelling it = naked).
        _seed_entry(conn, "s1")
        _seed_sl(conn, "s1", "sl1")
        lone = _oo(oid="LONE", cloid="0xnotkeeper")        # untracked, no keeper live
        with _patch_exchange(_POS, [lone]) as cancel:
            n = reconcile_positions(_cfg(), conn)
        assert n == 0
        cancel.assert_not_called()


class TestFailClosed:
    def test_positions_read_raises_zero_cancel(self, conn):
        _seed_entry(conn, "s1")
        _seed_sl(conn, "s1", "sl1")
        with ExitStack() as es:
            es.enter_context(patch(
                "scanner.execution.exchange.get_open_positions",
                side_effect=RuntimeError("net down")))
            es.enter_context(patch(
                "scanner.execution.exchange.get_exchange_open_orders",
                return_value=[_oo(oid="ORPH", cloid="0xdead")]))
            cancel = es.enter_context(patch(
                "scanner.execution.exchange.cancel_order", return_value=True))
            assert reconcile_positions(_cfg(), conn) == 0
        cancel.assert_not_called()

    def test_open_orders_read_raises_zero_cancel(self, conn):
        _seed_entry(conn, "s1")
        _seed_sl(conn, "s1", "sl1")
        with ExitStack() as es:
            es.enter_context(patch(
                "scanner.execution.exchange.get_open_positions",
                return_value=_POS))
            es.enter_context(patch(
                "scanner.execution.exchange.get_exchange_open_orders",
                side_effect=RuntimeError("orders fetch down")))
            cancel = es.enter_context(patch(
                "scanner.execution.exchange.cancel_order", return_value=True))
            assert reconcile_positions(_cfg(), conn) == 0
        cancel.assert_not_called()


class TestSLvsTP:
    def test_long_tp_alongside_orphan_sl_only_sl_cancelled(self, conn):
        _seed_entry(conn, "s1")
        _seed_sl(conn, "s1", "sl1")
        keeper = _oo(oid="K", cloid=make_cloid("sl1"), condition="Price below 95.0")
        tp = _oo(oid="TP", cloid="0xtp", side="sell", condition="Price above 130.0")
        orphan = _oo(oid="ORPH", cloid="0xdead", condition="Price below 94.0")
        with _patch_exchange(_POS, [keeper, tp, orphan]) as cancel:
            n = reconcile_positions(_cfg(), conn)
        assert n == 1
        cancel.assert_called_once()
        assert cancel.call_args[0][0] == "ORPH"           # TP never cancelled

    def test_short_mirror_and_na_spared(self, conn):
        _seed_entry(conn, "s2", coin="ZZZ", direction="SHORT")
        _seed_sl(conn, "s2", "slz", coin="ZZZ", direction="SHORT")
        pos = [{"coin": "ZZZ", "direction": "SHORT", "size": 1.0}]
        keeper = _oo(coin="ZZZ", oid="K", cloid=make_cloid("slz"),
                     side="buy", condition="Price above 105.0")   # SHORT SL
        tp = _oo(coin="ZZZ", oid="TP", cloid="0xtp", side="buy",
                 condition="Price below 80.0")                    # SHORT TP
        na = _oo(coin="ZZZ", oid="NA", cloid="0xna", side="buy",
                 condition="N/A")                                 # ambiguous
        orphan = _oo(coin="ZZZ", oid="ORPH", cloid="0xdead", side="buy",
                     condition="Price above 106.0")               # untracked SL
        with _patch_exchange(pos, [keeper, tp, na, orphan]) as cancel:
            n = reconcile_positions(_cfg(), conn)
        assert n == 1
        assert cancel.call_args[0][0] == "ORPH"           # TP + N/A spared


class TestReplacementKeeperEndToEnd:
    def test_drift_replaced_sl_is_keeper_not_cancelled(self, conn):
        # The R1-HIGH case end-to-end: predecessor SL row order_id=A
        # (CANCELLED), current PLACED row order_id=B (B != A); the live SL's
        # clientOrderId = make_cloid(A + ":drift") (predecessor-derived). It
        # MUST be recognised as a keeper, and a real orphan cancelled.
        _seed_entry(conn, "s1")
        _seed_sl(conn, "s1", "A", status="CANCELLED",
                 created="2026-01-01T00:00:00+00:00")
        _seed_sl(conn, "s1", "B", status="PLACED",
                 created="2026-01-01T01:00:00+00:00")
        live_keeper = _oo(oid="K", cloid=make_cloid("A" + ":drift"))
        orphan = _oo(oid="ORPH", cloid="0xdead")
        with _patch_exchange(_POS, [live_keeper, orphan]) as cancel:
            n = reconcile_positions(_cfg(), conn)
        assert n == 1
        cancel.assert_called_once()
        assert cancel.call_args[0][0] == "ORPH"           # keeper NOT cancelled


class TestNetDedup:
    def test_two_scans_one_net_position_orphan_cancelled_once(self, conn):
        # 2 DB-open rows share one HL net (coin,direction). Each scan's own SL
        # is a keeper; one true orphan → cancelled exactly once, count==1, no
        # double cancel_order call.
        _seed_entry(conn, "s1", coin="ABC", direction="LONG")
        _seed_entry(conn, "s2", coin="ABC", direction="LONG")
        _seed_sl(conn, "s1", "sl1")
        _seed_sl(conn, "s2", "sl2")
        k1 = _oo(oid="K1", cloid=make_cloid("sl1"))
        k2 = _oo(oid="K2", cloid=make_cloid("sl2"))
        orphan = _oo(oid="ORPH", cloid="0xdead")
        with _patch_exchange(_POS, [k1, k2, orphan]) as cancel:
            n = reconcile_positions(_cfg(), conn)
        assert n == 1
        cancel.assert_called_once()
        assert cancel.call_args[0][0] == "ORPH"

    def test_mixed_group_scan_without_db_chain_is_never_naked(self, conn):
        # codex phase-review-arch HIGH (reproduced): two DB-open scans share
        # one net (coin,direction). s1 has a tracked SL (keeper K1 live); s2
        # has NO stop_loss DB row but its OWN live SL on the exchange (the
        # lost-order_id crash this window exists for). A group-wide "any
        # keeper live" gate would let K1 prove the group and cancel s2's only
        # live SL → s2 transiently NAKED. The per-scan-complete gate must
        # SKIP the whole group → zero cancels, s2's SL preserved.
        _seed_entry(conn, "s1", coin="ABC", direction="LONG")
        _seed_entry(conn, "s2", coin="ABC", direction="LONG")
        _seed_sl(conn, "s1", "sl1")                       # only s1 has a chain
        k1 = _oo(oid="K1", cloid=make_cloid("sl1"))        # s1 keeper live
        s2_sl = _oo(oid="S2SL", cloid="0xs2only")          # s2's only live SL
        with _patch_exchange(_POS, [k1, s2_sl]) as cancel:
            n = reconcile_positions(_cfg(), conn)
        assert n == 0
        cancel.assert_not_called()                         # s2 NOT left naked


def test_get_exchange_open_orders_pins_frontend_method():
    # codex plan-review R2: the discriminator's info.* fields only exist on
    # the frontendOpenOrders shape — the method must be pinned explicitly.
    from scanner.execution.exchange import get_exchange_open_orders

    fake = type("X", (), {"fetch_open_orders": lambda self, **kw: kw})()
    with patch("scanner.execution.exchange.get_auth_exchange",
               return_value=fake):
        out = get_exchange_open_orders(_cfg())
    assert out["params"] == {"method": "frontendOpenOrders"}
