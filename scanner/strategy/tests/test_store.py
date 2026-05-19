"""Tests for strategy CRUD — disabled_types per strategy_id (decision #31)."""

from scanner.strategy.models import Strategy
from scanner.strategy.store import (
    ensure_baseline,
    get_strategy,
    list_strategies,
    set_disabled_types,
)


class TestGetAndBaseline:
    def test_get_nonexistent_returns_none(self, db_conn):
        assert get_strategy("nope", db_conn) is None

    def test_ensure_baseline_creates_v0(self, db_conn):
        baseline = ensure_baseline(db_conn)
        assert baseline.strategy_id == "v0"
        assert baseline.disabled_types == ""
        got = get_strategy("v0", db_conn)
        assert got is not None and got.strategy_id == "v0"

    def test_ensure_baseline_idempotent(self, db_conn):
        ensure_baseline(db_conn)
        ensure_baseline(db_conn)
        v0s = [s for s in list_strategies(db_conn) if s.strategy_id == "v0"]
        assert len(v0s) == 1


class TestSetDisabledTypes:
    def test_set_and_read_back(self, db_conn):
        ensure_baseline(db_conn)
        set_disabled_types("v0", "A,E", db_conn)
        got = get_strategy("v0", db_conn)
        assert got.disabled_types == "A,E"
        assert got.disabled_types_set == {"A", "E"}

    def test_clear(self, db_conn):
        ensure_baseline(db_conn)
        set_disabled_types("v0", "A,E", db_conn)
        set_disabled_types("v0", "", db_conn)
        got = get_strategy("v0", db_conn)
        assert got.disabled_types == ""
        assert got.disabled_types_set == set()

    def test_preserves_rest_of_row_no_created_at_churn(self, db_conn):
        # decision #31 no-clobber: a single-column in-place UPDATE must NOT
        # rewrite created_at or the inert defaulted columns.
        ensure_baseline(db_conn)
        before = db_conn.execute(
            "SELECT created_at, label, swing_lookback FROM strategies WHERE strategy_id='v0'"
        ).fetchone()
        set_disabled_types("v0", "G", db_conn)
        after = db_conn.execute(
            "SELECT created_at, label, swing_lookback FROM strategies WHERE strategy_id='v0'"
        ).fetchone()
        assert after["created_at"] == before["created_at"]
        assert after["label"] == before["label"]
        assert after["swing_lookback"] == before["swing_lookback"]


class TestListStrategies:
    def test_empty(self, db_conn):
        assert list_strategies(db_conn) == []

    def test_lists_ids_and_disabled(self, db_conn):
        ensure_baseline(db_conn)
        set_disabled_types("v0", "E,G", db_conn)
        strategies = list_strategies(db_conn)
        assert len(strategies) == 1
        assert strategies[0].strategy_id == "v0"
        assert strategies[0].disabled_types == "E,G"


class TestDisabledTypesSet:
    def test_parsing(self):
        assert Strategy("v0", "").disabled_types_set == set()
        assert Strategy("v0", "a, e ,G").disabled_types_set == {"A", "E", "G"}
