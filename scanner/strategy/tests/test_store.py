"""Tests for strategy CRUD operations."""

import pytest

from scanner.strategy.models import Strategy
from scanner.strategy.store import (
    save_strategy,
    get_strategy,
    get_active_strategy,
    list_strategies,
    activate_strategy,
    ensure_baseline,
)


class TestSaveAndGet:
    def test_save_and_retrieve(self, db_conn, baseline_strategy):
        save_strategy(baseline_strategy, db_conn)
        got = get_strategy("v0", db_conn)
        assert got is not None
        assert got.strategy_id == "v0"
        assert got.label == "baseline"
        assert got.swing_lookback == 5
        assert got.sr_cluster_pct == 0.015

    def test_get_nonexistent_returns_none(self, db_conn):
        assert get_strategy("nope", db_conn) is None

    def test_save_updates_on_conflict(self, db_conn, baseline_strategy):
        save_strategy(baseline_strategy, db_conn)
        baseline_strategy.label = "updated"
        save_strategy(baseline_strategy, db_conn)
        got = get_strategy("v0", db_conn)
        assert got.label == "updated"

    def test_custom_params_roundtrip(self, db_conn, custom_strategy):
        save_strategy(custom_strategy, db_conn)
        got = get_strategy("v1-wider", db_conn)
        assert got.swing_lookback == 7
        assert got.sr_cluster_pct == 0.02
        assert got.proximity_atr_mult == 2.0
        assert got.max_setups_per_coin == 5
        assert got.atr_stop_mult == 2.0
        assert got.atr_target_mult == 4.0
        assert got.notes == "Testing wider parameters"


class TestActivate:
    def test_activate_deactivates_others(self, db_conn, baseline_strategy, custom_strategy):
        baseline_strategy.is_active = True
        save_strategy(baseline_strategy, db_conn)
        save_strategy(custom_strategy, db_conn)

        activate_strategy("v1-wider", db_conn)

        v0 = get_strategy("v0", db_conn)
        v1 = get_strategy("v1-wider", db_conn)
        assert v0.is_active is False
        assert v1.is_active is True

    def test_get_active_strategy(self, db_conn, baseline_strategy):
        baseline_strategy.is_active = True
        save_strategy(baseline_strategy, db_conn)
        active = get_active_strategy(db_conn)
        assert active is not None
        assert active.strategy_id == "v0"

    def test_no_active_returns_none(self, db_conn):
        assert get_active_strategy(db_conn) is None


class TestListStrategies:
    def test_empty_list(self, db_conn):
        assert list_strategies(db_conn) == []

    def test_lists_all(self, db_conn, baseline_strategy, custom_strategy):
        save_strategy(baseline_strategy, db_conn)
        save_strategy(custom_strategy, db_conn)
        strategies = list_strategies(db_conn)
        assert len(strategies) == 2
        ids = {s.strategy_id for s in strategies}
        assert ids == {"v0", "v1-wider"}


class TestEnsureBaseline:
    def test_creates_baseline(self, db_conn):
        baseline = ensure_baseline(db_conn)
        assert baseline.strategy_id == "v0"
        assert baseline.label == "baseline"
        assert baseline.is_active is True
        # Verify persisted
        got = get_strategy("v0", db_conn)
        assert got is not None

    def test_idempotent(self, db_conn):
        ensure_baseline(db_conn)
        ensure_baseline(db_conn)
        strategies = list_strategies(db_conn)
        v0s = [s for s in strategies if s.strategy_id == "v0"]
        assert len(v0s) == 1


class TestToScannerConfig:
    def test_defaults_match(self, baseline_strategy):
        cfg = baseline_strategy.to_scanner_config()
        assert cfg.swing_lookback == 5
        assert cfg.sr_cluster_pct == 0.015
        assert cfg.proximity_atr_mult == 1.5
        assert cfg.max_setups_per_coin == 3

    def test_custom_params(self, custom_strategy):
        cfg = custom_strategy.to_scanner_config()
        assert cfg.swing_lookback == 7
        assert cfg.sr_cluster_pct == 0.02
        assert cfg.proximity_atr_mult == 2.0
        assert cfg.max_setups_per_coin == 5
