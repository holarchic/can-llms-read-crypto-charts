"""Shared fixtures for strategy tests."""

import pytest

from scanner.db import get_connection
from scanner.strategy.models import Strategy


@pytest.fixture
def db_conn(tmp_path):
    """Fresh DB connection with all tables created."""
    conn = get_connection(tmp_path / "test.db")
    yield conn
    conn.close()


@pytest.fixture
def baseline_strategy():
    """Default v0 strategy with all defaults."""
    return Strategy(strategy_id="v0", label="baseline", is_active=True)


@pytest.fixture
def custom_strategy():
    """Non-default strategy for testing."""
    return Strategy(
        strategy_id="v1-wider",
        label="wider stops",
        swing_lookback=7,
        sr_cluster_pct=0.02,
        proximity_atr_mult=2.0,
        max_setups_per_coin=5,
        atr_stop_mult=2.0,
        atr_target_mult=4.0,
        notes="Testing wider parameters",
    )
