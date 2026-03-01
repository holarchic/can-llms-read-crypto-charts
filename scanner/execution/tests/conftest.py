"""Shared fixtures for execution tests."""

import sqlite3

import pytest

from scanner.execution.config import ExecutionConfig


@pytest.fixture
def exec_config():
    """Default execution config for testing (disabled, testnet)."""
    return ExecutionConfig()


@pytest.fixture
def enabled_config():
    """Execution config with execution enabled (testnet)."""
    return ExecutionConfig(
        enabled=True,
        mode="testnet",
        private_key="0xtest_key",
        wallet_address="0xtest_wallet",
        risk_pct=1.0,
        gcp_project_id="test-project",
    )


@pytest.fixture
def db_conn(tmp_path):
    """In-memory SQLite connection with orders table."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn
