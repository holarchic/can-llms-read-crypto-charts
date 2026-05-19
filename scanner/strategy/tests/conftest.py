"""Shared fixtures for strategy tests."""

import pytest

from scanner.db import get_connection


@pytest.fixture
def db_conn(tmp_path):
    """Fresh DB connection with all tables created."""
    conn = get_connection(tmp_path / "test.db")
    yield conn
    conn.close()
