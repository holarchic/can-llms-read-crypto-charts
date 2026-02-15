"""Tests for SQLite storage — CRUD, dedup, cost tracking."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scanner.storage import get_connection, save_result, is_duplicate, mark_alerted, daily_cost, stats
from scanner.models import (
    ScanResult, SetupSignal, SetupType, Direction,
    MarketContext, Structure, SRLevel, SwingPoint,
)


@pytest.fixture
def db_conn(tmp_path):
    """In-memory-like DB using a temp file."""
    conn = get_connection(tmp_path / "test.db")
    yield conn
    conn.close()


@pytest.fixture
def sample_result():
    """A minimal ScanResult for testing."""
    ctx = MarketContext(
        structure=Structure.RANGING,
        swings=[],
        sr_levels=[],
        msbs=[],
        range_bounds=(95.0, 105.0),
        atr=2.0,
        current_price=100.0,
        fib_levels=None,
    )
    signal = SetupSignal(
        coin="BTC",
        setup_type=SetupType.C,
        direction=Direction.LONG,
        confidence=0.7,
        key_level=65000.0,
        context=ctx,
        description="Test signal",
    )
    return ScanResult(
        scan_id="test1234",
        coin="BTC",
        timestamp=datetime.now(timezone.utc),
        setup_signal=signal,
        decision="CONFIRM",
        confluence_score="9/17 — A+",
        analysis_text="Full analysis text here",
        parsed_decision={"decision": "CONFIRM", "entry_zone": "$64,500", "stop_loss": "$62,000"},
        chart_paths=["/tmp/BTC_4H.png"],
        cost_usd=0.05,
    )


class TestSaveAndRetrieve:
    def test_save_result(self, db_conn, sample_result):
        save_result(sample_result, db_conn)
        row = db_conn.execute("SELECT * FROM scans WHERE scan_id = ?", ("test1234",)).fetchone()
        assert row is not None
        assert row["coin"] == "BTC"
        assert row["decision"] == "CONFIRM"

    def test_save_updates_on_conflict(self, db_conn, sample_result):
        save_result(sample_result, db_conn)
        sample_result.decision = "REJECT"
        save_result(sample_result, db_conn)
        row = db_conn.execute("SELECT * FROM scans WHERE scan_id = ?", ("test1234",)).fetchone()
        assert row["decision"] == "REJECT"


class TestDedup:
    def test_not_duplicate_when_empty(self, db_conn):
        assert is_duplicate("BTC", "C", "LONG", 12, db_conn) is False

    def test_is_duplicate_after_save(self, db_conn, sample_result):
        save_result(sample_result, db_conn)
        assert is_duplicate("BTC", "C", "LONG", 12, db_conn) is True

    def test_different_coin_not_duplicate(self, db_conn, sample_result):
        save_result(sample_result, db_conn)
        assert is_duplicate("ETH", "C", "LONG", 12, db_conn) is False

    def test_different_direction_not_duplicate(self, db_conn, sample_result):
        save_result(sample_result, db_conn)
        assert is_duplicate("BTC", "C", "SHORT", 12, db_conn) is False


class TestMarkAlerted:
    def test_marks_alerted(self, db_conn, sample_result):
        save_result(sample_result, db_conn)
        mark_alerted("test1234", db_conn)
        row = db_conn.execute("SELECT alerted FROM scans WHERE scan_id = ?", ("test1234",)).fetchone()
        assert row["alerted"] == 1


class TestDailyCost:
    def test_zero_when_empty(self, db_conn):
        assert daily_cost(db_conn) == 0.0

    def test_sums_cost(self, db_conn, sample_result):
        save_result(sample_result, db_conn)
        assert daily_cost(db_conn) == 0.05


class TestStats:
    def test_empty_stats(self, db_conn):
        s = stats(db_conn)
        assert s["total_scans"] == 0
        assert s["confirms"] == 0

    def test_stats_after_save(self, db_conn, sample_result):
        save_result(sample_result, db_conn)
        s = stats(db_conn)
        assert s["total_scans"] == 1
        assert s["confirms"] == 1
        assert len(s["recent"]) == 1
