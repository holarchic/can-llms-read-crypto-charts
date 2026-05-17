"""Tests for SQLite storage — CRUD, dedup, cost tracking."""

import sqlite3
from datetime import datetime, timezone

import pytest

from scanner import db as scanner_db
from scanner.db import get_connection
from scanner.models import (
    Direction,
    MachineBlock,
    MarketContext,
    ScanResult,
    SetupSignal,
    SetupType,
    Structure,
)
from scanner.storage import daily_cost, is_duplicate, mark_alerted, save_result, save_scan_universe, stats


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

    def test_save_result_with_typed_machine_block(self, db_conn, sample_result):
        """T-016 phase-review regression: save_result must persist a CONFIRM
        carrying a typed MachineBlock. The pre-fix code did
        result.machine_block.get("direction") which AttributeError'd after
        the C6 shim removal — the suite stayed green only because no fixture
        ever set a non-None machine_block here (the coverage gap that let the
        bug escape). Asserts: no crash, scans.machine_block JSON == to_raw(),
        llm_direction column == mb.direction.
        """
        import json

        mb = MachineBlock.from_raw({
            "entry_low": 99.0, "entry_high": 101.0, "stop_loss": 95.0,
            "tp1": 110.0, "tp1_exit_pct": 50, "direction": "LONG",
            "confidence": 7, "leverage": 5, "tp2": 115.0, "tp2_exit_pct": 50,
            "setup_type": "C", "confluence_score": 9,
            "confirming_event": "1H close above flipped resistance level",
        })
        assert mb is not None
        sample_result.machine_block = mb

        save_result(sample_result, db_conn)

        row = db_conn.execute(
            "SELECT machine_block, llm_direction FROM scans WHERE scan_id = ?",
            ("test1234",),
        ).fetchone()
        assert row is not None
        assert row["llm_direction"] == "LONG"
        assert json.loads(row["machine_block"]) == mb.to_raw()

    def test_collision_raises_integrity_error(self, db_conn, sample_result):
        """scan_id reuse now raises — caller is expected to use unique IDs.

        Silent regeneration mutated result.scan_id after other rows had
        already referenced the old ID, creating orphaned cross-table links.
        """
        import pytest as _pytest

        save_result(sample_result, db_conn)
        sample_result.decision = "REJECT"
        with _pytest.raises(sqlite3.IntegrityError):
            save_result(sample_result, db_conn)


class TestLegacyMigration:
    def test_legacy_direction_column_migrates_without_machine_block(self, tmp_path):
        """Legacy DB has 'direction' column and no machine_block. The migration
        must not blow up when trying to backfill llm_direction from a column
        that doesn't yet exist — it should add llm_direction and skip the
        backfill gracefully."""
        import sqlite3

        # Simulate pre-rename schema
        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE scans ("
            "scan_id TEXT PRIMARY KEY, coin TEXT NOT NULL, timestamp TEXT NOT NULL, "
            "setup_type TEXT NOT NULL, setup_name TEXT, direction TEXT NOT NULL, "
            "screener_confidence REAL, key_level REAL, decision TEXT, "
            "confluence_score TEXT, entry_zone TEXT, stop_loss TEXT, "
            "take_profit TEXT, reasoning TEXT, cost_usd REAL DEFAULT 0, "
            "alerted INTEGER DEFAULT 0, created_at TEXT NOT NULL)",
        )
        conn.commit()
        conn.close()

        # Open via get_connection — runs migrations. Should succeed.
        conn = get_connection(db_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(scans)").fetchall()}
        assert "screener_direction" in cols
        assert "llm_direction" in cols
        assert "machine_block" in cols
        conn.close()


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


class TestScanUniverse:
    def test_save_scan_universe(self, db_conn):
        pairs = [
            {"coin": "BTC", "volume_24h": 100_000_000, "max_leverage": 50},
            {"coin": "ETH", "volume_24h": 50_000_000, "max_leverage": 25},
        ]
        count = save_scan_universe("batch1", pairs, "2026-02-20T12:00:00", db_conn)
        assert count == 2

        rows = db_conn.execute("SELECT * FROM scan_universe WHERE batch_id = 'batch1'").fetchall()
        assert len(rows) == 2
        coins = {r["coin"] for r in rows}
        assert coins == {"BTC", "ETH"}

    def test_dedup_on_batch_coin(self, db_conn):
        pairs = [{"coin": "BTC", "volume_24h": 100_000_000}]
        save_scan_universe("batch1", pairs, "2026-02-20T12:00:00", db_conn)
        # Insert same batch_id + coin again — should be ignored
        save_scan_universe("batch1", pairs, "2026-02-20T12:00:00", db_conn)

        rows = db_conn.execute("SELECT COUNT(*) as cnt FROM scan_universe WHERE batch_id = 'batch1'").fetchone()
        assert rows["cnt"] == 1

    def test_different_batches_separate(self, db_conn):
        pairs = [{"coin": "BTC"}]
        save_scan_universe("batch1", pairs, "2026-02-20T12:00:00", db_conn)
        save_scan_universe("batch2", pairs, "2026-02-20T13:00:00", db_conn)

        rows = db_conn.execute("SELECT COUNT(DISTINCT batch_id) as cnt FROM scan_universe").fetchone()
        assert rows["cnt"] == 2

    def test_empty_pairs_returns_zero(self, db_conn):
        assert save_scan_universe("batch1", [], "2026-02-20T12:00:00", db_conn) == 0


class TestPromptVersion:
    def test_prompt_version_roundtrips(self, db_conn, sample_result):
        save_result(sample_result, db_conn, prompt_version="2")
        row = db_conn.execute("SELECT prompt_version FROM scans WHERE scan_id = ?", ("test1234",)).fetchone()
        assert row["prompt_version"] == "2"

    def test_prompt_version_defaults_to_null(self, db_conn, sample_result):
        save_result(sample_result, db_conn)
        row = db_conn.execute("SELECT prompt_version FROM scans WHERE scan_id = ?", ("test1234",)).fetchone()
        assert row["prompt_version"] is None


class TestMigrationCache:
    def setup_method(self):
        """Clear migration cache between tests."""
        scanner_db._migrated.clear()

    def test_get_connection_migrations_run_once(self, tmp_path):
        """Second get_connection() with same path skips migrations; the
        assembly root composes BOTH scanner + execution schema (T-016 2a)."""
        db = tmp_path / "cache_test.db"
        conn1 = get_connection(db)
        conn1.close()

        resolved = str(db.resolve())
        assert resolved in scanner_db._migrated

        # Second call — _migrated already contains the path
        conn2 = get_connection(db)
        conn2.close()

        # Fresh close/reopen — verify the composed bootstrap actually ran
        # from scanner.db (scanner-owned AND execution-owned tables).
        conn3 = get_connection(db)
        tables = {
            row[0] for row in conn3.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn3.close()
        assert "scans" in tables
        assert "signals" in tables
        assert "orders" in tables  # execution schema composed by scanner.db
        assert "pending_entries" in tables

    def test_get_connection_different_paths_both_migrate(self, tmp_path):
        """Two different db paths each get their own migrations."""
        db1 = tmp_path / "a.db"
        db2 = tmp_path / "b.db"

        conn1 = get_connection(db1)
        conn1.close()
        conn2 = get_connection(db2)
        conn2.close()

        assert str(db1.resolve()) in scanner_db._migrated
        assert str(db2.resolve()) in scanner_db._migrated
