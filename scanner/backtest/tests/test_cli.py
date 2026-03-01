"""Tests for backtest CLI helpers."""

from datetime import datetime, timezone

import pytest

from scanner.backtest.cli import _load_llm_signals
from scanner.models import (
    Direction,
    MarketContext,
    ScanResult,
    SetupSignal,
    SetupType,
    Structure,
)
from scanner.storage import get_connection, save_result


@pytest.fixture
def db_conn(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    yield conn
    conn.close()


def _make_result(scan_id, coin="BTC", decision="CONFIRM"):
    ctx = MarketContext(
        structure=Structure.RANGING, swings=[], sr_levels=[], msbs=[],
        range_bounds=(95.0, 105.0), atr=2.0, current_price=100.0, fib_levels=None,
    )
    signal = SetupSignal(
        coin=coin, setup_type=SetupType.C, direction=Direction.LONG,
        confidence=0.7, key_level=65000.0, context=ctx, description="Test",
    )
    return ScanResult(
        scan_id=scan_id, coin=coin, timestamp=datetime.now(timezone.utc),
        setup_signal=signal, decision=decision, confluence_score=None,
        analysis_text="test", parsed_decision={"decision": decision},
        chart_paths=[], cost_usd=0.01,
    )


class TestLoadLlmSignalsPromptVersion:
    def test_filters_by_prompt_version(self, db_conn):
        r1 = _make_result("v1_scan")
        r2 = _make_result("v2_scan")
        save_result(r1, db_conn, prompt_version=None)
        save_result(r2, db_conn, prompt_version="2")

        # Filter for v2 only
        rows = _load_llm_signals(db_conn, days=30, coin=None, setup_type=None, prompt_version="2")
        assert len(rows) == 1
        assert rows[0]["scan_id"] == "v2_scan"

    def test_no_filter_returns_all(self, db_conn):
        r1 = _make_result("v1_scan")
        r2 = _make_result("v2_scan")
        save_result(r1, db_conn, prompt_version=None)
        save_result(r2, db_conn, prompt_version="2")

        rows = _load_llm_signals(db_conn, days=30, coin=None, setup_type=None)
        assert len(rows) == 2

    def test_excludes_rejects(self, db_conn):
        r1 = _make_result("reject_scan", decision="REJECT")
        save_result(r1, db_conn, prompt_version="2")

        rows = _load_llm_signals(db_conn, days=30, coin=None, setup_type=None, prompt_version="2")
        assert len(rows) == 0
