"""T-020.1 shadow-mode capture (SPEC #41) — `shadow.store.save_shadow_signal`.

The store reads only a small attribute surface off a post-validator CONFIRM
(`scan_id`, `coin`, `machine_block`, `setup_signal.confluence`, `prompt_version`,
`validator_verdict`), so the cohort is duck-typed here — no heavy `ScanResult` /
`BoundarySignal` fixture needed. The point under test is the capture seam, not the
upstream pipeline.
"""

import json
import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import scanner.schema as schema
from scanner.models import ConfluenceFactor, MachineBlock
from scanner.shadow.store import save_shadow_signal

_ANCHOR = datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    schema._run_migrations(c)
    yield c
    c.close()


def _machine_block() -> MachineBlock:
    return MachineBlock.from_raw({
        "entry_low": 100.0, "entry_high": 101.0, "stop_loss": 95.0,
        "tp1": 110.0, "tp1_exit_pct": 50, "tp2": 120.0, "tp2_exit_pct": 50,
        "direction": "LONG", "setup_type": "trend_continuation",
        "setup_quality": "textbook", "confidence": 7, "leverage": 5,
    })


def _confirm(scan_id: str = "scan-1"):
    """A post-validator CONFIRM, duck-typed to the store's read surface."""
    sig = SimpleNamespace(
        confluence=frozenset({ConfluenceFactor.TREND_ALIGNED, ConfluenceFactor.RETEST}),
    )
    return SimpleNamespace(
        scan_id=scan_id, coin="BTC", machine_block=_machine_block(),
        setup_signal=sig, prompt_version="10.00", validator_verdict=None,
    )


def test_capture_persists_post_validator_confirm(conn):
    save_shadow_signal(_confirm(), _ANCHOR, "hyperliquid", conn)

    row = conn.execute(
        "SELECT scan_id, coin, exchange, anchor_ts, setup_type, direction, "
        "prompt_version, confluence_count FROM shadow_signals"
    ).fetchone()
    assert row is not None
    assert row[0] == "scan-1"          # scan_id — the deterministic link to scans
    assert row[1] == "BTC"
    assert row[2] == "hyperliquid"     # venue pinned for forward OHLCV fetch
    assert row[3] == _ANCHOR.isoformat()
    assert row[4] == "trend_continuation"
    assert row[5] == "LONG"
    assert row[6] == "10.00"
    assert row[7] == 2                 # two confluence factors captured at scan time


def test_capture_records_full_geometry(conn):
    save_shadow_signal(_confirm("scan-geo"), _ANCHOR, "hyperliquid", conn)
    mb = json.loads(conn.execute(
        "SELECT machine_block FROM shadow_signals WHERE scan_id='scan-geo'"
    ).fetchone()[0])
    assert mb["stop_loss"] == 95.0 and mb["tp1"] == 110.0 and mb["entry_low"] == 100.0


def test_capture_idempotent_on_scan_id(conn):
    save_shadow_signal(_confirm("dup"), _ANCHOR, "hyperliquid", conn)
    save_shadow_signal(_confirm("dup"), _ANCHOR, "hyperliquid", conn)
    n = conn.execute(
        "SELECT COUNT(*) FROM shadow_signals WHERE scan_id='dup'"
    ).fetchone()[0]
    assert n == 1
