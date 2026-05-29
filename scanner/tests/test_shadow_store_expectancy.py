"""T-020.1 (SPEC #41): shadow `store` read/upsert seams + the `expectancy` Gate-#10 readout."""

import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import scanner.schema as schema
from scanner.models import ConfluenceFactor, MachineBlock, ShadowOutcome
from scanner.shadow.expectancy import bootstrap_ci, expectancy_report
from scanner.shadow.store import (
    load_unresolved_signals,
    resolved_r_values,
    save_resolution,
    save_shadow_signal,
)

_ANCHOR = datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc)
XP = "v1"  # exit_policy_version
EP = "v1"  # entry_policy_version


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row   # mirror production get_connection (named row access)
    schema._run_migrations(c)
    yield c
    c.close()


def _capture(conn, scan_id: str) -> int:
    mb = MachineBlock.from_raw({
        "entry_low": 100.0, "entry_high": 101.0, "stop_loss": 95.0,
        "tp1": 110.0, "tp1_exit_pct": 50, "tp2": 120.0, "tp2_exit_pct": 50,
        "direction": "LONG", "setup_type": "trend_continuation",
        "setup_quality": "textbook", "confidence": 7, "leverage": 5,
    })
    sig = SimpleNamespace(
        confluence=frozenset({ConfluenceFactor.TREND_ALIGNED, ConfluenceFactor.RETEST}),
    )
    confirm = SimpleNamespace(
        scan_id=scan_id, coin="BTC", machine_block=mb,
        setup_signal=sig, prompt_version="10.00", validator_verdict=None,
    )
    save_shadow_signal(confirm, _ANCHOR, "hyperliquid", conn)
    return conn.execute(
        "SELECT id FROM shadow_signals WHERE scan_id = ?", (scan_id,)
    ).fetchone()[0]


# --- store ---------------------------------------------------------------

def test_load_unresolved_rehydrates_signal(conn):
    _capture(conn, "s1")
    rows = load_unresolved_signals(conn, XP, EP)
    assert len(rows) == 1
    assert rows[0].coin == "BTC" and rows[0].exchange == "hyperliquid"
    assert rows[0].anchor_ts == _ANCHOR
    assert rows[0].machine_block.direction == "LONG"


def test_resolved_excluded_open_reincluded(conn):
    sid1 = _capture(conn, "s1")
    save_resolution(conn, sid1, ShadowOutcome(status="RESOLVED", realized_r=1.2), XP, EP, _ANCHOR)
    sid2 = _capture(conn, "s2")
    save_resolution(conn, sid2, ShadowOutcome(status="OPEN"), XP, EP, _ANCHOR)
    ids = {r.id for r in load_unresolved_signals(conn, XP, EP)}
    assert ids == {sid2}  # terminal excluded; OPEN re-included for re-resolution


def test_save_resolution_upserts(conn):
    sid = _capture(conn, "s1")
    save_resolution(conn, sid, ShadowOutcome(status="OPEN"), XP, EP, _ANCHOR)
    save_resolution(conn, sid, ShadowOutcome(status="RESOLVED", realized_r=0.8), XP, EP, _ANCHOR)
    n = conn.execute(
        "SELECT COUNT(*) FROM shadow_resolutions WHERE shadow_signal_id = ?", (sid,)
    ).fetchone()[0]
    assert n == 1                       # upsert, not a duplicate row
    assert resolved_r_values(conn, XP, EP) == [0.8]


def test_resolved_r_values_only_resolved_trades(conn):
    a = _capture(conn, "a")
    b = _capture(conn, "b")
    c = _capture(conn, "c")
    save_resolution(conn, a, ShadowOutcome(status="RESOLVED", realized_r=1.5), XP, EP, _ANCHOR)
    save_resolution(conn, b, ShadowOutcome(status="TRIGGER_EXPIRED"), XP, EP, _ANCHOR)
    save_resolution(conn, c, ShadowOutcome(status="RESOLVED", realized_r=-1.0), XP, EP, _ANCHOR)
    assert sorted(resolved_r_values(conn, XP, EP)) == [-1.0, 1.5]  # non-trade excluded


# --- expectancy ----------------------------------------------------------

def test_gate_passes_on_positive_sample_n30():
    rep = expectancy_report([1.0] * 30, iters=500, seed=1)
    assert rep.n == 30 and rep.avg_r == 1.0 and rep.ci_lo > 0 and rep.gate_pass is True


def test_gate_fails_below_min_n():
    assert expectancy_report([1.0] * 10, iters=500).gate_pass is False


def test_gate_fails_when_ci_straddles_zero():
    rep = expectancy_report([1.0] * 20 + [-1.0] * 20, iters=800, seed=2)
    assert rep.n == 40 and rep.ci_lo <= 0 and rep.gate_pass is False


def test_zero_r_trades_stay_in_sample():
    rep = expectancy_report([1.0, 1.0, 0.0, -1.0], iters=200)
    assert rep.n == 4 and rep.win_rate == 0.5  # 0R counted in n, not a win


def test_bootstrap_is_deterministic_with_seed():
    rs = [0.5, -1.0, 2.0, 0.3, -1.0, 1.5]
    assert bootstrap_ci(rs, iters=300, seed=7) == bootstrap_ci(rs, iters=300, seed=7)


def test_profit_factor():
    assert expectancy_report([2.0, 1.0, -1.0], iters=100).profit_factor == 3.0
