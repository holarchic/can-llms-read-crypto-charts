"""Tests for signal persistence — save_signals, signal_history, signal_stats."""


import pytest

from scanner.db import get_connection
from scanner.models import (
    BoundarySignal,
    Direction,
    FibLevels,
    MarketContext,
    SetupSignal,
    SetupType,
    Structure,
)
from scanner.scan_queries import signal_distribution, signal_history, signal_stats
from scanner.scan_store import save_signals


@pytest.fixture
def db_conn(tmp_path):
    conn = get_connection(tmp_path / "test_signals.db")
    yield conn
    conn.close()


def _make_signal(
    coin="BTC",
    setup_type=SetupType.C,
    direction=Direction.LONG,
    confidence=0.7,
    key_level=65000.0,
    structure=Structure.RANGING,
    range_bounds=(95.0, 105.0),
    atr=2.0,
    current_price=100.0,
    fib_levels=None,
):
    ctx = MarketContext(
        structure=structure,
        swings=[],
        sr_levels=[],
        msbs=[],
        range_bounds=range_bounds,
        atr=atr,
        current_price=current_price,
        fib_levels=fib_levels,
    )
    return SetupSignal(
        coin=coin,
        setup_type=setup_type,
        direction=direction,
        confidence=confidence,
        key_level=key_level,
        context=ctx,
        description=f"Test signal for {coin}",
    )


class TestSaveSignals:
    def test_single_signal(self, db_conn):
        sig = _make_signal()
        save_signals([sig], "batch001", 1, db_conn)

        row = db_conn.execute("SELECT * FROM signals").fetchone()
        assert row is not None
        assert row["coin"] == "BTC"
        assert row["setup_type"] == "C"
        assert row["direction"] == "LONG"
        assert row["batch_id"] == "batch001"

    def test_multiple_signals(self, db_conn):
        sigs = [
            _make_signal(coin="BTC", confidence=0.9),
            _make_signal(coin="ETH", confidence=0.8),
            _make_signal(coin="SOL", confidence=0.6),
        ]
        save_signals(sigs, "batch002", 2, db_conn)

        rows = db_conn.execute("SELECT * FROM signals ORDER BY rank_in_batch").fetchall()
        assert len(rows) == 3
        assert rows[0]["coin"] == "BTC"
        assert rows[2]["coin"] == "SOL"

    def test_ranking(self, db_conn):
        sigs = [
            _make_signal(coin="BTC", confidence=0.9),
            _make_signal(coin="ETH", confidence=0.7),
        ]
        save_signals(sigs, "batch003", 2, db_conn)

        rows = db_conn.execute("SELECT * FROM signals ORDER BY rank_in_batch").fetchall()
        assert rows[0]["rank_in_batch"] == 1
        assert rows[1]["rank_in_batch"] == 2
        assert rows[0]["total_in_batch"] == 2
        assert rows[1]["total_in_batch"] == 2

    def test_sent_to_llm_flag(self, db_conn):
        sigs = [
            _make_signal(coin="BTC", confidence=0.9),
            _make_signal(coin="ETH", confidence=0.7),
            _make_signal(coin="SOL", confidence=0.5),
        ]
        save_signals(sigs, "batch004", 2, db_conn)

        rows = db_conn.execute("SELECT * FROM signals ORDER BY rank_in_batch").fetchall()
        assert rows[0]["sent_to_llm"] == 1  # rank 1 <= 2
        assert rows[1]["sent_to_llm"] == 1  # rank 2 <= 2
        assert rows[2]["sent_to_llm"] == 0  # rank 3 > 2

    def test_context_scalars(self, db_conn):
        sig = _make_signal(current_price=64500.0, atr=1200.0, structure=Structure.BULLISH)
        save_signals([sig], "batch005", 1, db_conn)

        row = db_conn.execute("SELECT * FROM signals").fetchone()
        assert row["current_price"] == 64500.0
        assert row["atr"] == 1200.0
        assert row["structure"] == "BULLISH"

    def test_range_bounds(self, db_conn):
        sig = _make_signal(range_bounds=(60000.0, 70000.0))
        save_signals([sig], "batch006", 1, db_conn)

        row = db_conn.execute("SELECT * FROM signals").fetchone()
        assert row["range_low"] == 60000.0
        assert row["range_high"] == 70000.0

    def test_no_range_bounds(self, db_conn):
        sig = _make_signal(range_bounds=None)
        save_signals([sig], "batch007", 1, db_conn)

        row = db_conn.execute("SELECT * FROM signals").fetchone()
        assert row["range_low"] is None
        assert row["range_high"] is None

    def test_fib_levels(self, db_conn):
        fibs = FibLevels(
            swing_low=60000.0,
            swing_high=70000.0,
            fib_618=63820.0,
            fib_786=62140.0,
            ote_zone=(63820.0, 62140.0),
        )
        sig = _make_signal(fib_levels=fibs)
        save_signals([sig], "batch008", 1, db_conn)

        row = db_conn.execute("SELECT * FROM signals").fetchone()
        assert row["fib_618"] == 63820.0
        assert row["fib_786"] == 62140.0

    def test_no_fib_levels(self, db_conn):
        sig = _make_signal(fib_levels=None)
        save_signals([sig], "batch009", 1, db_conn)

        row = db_conn.execute("SELECT * FROM signals").fetchone()
        assert row["fib_618"] is None
        assert row["fib_786"] is None

    def test_empty_list(self, db_conn):
        save_signals([], "batch010", 0, db_conn)
        count = db_conn.execute("SELECT COUNT(*) as cnt FROM signals").fetchone()["cnt"]
        assert count == 0

    def test_separate_batches(self, db_conn):
        save_signals([_make_signal(coin="BTC")], "batch_a", 1, db_conn)
        save_signals([_make_signal(coin="ETH")], "batch_b", 1, db_conn)

        batches = db_conn.execute(
            "SELECT DISTINCT batch_id FROM signals ORDER BY batch_id"
        ).fetchall()
        assert len(batches) == 2
        assert batches[0]["batch_id"] == "batch_a"
        assert batches[1]["batch_id"] == "batch_b"


class TestSignalHistory:
    def test_empty(self, db_conn):
        rows = signal_history(conn=db_conn)
        assert rows == []

    def test_basic(self, db_conn):
        save_signals([_make_signal()], "batch001", 1, db_conn)
        rows = signal_history(conn=db_conn)
        assert len(rows) == 1
        assert rows[0]["coin"] == "BTC"

    def test_coin_filter(self, db_conn):
        save_signals([
            _make_signal(coin="BTC"),
            _make_signal(coin="ETH"),
        ], "batch001", 2, db_conn)

        rows = signal_history(coin="ETH", conn=db_conn)
        assert len(rows) == 1
        assert rows[0]["coin"] == "ETH"

    def test_type_filter(self, db_conn):
        save_signals([
            _make_signal(setup_type=SetupType.A),
            _make_signal(setup_type=SetupType.C, coin="ETH"),
        ], "batch001", 2, db_conn)

        rows = signal_history(setup_type="A", conn=db_conn)
        assert len(rows) == 1
        assert rows[0]["setup_type"] == "A"

    def test_sent_only_filter(self, db_conn):
        save_signals([
            _make_signal(coin="BTC", confidence=0.9),
            _make_signal(coin="ETH", confidence=0.5),
        ], "batch001", 1, db_conn)

        rows = signal_history(sent_only=True, conn=db_conn)
        assert len(rows) == 1
        assert rows[0]["coin"] == "BTC"


class TestSignalStats:
    def test_empty(self, db_conn):
        s = signal_stats(conn=db_conn)
        assert s["total_signals"] == 0
        assert s["sent_to_llm"] == 0
        assert s["filtered"] == 0
        assert s["batches"] == 0
        assert s["by_type"] == []
        assert s["top_coins"] == []

    def test_populated(self, db_conn):
        save_signals([
            _make_signal(coin="BTC", setup_type=SetupType.A, confidence=0.9),
            _make_signal(coin="ETH", setup_type=SetupType.C, confidence=0.7),
            _make_signal(coin="SOL", setup_type=SetupType.A, confidence=0.5),
        ], "batch001", 2, db_conn)

        s = signal_stats(conn=db_conn)
        assert s["total_signals"] == 3
        assert s["sent_to_llm"] == 2
        assert s["filtered"] == 1
        assert s["batches"] == 1

        # by_type sorted by count desc — A has 2, C has 1
        assert len(s["by_type"]) == 2
        assert s["by_type"][0]["setup_type"] == "A"
        assert s["by_type"][0]["cnt"] == 2

        # top_coins
        assert len(s["top_coins"]) == 3


class TestSignalDistribution:
    """Direct coverage for signal_distribution — runtime/test-dead pre the
    T-016 2b split (no caller, no test); this gates its verbatim move."""

    def test_empty(self, db_conn):
        d = signal_distribution(conn=db_conn)
        assert d["total"] == 0
        assert d["batches"] == 0
        assert d["by_type"] == []
        assert d["by_direction"] == []

    def test_grouping(self, db_conn):
        save_signals([
            _make_signal(coin="BTC", setup_type=SetupType.A, direction=Direction.LONG),
            _make_signal(coin="ETH", setup_type=SetupType.A, direction=Direction.SHORT),
            _make_signal(coin="SOL", setup_type=SetupType.C, direction=Direction.LONG),
        ], "batch001", 3, db_conn)

        d = signal_distribution(conn=db_conn)
        assert d["total"] == 3
        assert d["batches"] == 1

        by_type = {r["setup_type"]: r for r in d["by_type"]}
        assert by_type["A"]["count"] == 2
        assert by_type["A"]["long"] == 1
        assert by_type["A"]["short"] == 1
        assert by_type["C"]["count"] == 1
        assert by_type["C"]["long"] == 1

        by_dir = {r["dir"]: r["count"] for r in d["by_direction"]}
        assert by_dir["LONG"] == 2
        assert by_dir["SHORT"] == 1


def test_storage_split_move_completeness():
    """T-016 2b move-completeness tripwire (SPEC #23): the union of the new
    modules' declared public surface must EXACTLY equal the pre-split
    `scanner/storage.py` public surface — a dropped/renamed/misplaced
    function during the verbatim split fails closed here. `__all__` (not a
    dir() scan) so imported names (get_connection, SETUP_NAMES) don't
    pollute the surface."""
    import scanner.scan_queries as q
    import scanner.scan_store as s

    store_all = set(s.__all__)
    queries_all = set(q.__all__)

    assert store_all == {
        "save_result", "mark_alerted", "save_signals", "save_scan_universe",
    }
    assert queries_all == {
        "is_duplicate", "daily_cost", "stats",
        "signal_history", "signal_distribution", "signal_stats",
    }
    # disjoint — no function lives in both halves
    assert store_all.isdisjoint(queries_all)
    # union == the exact pre-split scanner/storage.py public surface
    assert store_all | queries_all == {
        "save_result", "mark_alerted", "save_signals", "save_scan_universe",
        "is_duplicate", "daily_cost", "stats",
        "signal_history", "signal_distribution", "signal_stats",
    }
    # every declared name is actually present + callable in its module
    for name in store_all:
        assert callable(getattr(s, name))
    for name in queries_all:
        assert callable(getattr(q, name))


# --- T-019 Chunk 7b.2a: BoundarySignal spine equivalence ----------------------

_VARYING_COLS = {"id", "signal_id", "batch_id", "created_at", "timestamp", "rowid"}


def test_boundary_signal_v1_adapter_preserves_fields():
    """The Option-C v1 adapter copies every v1 field verbatim; v2-only fields empty."""
    s = _make_signal(coin="ETH", confidence=0.55, key_level=3000.0)
    s.max_leverage = 20
    b = BoundarySignal.from_setup_signal(s)
    for f in ("coin", "setup_type", "direction", "confidence", "key_level",
              "context", "description", "max_leverage", "score_components", "detected_at"):
        assert getattr(b, f) == getattr(s, f), f"adapter dropped/changed {f}"
    assert b.setup_tf == "4h"
    assert b.entry is None and b.hard_stop is None and b.soft_stop is None
    assert b.confluence == frozenset() and b.weakness_flag is False


def test_save_signals_setupsignal_vs_boundary_byte_identical(db_conn):
    """save_signals — the heaviest spine consumer (reads context deeply) — writes an
    identical row for a v1 SetupSignal and its BoundarySignal lift (equivalence gate)."""
    s = _make_signal(coin="SOL", confidence=0.6, key_level=150.0, current_price=152.0,
                     atr=3.0, structure=Structure.BULLISH, range_bounds=(140.0, 160.0))
    b = BoundarySignal.from_setup_signal(s)
    save_signals([s], "batchA", 1, db_conn)
    save_signals([b], "batchB", 1, db_conn)
    rows = db_conn.execute("SELECT * FROM signals ORDER BY batch_id").fetchall()
    assert len(rows) == 2
    ra, rb = dict(rows[0]), dict(rows[1])
    diffs = {k: (ra[k], rb[k]) for k in ra if k not in _VARYING_COLS and ra[k] != rb[k]}
    assert not diffs, f"save_signals diverged for BoundarySignal vs SetupSignal: {diffs}"
