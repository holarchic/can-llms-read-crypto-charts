"""T-020.17 Binance-DB replay datasource (SPEC #54) — `scanner.shadow.replay_source`.

The as-of clamp + 15m→TF resample fetcher, the as-of notional universe, and `resolve_source`
dispatch, against a tiny purpose-built Binance-shaped sqlite fixture (no live network)."""

import sqlite3
from datetime import datetime, timezone

import pandas as pd
import pytest

from scanner.shadow.replay_source import (
    DEFAULT_MIN_NOTIONAL_USD,
    LIVE_HL,
    binance_db_max_ts,
    binance_source,
    make_binance_fetcher,
    make_binance_universe,
    resolve_source,
)

_T0 = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)


def _make_db(path, series: dict):
    """Build a Binance-shaped 15m DB. `series` = {coin: [(open_ts, o, h, l, c, v), ...]}."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE candles (coin TEXT, interval TEXT, timestamp TEXT, open REAL, high REAL, "
        "low REAL, close REAL, volume REAL, exchange TEXT, PRIMARY KEY (coin, interval, timestamp))"
    )
    rows = [
        (coin, "15m", ts.isoformat(), o, h, l, c, v, "binance")
        for coin, bars in series.items()
        for (ts, o, h, l, c, v) in bars
    ]
    conn.executemany("INSERT INTO candles VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def _ramp(coin_close: float, n: int, vol: float = 10.0, start=_T0):
    """n consecutive 15m bars from `start`, flat OHLC at `coin_close`, constant volume."""
    return [
        (start + pd.Timedelta(minutes=15 * i), coin_close, coin_close, coin_close, coin_close, vol)
        for i in range(n)
    ]


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "binance-backtest.db"
    # BTC: 200 15m bars (≈ 50h) flat at 100.0, vol 10 ⇒ notional/bar = 1000.
    _make_db(path, {"BTC": _ramp(100.0, 200, vol=10.0)})
    return path


# ── fetcher: resample + as_of clamp ───────────────────────────────────────────────────────────

def test_fetcher_15m_passthrough_clamps_as_of(db):
    fetch = make_binance_fetcher(db)
    # as_of at the 10th bar's open ⇒ bars 0..10 are <= as_of (11 bars); tail(5) = bars 6..10.
    as_of = _T0 + pd.Timedelta(minutes=15 * 10)
    df = fetch("BTC", "15m", 5, as_of=as_of)
    assert len(df) == 5
    assert df.index[-1] == as_of                       # the bar CONTAINING as_of is present (forming)
    assert (df.index <= as_of).all()                   # nothing past T leaks
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_fetcher_resamples_15m_to_1h_open_time_left(db):
    fetch = make_binance_fetcher(db)
    # as_of at 02:00 sharp ⇒ 1h buckets 00:00, 01:00 complete (4 sub-bars each) + 02:00 partial (1).
    as_of = _T0 + pd.Timedelta(hours=2)
    df = fetch("BTC", "1h", 10, as_of=as_of)
    assert df.index[-1] == as_of                       # the partial 02:00 bucket is returned (forming)
    assert df.index[-2] == _T0 + pd.Timedelta(hours=1)
    assert (df.index.minute == 0).all()                # open-time, left-labeled hourly


def test_fetcher_forming_bucket_present_for_drop(db):
    """The partial last bucket containing as_of must be RETURNED (fetch_multi_tf drops it) — NOT
    dropped here, which would double-drop the most-recent closed bar."""
    fetch = make_binance_fetcher(db)
    as_of = _T0 + pd.Timedelta(hours=1, minutes=30)    # mid-bucket ⇒ 01:00 bucket has 2 of 4 sub-bars
    df = fetch("BTC", "1h", 10, as_of=as_of)
    assert df.index[-1] == _T0 + pd.Timedelta(hours=1)  # the partial 01:00 bucket survives


def test_fetcher_forward_as_of_none_returns_most_recent(db):
    """The resolver's forward fetch (as_of=None) returns the most-recent `count` bars up to DB-max."""
    fetch = make_binance_fetcher(db)
    df = fetch("BTC", "1h", 3, as_of=None)
    assert len(df) == 3
    assert df.index[-1] == binance_db_max_ts(db) - pd.Timedelta(minutes=45)  # last full 1h open


def test_fetcher_empty_raises(db):
    fetch = make_binance_fetcher(db)
    with pytest.raises(ValueError, match="empty"):
        fetch("BTC", "15m", 5, as_of=_T0 - pd.Timedelta(days=1))   # before any history
    with pytest.raises(ValueError, match="empty"):
        fetch("NOTLISTED", "15m", 5, as_of=_T0 + pd.Timedelta(hours=5))


def test_fetcher_malformed_raises(tmp_path):
    path = tmp_path / "bad.db"
    _make_db(path, {"BAD": [(_T0, 0.0, 0.0, 0.0, 0.0, 5.0)] + _ramp(50.0, 5, start=_T0 + pd.Timedelta(minutes=15))})
    fetch = make_binance_fetcher(path)
    with pytest.raises(ValueError, match="malformed"):
        fetch("BAD", "15m", 10, as_of=_T0 + pd.Timedelta(hours=1))


def test_fetcher_accepts_resolver_call_shape(db):
    """The resolver calls `fetch(coin, '1h', bars, exchange_id=...)` — positional + exchange_id kw."""
    fetch = make_binance_fetcher(db)
    df = fetch("BTC", "1h", 3, exchange_id="hyperliquid")
    assert len(df) == 3


# ── universe: as-of 24h notional floor ─────────────────────────────────────────────────────────

def test_universe_notional_floor(tmp_path):
    path = tmp_path / "u.db"
    # 96 bars = 24h. BIG: close 1000 vol 100 ⇒ 96*100000 = 9.6M/bar-window... per bar 100000, ×96=9.6M.
    # SMALL: close 1 vol 1 ⇒ 96*1 = 96 total. Floor at 1M ⇒ only BIG survives.
    _make_db(path, {
        "BIG": _ramp(1000.0, 96, vol=100.0),
        "SMALL": _ramp(1.0, 96, vol=1.0),
    })
    universe = make_binance_universe(path, min_notional=1_000_000.0)
    as_of = _T0 + pd.Timedelta(hours=24)               # includes all 96 bars (open-time <= as_of)
    pairs = universe(as_of)
    coins = [p["coin"] for p in pairs]
    assert coins == ["BIG"]                              # SMALL below floor; sorted desc
    assert pairs[0]["max_leverage"] is None              # #53 metadata neglect
    assert pairs[0]["volume_24h"] > 1_000_000.0


def test_universe_caps_at_max_pairs(tmp_path):
    """The Binance universe honours `config.max_pairs` so a replay anchor scans the same-size
    top-notional cohort as live (`fetch_pairs_with_volume` returns `pairs[:max_pairs]`)."""
    from scanner.config import ScannerConfig
    path = tmp_path / "cap.db"
    # 5 coins all clearing a $1 floor, descending notional by close price.
    _make_db(path, {f"C{k}": _ramp(float(100 - k), 96, vol=100.0) for k in range(5)})
    universe = make_binance_universe(path, min_notional=1.0)
    as_of = _T0 + pd.Timedelta(hours=24)
    full = universe(as_of, config=ScannerConfig(max_pairs=99))
    capped = universe(as_of, config=ScannerConfig(max_pairs=2))
    assert len(full) == 5 and len(capped) == 2
    assert [p["coin"] for p in capped] == ["C0", "C1"]   # top-2 by notional (C0 highest close)
    assert len(universe(as_of, config=None)) == 5         # None ⇒ uncapped


def test_universe_as_of_none_uses_db_max(tmp_path):
    path = tmp_path / "u2.db"
    _make_db(path, {"BIG": _ramp(1000.0, 96, vol=100.0)})
    universe = make_binance_universe(path, min_notional=1_000_000.0)
    assert [p["coin"] for p in universe(None)] == ["BIG"]


# ── max-ts + dispatch ──────────────────────────────────────────────────────────────────────────

def test_db_max_ts(db):
    assert binance_db_max_ts(db) == _T0 + pd.Timedelta(minutes=15 * 199)


def test_resolve_source_dispatch(db):
    assert resolve_source("live-hl") is LIVE_HL
    src = resolve_source("binance-db", db_path=db)
    assert src.name == "binance-db"
    assert callable(src.fetch) and callable(src.universe) and callable(src.now)
    assert src.now() == binance_db_max_ts(db)
    with pytest.raises(ValueError, match="unknown replay source"):
        resolve_source("bogus")


def test_binance_source_missing_db_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        binance_source(tmp_path / "does-not-exist.db")


def test_binance_source_empty_db_fails_fast(tmp_path):
    path = tmp_path / "empty.db"
    _make_db(path, {})                                   # schema but zero rows
    with pytest.raises(ValueError, match="no 15m candles"):
        binance_source(path)


def test_live_hl_universe_ignores_as_of(monkeypatch):
    """LIVE_HL.universe delegates to fetch_pairs_with_volume(config), ignoring as_of (#53)."""
    import scanner.shadow.replay_source as rs
    seen = {}

    def _fake_pairs(config):
        seen["called"] = config
        return [{"coin": "X"}]

    monkeypatch.setattr(rs, "fetch_pairs_with_volume", _fake_pairs)
    # LIVE_HL captured the original ref at import; rebuild the lambda's target via a fresh source.
    src = rs.ReplaySource("live-hl", rs.fetch_candles, lambda as_of, config=None: rs.fetch_pairs_with_volume(config), LIVE_HL.now)
    assert src.universe(as_of=_T0, config="CFG") == [{"coin": "X"}]
    assert seen["called"] == "CFG"
    assert DEFAULT_MIN_NOTIONAL_USD == 40_000_000.0
