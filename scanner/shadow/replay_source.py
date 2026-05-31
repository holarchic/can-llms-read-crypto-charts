"""Replay datasources for the historical-replay shadow harness (SPEC #53/#54).

A ``ReplaySource`` is the cohesive triple a ``--source`` flag selects: a ``fetch_candles``-shaped
candle fetcher, an as-of universe resolver, and the replay "now". Two instances:

- ``LIVE_HL`` — the original .15/.16 source: live-HyperLiquid ``fetch_candles`` + the current
  volume-ranked ``fetch_pairs_with_volume`` universe + wall-clock now. ``--source live-hl`` (the
  capture default) reproduces .16 byte-identically (``as_of`` is ignored by the live universe — it
  has no historical mode, the SPEC #53 metadata neglect).
- the Binance-DB source (``binance_source``) — backed by the local Binance 15m archive
  (``data/binance-backtest.db``): resample 15m → any TF clamped to ``as_of``, an
  as-of 24h ``volume*close`` notional universe ($40M floor), and ``MAX(timestamp)`` as the replay
  now (so anchors land inside the static archive, not the later wall-clock). An HL liquidity
  *proxy* for the $40M-floor survivors (#54) — the resolver still nets HL taker fees.

The pipeline threads a ``ReplaySource | None`` (None ⇒ live); leaf data-plane fns take only the raw
``fetch`` callable, so this abstraction lives at the orchestration layer alone.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from scanner.data import fetch_candles, fetch_pairs_with_volume

__all__ = [
    "DEFAULT_MIN_NOTIONAL_USD",
    "DEFAULT_BINANCE_DB",
    "ReplaySource",
    "LIVE_HL",
    "binance_source",
    "resolve_source",
    "binance_db_max_ts",
    "make_binance_fetcher",
]

# The 24h-notional floor for the Binance universe — higher than HL's $2M to account for Binance's
# deeper book, so the survivors are fair HL proxies (#54). Operator-overridable via the capture CLI.
DEFAULT_MIN_NOTIONAL_USD = 40_000_000.0

# The Binance archive lives in the SIBLING candle archive project. Script-anchored (never
# CWD-relative — operators run from arbitrary dirs); the CLIs default --db here.
DEFAULT_BINANCE_DB = Path(__file__).resolve().parents[2].parent / "candle archive" / "data" / "binance-backtest.db"

# 15m bars per one bar of the target ccxt interval — bounds the as-of query LIMIT so a resample has
# enough constituents without scanning a coin's whole history. ``1M`` uses 31d (longest month).
_BARS_PER_15M: dict[str, int] = {"15m": 1, "1h": 4, "4h": 16, "1d": 96, "1w": 672, "1M": 2976}

# Target ccxt interval → pandas resample rule (open-time, left-labeled). ``15m`` is passthrough
# (absent). ``W-MON`` = Monday-anchored weekly (Binance/HL weekly open); ``MS`` = month-start. All
# verified open-time-left-labeled on real BTC.
_RESAMPLE_RULE: dict[str, str] = {"1h": "1h", "4h": "4h", "1d": "1D", "1w": "W-MON", "1M": "MS"}

# Overshoot the query a few buckets so edge/gap slop clears before the .tail(count) trim.
_QUERY_PAD_BARS = 8


@dataclass(frozen=True)
class ReplaySource:
    """The (fetch, universe, now) triple a ``--source`` flag selects. ``fetch`` is
    ``fetch_candles``-shaped; ``universe(as_of, config)`` returns pair-dicts (≥ ``{"coin": …}``);
    ``now()`` is the replay clock (DB-max for the static archive, wall-clock for live)."""

    name: str
    fetch: Callable
    universe: Callable
    now: Callable[[], datetime]


def _iso(dt: datetime) -> str:
    """Normalize to a UTC ISO-8601 string matching the DB's stored ``timestamp`` text (``+00:00``),
    so lexicographic ``<=`` comparison on the TEXT column is correct."""
    ts = pd.Timestamp(dt)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.isoformat()


def _query(db_path: Path, sql: str, params: list) -> list:
    """One read-only query against the Binance archive. Per-call open/close — thread-safe (the
    stage-2 recovery runs in a ThreadPool; sqlite conns aren't shareable across threads) and cheap
    for a read-only open (header-only, no DB load)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def binance_db_max_ts(db_path: Path) -> datetime:
    """``MAX(timestamp)`` of the 15m table → tz-aware datetime — the replay "now" so anchors land
    inside the static archive. Raises on an empty/wrong DB (fail-fast before any batch)."""
    rows = _query(db_path, "SELECT MAX(timestamp) FROM candles WHERE interval = '15m'", [])
    mx = rows[0][0] if rows else None
    if mx is None:
        raise ValueError(f"{db_path}: no 15m candles (empty or wrong DB?)")
    return pd.Timestamp(mx).tz_convert("UTC").to_pydatetime()


def _frame_from_rows(rows: list) -> pd.DataFrame:
    """DESC sqlite rows → ASC OHLCV DataFrame with a UTC DatetimeIndex (``fetch_candles`` shape)."""
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if not rows:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="timestamp", tz="UTC"))
    df = pd.DataFrame(rows[::-1], columns=["timestamp", *cols])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """OHLCV-correct 15m → ``rule`` downsample (open-time, left-labeled). NO short-last-bucket drop
    — the caller's ``fetch_multi_tf._drop_forming_bar`` drops the forming-at-T bucket; dropping it
    here too would double-drop a genuinely-closed bar. Empty (all-NaN) buckets from a gap are
    dropped."""
    agg = df.resample(rule, label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return agg.dropna(how="all", subset=["Open", "High", "Low", "Close"])


def _validated(df: pd.DataFrame, coin: str, interval: str) -> pd.DataFrame:
    """Same ingestion-boundary contract as ``data.fetch_candles``: empty / non-finite /
    non-positive O/H/L/C raises, so a coin without history at T fails loud → per-coin skip, never a
    fabricated bar (#53/#54)."""
    if df.empty:
        raise ValueError(f"binance fetch({coin}, {interval}): empty (no 15m history at/before as_of)")
    prices = df[["Open", "High", "Low", "Close"]].to_numpy()
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError(f"binance fetch({coin}, {interval}): malformed (non-finite or non-positive price)")
    return df


def make_binance_fetcher(db_path: Path) -> Callable:
    """A ``fetch_candles``-signature closure backed by the Binance 15m DB.

    Per call: pull the most-recent ``(count+pad)*bars_per`` 15m rows with open-time ``≤ as_of``
    (no upper bound when ``as_of`` is None — the resolver's forward fetch), resample to ``interval``,
    and ``.tail(count)`` INCLUDING the partial bucket containing T (so ``fetch_multi_tf`` drops the
    forming-at-T bar). ``config``/``exchange_id`` are accepted + ignored (single-exchange DB; the
    resolver passes ``exchange_id`` with no ``config``). Validates like ``fetch_candles``."""

    def fetch(coin, interval, count, config=None, exchange_id=None, as_of=None):
        sql = "SELECT timestamp, open, high, low, close, volume FROM candles WHERE coin = ? AND interval = '15m'"
        params: list = [coin]
        if as_of is not None:
            sql += " AND timestamp <= ?"
            params.append(_iso(as_of))
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append((count + _QUERY_PAD_BARS) * _BARS_PER_15M[interval])

        df = _frame_from_rows(_query(db_path, sql, params))
        if interval != "15m" and not df.empty:
            df = _resample(df, _RESAMPLE_RULE[interval])
        return _validated(df.tail(count), coin, interval)

    return fetch


def make_binance_universe(db_path: Path, min_notional: float) -> Callable:
    """An as-of universe resolver: coins whose 24h rolling notional (``Σ volume*close`` over the 96
    15m bars in ``(as_of-24h, as_of]``) clears ``min_notional``, newest-anchor-first. Returns the
    ``fetch_pairs_with_volume`` dict shape stage1 + ``save_scan_universe`` read; ``max_leverage`` is
    None (the #53 metadata neglect — no live HL lookup)."""

    def universe(as_of, config=None) -> list[dict]:
        end = as_of if as_of is not None else binance_db_max_ts(db_path)
        end_ts = pd.Timestamp(end)
        end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
        start_ts = end_ts - pd.Timedelta(hours=24)
        rows = _query(
            db_path,
            "SELECT coin, SUM(volume * close) AS notional FROM candles "
            "WHERE interval = '15m' AND timestamp <= ? AND timestamp > ? "
            "GROUP BY coin HAVING notional > ? ORDER BY notional DESC",
            [end_ts.isoformat(), start_ts.isoformat(), min_notional],
        )
        pairs = [{"coin": coin, "volume_24h": notional, "max_leverage": None} for coin, notional in rows]
        # Honour the live universe cap (`fetch_pairs_with_volume` returns `pairs[:max_pairs]`) so a
        # replay anchor scans the SAME-size top-notional cohort as live, not the full $40M survivor set.
        cap = config.max_pairs if config is not None else None
        return pairs[:cap] if cap is not None else pairs

    return universe


LIVE_HL = ReplaySource(
    name="live-hl",
    fetch=fetch_candles,
    universe=lambda as_of, config=None: fetch_pairs_with_volume(config),
    now=lambda: datetime.now(timezone.utc),
)


def binance_source(db_path, min_notional: float = DEFAULT_MIN_NOTIONAL_USD) -> ReplaySource:
    """Build the Binance-DB replay source. Fail-fast: a missing file raises before any LLM batch
    (a bare ``sqlite3.connect`` would silently create an empty DB — LESSONS-PYTHON), and
    ``binance_db_max_ts`` is probed up front so an empty/wrong table errors here too."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Binance replay DB not found: {db_path} (pass --db; default is the sibling "
            "candle archive archive)"
        )
    binance_db_max_ts(db_path)  # probe: empty / unreadable 15m table fails here, before the batch
    return ReplaySource(
        name="binance-db",
        fetch=make_binance_fetcher(db_path),
        universe=make_binance_universe(db_path, min_notional),
        now=lambda: binance_db_max_ts(db_path),
    )


def resolve_source(name: str, db_path=None, min_notional: float = DEFAULT_MIN_NOTIONAL_USD) -> ReplaySource:
    """Map a ``--source`` flag to a ``ReplaySource`` (the CLI seam). ``binance-db`` defaults
    ``db_path`` to the sibling archive."""
    if name == "live-hl":
        return LIVE_HL
    if name == "binance-db":
        return binance_source(db_path or DEFAULT_BINANCE_DB, min_notional)
    raise ValueError(f"unknown replay source {name!r} (live-hl | binance-db)")
