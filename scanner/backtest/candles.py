"""OHLCV candle cache — fetch from ccxt, store/retrieve from SQLite."""

import sqlite3
import time

import pandas as pd

from ..config import ScannerConfig, get_exchange_meta


def get_cached_candles(
    coin: str,
    interval: str,
    since: str,
    until: str,
    conn: sqlite3.Connection,
    exchange: str = "hyperliquid",
) -> pd.DataFrame | None:
    """Retrieve cached candles for a coin/interval within a time range.

    Returns DataFrame with DatetimeIndex + OHLCV columns, or None if no data.
    since/until are ISO-format strings.
    """
    rows = conn.execute(
        """SELECT timestamp, open, high, low, close, volume
        FROM candles
        WHERE coin = ? AND interval = ? AND timestamp >= ? AND timestamp <= ?
        AND exchange = ?
        ORDER BY timestamp""",
        (coin, interval, since, until, exchange),
    ).fetchall()

    if not rows:
        return None

    df = pd.DataFrame(
        [dict(r) for r in rows],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def cache_candles(
    coin: str,
    interval: str,
    df: pd.DataFrame,
    conn: sqlite3.Connection,
    exchange: str = "hyperliquid",
) -> int:
    """Insert candles into cache. Returns number of rows inserted (ignores duplicates)."""
    if df is None or df.empty:
        return 0

    rows = []
    for ts, row in df.iterrows():
        rows.append((
            coin, interval, ts.isoformat(),
            float(row["Open"]), float(row["High"]),
            float(row["Low"]), float(row["Close"]),
            float(row["Volume"]),
            exchange,
        ))

    conn.executemany(
        """INSERT OR IGNORE INTO candles
        (coin, interval, timestamp, open, high, low, close, volume, exchange)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return len(rows)


def _fetch_range(ccxt_exchange, symbol: str, interval: str, since_ms: int, until_ms: int, cfg: ScannerConfig) -> pd.DataFrame | None:
    """Paginated ccxt fetch for a ms range. Returns DataFrame or None."""
    all_raw = []
    cursor = since_ms
    while cursor < until_ms:
        batch = ccxt_exchange.fetch_ohlcv(symbol, interval, since=cursor, limit=500)
        time.sleep(cfg.api_sleep_ms / 1000)
        if not batch:
            break
        all_raw.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= cursor:
            break
        cursor = last_ts + 1

    if not all_raw:
        return None

    df = pd.DataFrame(all_raw, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def _merge_frames(*frames: pd.DataFrame | None) -> pd.DataFrame:
    """Concat non-None frames, dedup, sort."""
    parts = [f for f in frames if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    merged = pd.concat(parts)
    merged = merged[~merged.index.duplicated(keep="last")]
    merged.sort_index(inplace=True)
    return merged


def fetch_and_cache(
    coin: str,
    interval: str,
    since: str,
    until: str,
    conn: sqlite3.Connection,
    config: ScannerConfig | None = None,
    exchange: str = "hyperliquid",
) -> pd.DataFrame:
    """Fetch candles, using cache where available. Fills gaps from ccxt.

    since/until are ISO-format strings. Returns DataFrame with DatetimeIndex + OHLCV.
    """
    from ..data import get_exchange as get_ccxt_exchange

    cached = get_cached_candles(coin, interval, since, until, conn, exchange=exchange)

    cfg = config or ScannerConfig()
    meta = get_exchange_meta(exchange)
    ccxt_exchange = get_ccxt_exchange(exchange)
    symbol = meta["symbol_template"].format(coin=coin)

    since_ms = int(pd.Timestamp(since).timestamp() * 1000)
    until_ms = int(pd.Timestamp(until).timestamp() * 1000)

    if cached is not None and not cached.empty:
        last_cached_ms = int(cached.index[-1].timestamp() * 1000)
        first_cached_ms = int(cached.index[0].timestamp() * 1000)
        if first_cached_ms <= since_ms and last_cached_ms >= until_ms:
            return cached

        # Fetch pre-cache gap if needed
        pre_gap = None
        if first_cached_ms > since_ms:
            pre_gap = _fetch_range(ccxt_exchange, symbol, interval, since_ms, first_cached_ms - 1, cfg)
            if pre_gap is not None:
                pre_gap = pre_gap[pre_gap.index < cached.index[0]]
                cache_candles(coin, interval, pre_gap, conn, exchange=exchange)

        # Fetch post-cache gap
        post_gap = _fetch_range(ccxt_exchange, symbol, interval, last_cached_ms + 1, until_ms, cfg)
        if post_gap is not None:
            post_gap = post_gap[post_gap.index <= pd.Timestamp(until, tz="UTC")]
            cache_candles(coin, interval, post_gap, conn, exchange=exchange)

        return _merge_frames(pre_gap, cached, post_gap)

    # No cache — fetch entire range
    new_df = _fetch_range(ccxt_exchange, symbol, interval, since_ms, until_ms, cfg)
    if new_df is not None:
        new_df = new_df[new_df.index <= pd.Timestamp(until, tz="UTC")]
        cache_candles(coin, interval, new_df, conn, exchange=exchange)
        return new_df

    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
