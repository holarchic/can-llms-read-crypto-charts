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

    # Determine what we need to fetch
    cfg = config or ScannerConfig()
    meta = get_exchange_meta(exchange)
    ccxt_exchange = get_ccxt_exchange(exchange)
    symbol = meta["symbol_template"].format(coin=coin)

    since_ms = int(pd.Timestamp(since).timestamp() * 1000)
    until_ms = int(pd.Timestamp(until).timestamp() * 1000)

    # If we have cached data, only fetch from after the last cached candle
    if cached is not None and not cached.empty:
        last_cached_ms = int(cached.index[-1].timestamp() * 1000)
        # If cache covers the full range, return it
        first_cached_ms = int(cached.index[0].timestamp() * 1000)
        if first_cached_ms <= since_ms and last_cached_ms >= until_ms:
            return cached
        # Fetch from after last cached candle
        fetch_since = last_cached_ms + 1
    else:
        fetch_since = since_ms
        cached = None

    # Paginate — ccxt returns max 500 candles per call
    all_new = []
    cursor = fetch_since
    while cursor < until_ms:
        batch = ccxt_exchange.fetch_ohlcv(symbol, interval, since=cursor, limit=500)
        time.sleep(cfg.api_sleep_ms / 1000)
        if not batch:
            break
        all_new.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= cursor:
            break
        cursor = last_ts + 1

    if all_new:
        new_df = pd.DataFrame(all_new, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms", utc=True)
        new_df.set_index("timestamp", inplace=True)
        # Filter to requested range
        new_df = new_df[new_df.index <= pd.Timestamp(until, tz="UTC")]
        cache_candles(coin, interval, new_df, conn, exchange=exchange)
    else:
        new_df = None

    # Merge cached + new
    if cached is not None and new_df is not None:
        merged = pd.concat([cached, new_df])
        merged = merged[~merged.index.duplicated(keep="last")]
        merged.sort_index(inplace=True)
        return merged
    elif cached is not None:
        return cached
    elif new_df is not None:
        return new_df
    else:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
