"""HyperLiquid data fetching via ccxt."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ccxt
import pandas as pd

from .config import ScannerConfig

# Module-level exchange instance — reused across all calls.
# load_markets() takes ~11s, so we only want to do it once.
_exchange: ccxt.hyperliquid | None = None


def get_exchange() -> ccxt.hyperliquid:
    """Get or create the shared exchange instance."""
    global _exchange
    if _exchange is None:
        _exchange = ccxt.hyperliquid()
        _exchange.load_markets()
    return _exchange


def fetch_pairs_with_volume(config: ScannerConfig | None = None) -> list[dict]:
    """Fetch all HyperLiquid perps, return sorted by 24h volume descending.

    Returns list of dicts: {coin, symbol, volume_24h, last_price}
    """
    cfg = config or ScannerConfig()
    exchange = get_exchange()
    tickers = exchange.fetch_tickers()

    pairs = []
    for symbol, ticker in tickers.items():
        if ":" not in symbol:
            continue
        coin = ticker["symbol"].split("/")[0]
        # Skip non-standard symbols (index products, combo tickers)
        if "-" in coin or not coin.isalpha():
            continue
        vol = float(ticker.get("quoteVolume") or 0)
        if vol < cfg.min_volume_usd:
            continue
        market = exchange.markets.get(symbol, {})
        max_lev = (market.get("limits", {}).get("leverage", {}) or {}).get("max")
        pairs.append({
            "coin": coin,
            "symbol": symbol,
            "volume_24h": vol,
            "last_price": ticker.get("last"),
            "max_leverage": int(max_lev) if max_lev else None,
        })

    pairs.sort(key=lambda x: x["volume_24h"], reverse=True)
    return pairs[:cfg.max_pairs]


def get_max_leverage(coin: str) -> int | None:
    """Get max leverage for a coin from HyperLiquid market data."""
    exchange = get_exchange()
    symbol = f"{coin}/USDC:USDC"
    market = exchange.markets.get(symbol, {})
    max_lev = (market.get("limits", {}).get("leverage", {}) or {}).get("max")
    return int(max_lev) if max_lev else None


def fetch_candles(
    coin: str,
    interval: str,
    count: int,
    config: ScannerConfig | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV candles for a coin. Returns DataFrame with DatetimeIndex + OHLCV columns.

    Column names are capitalized (Open, High, Low, Close, Volume) for mplfinance compatibility.
    """
    cfg = config or ScannerConfig()
    exchange = get_exchange()
    symbol = f"{coin}/USDC:USDC"

    for attempt in range(3):
        try:
            candles = exchange.fetch_ohlcv(symbol, interval, limit=count)
            break
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(2 ** attempt)  # 1s, 2s backoff
                continue
            raise
    time.sleep(cfg.api_sleep_ms / 1000)

    df = pd.DataFrame(candles, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def fetch_candles_batch(
    coins: list[str],
    interval: str,
    count: int,
    config: ScannerConfig | None = None,
    max_workers: int = 6,
) -> dict[str, pd.DataFrame]:
    """Fetch candles for multiple coins in parallel. Returns {coin: DataFrame}.

    Uses threading since ccxt is I/O-bound. Respects rate limits via max_workers.
    """
    results = {}

    def _fetch_one(coin: str) -> tuple[str, pd.DataFrame | None]:
        try:
            df = fetch_candles(coin, interval, count, config)
            return coin, df
        except Exception:
            return coin, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, coin): coin for coin in coins}
        for future in as_completed(futures):
            coin, df = future.result()
            if df is not None:
                results[coin] = df

    return results
