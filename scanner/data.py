"""Exchange data fetching via ccxt — supports HyperLiquid, Bybit, Binance."""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ccxt
import pandas as pd

from .config import ScannerConfig, get_exchange_meta

log = logging.getLogger(__name__)

# Module-level exchange instances — reused across all calls.
# load_markets() takes ~11s, so we only want to do it once per exchange.
_exchanges: dict[str, object] = {}
_lock = threading.Lock()
_exchange_locks: dict[str, threading.Lock] = {}


_TRANSIENT_CCXT = (
    ccxt.RateLimitExceeded,
    ccxt.DDoSProtection,
    ccxt.ExchangeNotAvailable,
    ccxt.NetworkError,
    ccxt.RequestTimeout,
)


def _is_transient_ccxt(exc: Exception) -> bool:
    """Broad retry predicate. Catches ccxt-class errors + vendor strings.

    ccxt often wraps HL rate-limits in a generic ExchangeError with
    "Too many cumulative requests" in str(exc); string checks catch HTTP
    4xx/5xx codes that don't always map cleanly to ccxt exception classes.
    """
    if isinstance(exc, _TRANSIENT_CCXT):
        return True
    msg = str(exc)
    return any(s in msg for s in ("429", "503", "504", "Too many cumulative"))


def get_exchange(exchange_id: str = "hyperliquid") -> object:
    """Get or create a shared exchange instance for the given exchange.

    Thread-safe: uses double-checked locking so load_markets() runs once.
    Retries transient errors on load_markets() so a single HL blip doesn't
    crash the whole scan cycle.
    """
    if exchange_id in _exchanges:
        return _exchanges[exchange_id]
    with _lock:
        if exchange_id not in _exchanges:
            meta = get_exchange_meta(exchange_id)
            cls = getattr(ccxt, meta["ccxt_class"])
            ex = cls({"timeout": 30000})
            for attempt in range(4):
                try:
                    ex.load_markets()
                    break
                except Exception as exc:
                    if attempt >= 3 or not _is_transient_ccxt(exc):
                        raise
                    sleep_s = 2 ** attempt
                    log.warning(
                        "load_markets(%s) attempt %d/4 failed: %s — retrying in %ds",
                        exchange_id, attempt + 1, exc, sleep_s,
                    )
                    time.sleep(sleep_s)
            _exchange_locks[exchange_id] = threading.Lock()
            _exchanges[exchange_id] = ex
    return _exchanges[exchange_id]


_testnet_coins_cache: dict[str, set[str]] = {}
_testnet_lock = threading.Lock()


def get_testnet_coins(exchange_id: str = "hyperliquid") -> set[str]:
    """Load testnet markets and return available coin names.

    Cached per-exchange per-process — load_markets() takes ~11s and testnet
    market listings change infrequently, so re-fetching every scan just adds
    latency without new information.
    """
    if exchange_id in _testnet_coins_cache:
        return _testnet_coins_cache[exchange_id]
    with _testnet_lock:
        if exchange_id in _testnet_coins_cache:
            return _testnet_coins_cache[exchange_id]
        meta = get_exchange_meta(exchange_id)
        cls = getattr(ccxt, meta["ccxt_class"])
        ex = cls({"timeout": 30000})
        ex.set_sandbox_mode(True)
        ex.load_markets()
        coins = set()
        for symbol in ex.markets:
            if ":" not in symbol:
                continue
            coin = symbol.split("/")[0]
            if coin.isalpha():
                coins.add(coin)
        _testnet_coins_cache[exchange_id] = coins
        return coins


def fetch_pairs_with_volume(config: ScannerConfig | None = None) -> list[dict]:
    """Fetch all perps for the configured exchange, return sorted by 24h volume descending.

    Returns list of dicts: {coin, symbol, volume_24h, last_price, max_leverage}
    """
    cfg = config or ScannerConfig()
    exchange = get_exchange(cfg.exchange)

    tickers = None
    for attempt in range(4):
        try:
            tickers = exchange.fetch_tickers()
            break
        except Exception as exc:
            if attempt >= 3 or not _is_transient_ccxt(exc):
                raise
            sleep_s = 2 ** attempt
            log.warning(
                "fetch_tickers attempt %d/4 failed: %s — retrying in %ds",
                attempt + 1, exc, sleep_s,
            )
            time.sleep(sleep_s)

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


def get_max_leverage(coin: str, exchange_id: str = "hyperliquid") -> int | None:
    """Get max leverage for a coin from exchange market data."""
    meta = get_exchange_meta(exchange_id)
    exchange = get_exchange(exchange_id)
    symbol = meta["symbol_template"].format(coin=coin)
    market = exchange.markets.get(symbol, {})
    max_lev = (market.get("limits", {}).get("leverage", {}) or {}).get("max")
    return int(max_lev) if max_lev else None


def fetch_candles(
    coin: str,
    interval: str,
    count: int,
    config: ScannerConfig | None = None,
    exchange_id: str | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV candles for a coin. Returns DataFrame with DatetimeIndex + OHLCV columns.

    Column names are capitalized (Open, High, Low, Close, Volume) for mplfinance compatibility.
    exchange_id overrides config.exchange when fetching from a non-primary exchange.
    """
    cfg = config or ScannerConfig()
    eid = exchange_id or cfg.exchange
    meta = get_exchange_meta(eid)
    exchange = get_exchange(eid)
    symbol = meta["symbol_template"].format(coin=coin)

    ex_lock = _exchange_locks.get(eid)
    for attempt in range(5):
        try:
            with ex_lock:
                candles = exchange.fetch_ohlcv(symbol, interval, limit=count)
            break
        except Exception as e:
            if attempt < 4 and _is_transient_ccxt(e):
                time.sleep(2 ** (attempt + 1))  # 2s, 4s, 8s, 16s backoff
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
    max_workers: int = 3,
    exchange_id: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch candles for multiple coins in parallel. Returns {coin: DataFrame}.

    Uses threading since ccxt is I/O-bound. Respects rate limits via max_workers.
    exchange_id overrides config.exchange when fetching from a non-primary exchange.
    """
    results = {}

    def _fetch_one(coin: str) -> tuple[str, pd.DataFrame | None]:
        try:
            df = fetch_candles(coin, interval, count, config, exchange_id=exchange_id)
            return coin, df
        except Exception as exc:
            log.warning("fetch_candles failed for %s: %s", coin, exc)
            return coin, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, coin): coin for coin in coins}
        for future in as_completed(futures):
            coin, df = future.result()
            if df is not None:
                results[coin] = df

    return results
