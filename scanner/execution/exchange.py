"""Authenticated ccxt exchange connection + position/balance queries."""

import logging

import ccxt

from scanner.config import get_exchange_meta

log = logging.getLogger(__name__)

# Singleton authenticated exchanges — keyed by (mode, wallet_address) so
# different configs (testnet vs live, multi-wallet) get separate instances.
# Thread safety: not locked — relies on flock in cron ensuring one scanner/monitor process.
_auth_exchanges: dict[tuple[str, str], object] = {}


def get_auth_exchange(config):
    """Get or create an authenticated ccxt HyperLiquid connection.

    Testnet mode: sets sandbox. Live mode: mainnet.
    Cached per (mode, wallet_address) key.
    """
    key = (config.mode, config.wallet_address)
    if key in _auth_exchanges:
        return _auth_exchanges[key]

    meta = get_exchange_meta(config.exchange)
    cls = getattr(ccxt, meta["ccxt_class"])

    exch = cls({
        "privateKey": config.private_key,
        "walletAddress": config.wallet_address,
        "timeout": 30000,
    })

    if config.mode == "testnet":
        exch.set_sandbox_mode(True)

    exch.load_markets()
    _auth_exchanges[key] = exch
    return exch


def reset_auth_exchange():
    """Clear all cached exchanges — for testing."""
    _auth_exchanges.clear()


def fetch_current_price(coin: str, config) -> float:
    """Get current price using the auth exchange (sandbox-aware in testnet)."""
    exchange = get_auth_exchange(config)
    meta = get_exchange_meta(config.exchange)
    symbol = meta["symbol_template"].format(coin=coin)
    ticker = exchange.fetch_ticker(symbol)
    return float(ticker["last"])



def fetch_balance(config) -> float:
    """Get USDC balance from authenticated exchange."""
    exchange = get_auth_exchange(config)
    balance = exchange.fetch_balance()
    return float(balance.get("USDC", {}).get("free", 0))


def get_open_positions(config) -> list[dict]:
    """Get open positions from exchange.

    Returns list of {coin, direction, size, entry_price, unrealized_pnl, leverage}.
    """
    exchange = get_auth_exchange(config)
    positions = exchange.fetch_positions()
    result = []
    for pos in positions:
        size = float(pos.get("contracts", 0))
        if size == 0:
            continue
        coin = pos["symbol"].split("/")[0]
        side = pos.get("side", "").upper()
        result.append({
            "coin": coin,
            "direction": "LONG" if side == "LONG" else "SHORT",
            "size": abs(size),
            "entry_price": float(pos.get("entryPrice", 0)),
            "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
            "leverage": int(float(pos.get("leverage", 1))),
        })
    return result


def get_exchange_open_orders(config) -> list[dict]:
    """Get open SL/TP orders on exchange."""
    exchange = get_auth_exchange(config)
    return exchange.fetch_open_orders()


def cancel_order(exchange_order_id: str, symbol: str, config) -> bool:
    """Cancel an order on the exchange. Returns True on success."""
    try:
        exchange = get_auth_exchange(config)
        exchange.cancel_order(exchange_order_id, symbol)
        return True
    except Exception as exc:
        log.warning("Failed to cancel order %s: %s", exchange_order_id, exc)
        return False
