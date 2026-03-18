"""Scanner configuration — env loading + constants."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

PROJECT_ROOT = Path(__file__).parent.parent
STRATEGY_PATH = PROJECT_ROOT / "strategy" / "unified-ta-ruleset.md"
DATA_DIR = PROJECT_ROOT / "data"
CHARTS_DIR = DATA_DIR / "charts"
DB_PATH = DATA_DIR / "scanner.db"

# Exchange profiles — one entry per supported exchange
EXCHANGE_META: dict[str, dict] = {
    "hyperliquid": {
        "ccxt_class": "hyperliquid",
        "symbol_template": "{coin}/USDC:USDC",
        "fee_rate": 0.00035,
        "label": "HyperLiquid",
    },
    "bybit": {
        "ccxt_class": "bybit",
        "symbol_template": "{coin}/USDT:USDT",
        "fee_rate": 0.00055,
        "label": "Bybit",
    },
    "binance": {
        "ccxt_class": "binanceusdm",
        "symbol_template": "{coin}/USDT:USDT",
        "fee_rate": 0.0004,
        "label": "Binance",
    },
}


def get_exchange_meta(exchange_id: str) -> dict:
    """Get metadata for an exchange. Raises ValueError for unknown exchanges."""
    if exchange_id not in EXCHANGE_META:
        raise ValueError(f"Unknown exchange: '{exchange_id}'. Supported: {', '.join(EXCHANGE_META)}")
    return EXCHANGE_META[exchange_id]


@dataclass
class ScannerConfig:
    # API keys
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    discord_urgent_webhook_url: str = field(default_factory=lambda: os.getenv("DISCORD_URGENT_WEBHOOK_URL", ""))
    discord_reports_webhook_url: str = field(default_factory=lambda: os.getenv("DISCORD_REPORTS_WEBHOOK_URL", ""))
    discord_ops_webhook_url: str = field(default_factory=lambda: os.getenv("DISCORD_OPS_WEBHOOK_URL", ""))

    # Exchange
    exchange: str = "hyperliquid"
    candle_exchanges: list = field(default_factory=lambda: ["hyperliquid", "bybit", "binance"])
    min_volume_usd: float = 2_000_000
    max_pairs: int = 80

    # Candle counts per timeframe
    candle_counts: dict = field(default_factory=lambda: {
        "1w": 52,
        "1d": 120,
        "4h": 200,
        "1h": 120,
        "15m": 200,
    })

    # Timeframes to collect across all candle_exchanges (data pipeline)
    collection_timeframes: list = field(default_factory=lambda: ["15m", "1h", "4h", "1d"])

    # Screener params
    swing_lookback: int = 5
    sr_cluster_pct: float = 0.015
    proximity_atr_mult: float = 1.5
    max_setups_per_coin: int = 3

    # LLM
    model: str = "claude-opus-4-6"
    max_llm_calls_per_scan: int = 40
    daily_cost_cap: float = 30.0

    # Resolution
    timeout_bars: int = 72  # 3 days of 1H candles
    fee_rate: float = 0.00035

    # Storage
    dedup_hours: int = 12

    # Analysis
    analysis_max_workers: int = 4

    # Rate limiting
    api_sleep_ms: int = 100

    def __post_init__(self):
        """Validate exchange identifiers against EXCHANGE_META."""
        if self.exchange not in EXCHANGE_META:
            raise ValueError(
                f"Unknown exchange: '{self.exchange}'. "
                f"Supported: {', '.join(EXCHANGE_META)}"
            )
        for ex in self.candle_exchanges:
            if ex not in EXCHANGE_META:
                raise ValueError(
                    f"Unknown candle exchange: '{ex}'. "
                    f"Supported: {', '.join(EXCHANGE_META)}"
                )
