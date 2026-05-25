"""Scanner configuration — env loading + constants."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

PROJECT_ROOT = Path(__file__).parent.parent
STRATEGY_PATH = PROJECT_ROOT / "strategy" / "unified-ta-ruleset.md"
STRATEGY_PATH_V2 = PROJECT_ROOT / "strategy" / "unified-ta-ruleset-v2.md"
# Canonical SETUP_VERSION set — the one source of the supported versions. Validated at
# the config boundary here; mapped to per-version prompt artifacts in `prompt_select`
# (which asserts its registry keys match this set, so the two cannot drift).
SETUP_VERSIONS = ("v1", "v2")
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


# Screener-param defaults — single source (T-016 Phase 3, SPEC #26 a).
# The live runtime config (ScannerConfig) owns these; strategy.Strategy
# mirrors them by reference, not by re-declaring literals. Reversible if the
# Phase-5 scope call makes an active Strategy the runtime source-of-truth.
DEFAULT_SWING_LOOKBACK = 5
DEFAULT_SR_CLUSTER_PCT = 0.015
DEFAULT_PROXIMITY_ATR_MULT = 1.5
DEFAULT_MAX_SETUPS_PER_COIN = 3


@dataclass
class ScannerConfig:
    # API keys
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    discord_urgent_webhook_url: str = field(default_factory=lambda: os.getenv("DISCORD_URGENT_WEBHOOK_URL", ""))
    discord_reports_webhook_url: str = field(default_factory=lambda: os.getenv("DISCORD_REPORTS_WEBHOOK_URL", ""))
    discord_ops_webhook_url: str = field(default_factory=lambda: os.getenv("DISCORD_OPS_WEBHOOK_URL", ""))
    discord_errors_webhook_url: str = field(default_factory=lambda: os.getenv("DISCORD_ERRORS_WEBHOOK_URL", ""))

    # Exchange
    exchange: str = "hyperliquid"
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

    # Screener params
    swing_lookback: int = DEFAULT_SWING_LOOKBACK
    sr_cluster_pct: float = DEFAULT_SR_CLUSTER_PCT
    proximity_atr_mult: float = DEFAULT_PROXIMITY_ATR_MULT
    max_setups_per_coin: int = DEFAULT_MAX_SETUPS_PER_COIN

    # LLM
    model: str = "claude-opus-4-7"
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

    # Setup detection version — "v1" (A–G screener, default) or "v2" (v10
    # framework). The cutover flag for T-019 / Decision #32; default v1 keeps the
    # live path byte-identical until the v2 dispatch (Chunk 7b) lands.
    setup_version: str = field(default_factory=lambda: os.getenv("SETUP_VERSION", "v1"))

    def __post_init__(self):
        """Validate the exchange identifier and the setup-version flag."""
        if self.exchange not in EXCHANGE_META:
            raise ValueError(
                f"Unknown exchange: '{self.exchange}'. "
                f"Supported: {', '.join(EXCHANGE_META)}"
            )
        if self.setup_version not in SETUP_VERSIONS:
            raise ValueError(
                f"Unknown setup_version: '{self.setup_version}'. Supported: {', '.join(SETUP_VERSIONS)}"
            )
