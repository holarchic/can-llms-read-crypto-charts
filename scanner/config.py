"""Scanner configuration — env loading + constants."""

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / ".env")

PROJECT_ROOT = Path(__file__).parent.parent
STRATEGY_PATH = PROJECT_ROOT / "strategy" / "UNIFIED_TA_RULESET.md"
DATA_DIR = PROJECT_ROOT / "data"
CHARTS_DIR = DATA_DIR / "charts"
DB_PATH = DATA_DIR / "scanner.db"


@dataclass
class ScannerConfig:
    # API keys
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    discord_webhook_url: str = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL", ""))
    discord_reports_webhook_url: str = field(default_factory=lambda: os.getenv("DISCORD_REPORTS_WEBHOOK_URL", ""))

    # HyperLiquid
    min_volume_usd: float = 2_000_000
    max_pairs: int = 80

    # Candle counts per timeframe
    candle_counts: dict = field(default_factory=lambda: {
        "1w": 52,
        "1d": 120,
        "4h": 200,
        "1h": 120,
    })

    # Screener params
    swing_lookback: int = 5
    sr_cluster_pct: float = 0.015
    proximity_atr_mult: float = 1.5
    max_setups_per_coin: int = 3

    # LLM
    model: str = "claude-opus-4-6"
    thinking_budget: int = 5000
    max_llm_calls_per_scan: int = 20
    daily_cost_cap: float = 15.0

    # Storage
    dedup_hours: int = 12

    # Rate limiting
    api_sleep_ms: int = 100
