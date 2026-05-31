"""Scanner configuration — env loading + constants."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

PROJECT_ROOT = Path(__file__).parent.parent
STRATEGY_PATH = PROJECT_ROOT / "strategy" / "unified-ta-ruleset.md"
STRATEGY_PATH_V2 = PROJECT_ROOT / "strategy" / "unified-ta-ruleset-v2.md"
# The authoritative registry of supported SETUP_VERSIONs → their static prompt artifacts
# (persisted prompt_version, ruleset file). The ONE owner of "which versions exist":
# config validation derives its allowed set from these keys (`SETUP_VERSIONS`). config is
# dependency-neutral (it imports nothing from the prompt layer), so the per-version system-
# prompt/scanner-suffix PROSE — which lives in `scanner/llm.py` — cannot live here without a
# config→llm cycle; `prompt_select._PROMPT_PROSE` holds those pairs and asserts at import that
# its key set equals this registry's, so the two cannot drift (a version added here without a
# prose pair fails loudly at load, never at scan time).
SETUP_VERSION_REGISTRY: dict[str, tuple[str, Path]] = {
    "v1": ("9.11", STRATEGY_PATH),
    "v2": ("10.01", STRATEGY_PATH_V2),
}
SETUP_VERSIONS = tuple(SETUP_VERSION_REGISTRY)
DATA_DIR = PROJECT_ROOT / "data"
CHARTS_DIR = DATA_DIR / "charts"
# `LLM_TRADER_DB` overrides the DB path so the T-020.15 replay harness can run against an
# isolated `data/replay.db` (set in-process by the replay/resolve CLIs before scanner import) —
# unset ⇒ the live `scanner.db`, so the bot/deploy/tests are byte-identical (SPEC #53).
DB_PATH = Path(os.getenv("LLM_TRADER_DB", str(DATA_DIR / "scanner.db")))

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

# Post-LLM `setup_validator` (T-019 Chunk 8a, SPEC #33) — the v2-path confluence-COUNT
# floor: a CONFIRM whose detector confluence-factor count is below this is demoted
# before alert AND execute. Placeholder pending Chunk-8b calibration; v2-only, inert
# until the Chunk-9 cutover (the v1 path never runs the validator).
DEFAULT_V2_MIN_CONFLUENCE = 2


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
    # v2-only: capture post-validator CONFIRMs (no orders) for the +R gate (T-020.1, SPEC #41)
    shadow_mode: bool = False

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

    # Post-LLM structural-gate floor (Chunk 8a / SPEC #33) — v2 path only.
    v2_min_confluence: int = DEFAULT_V2_MIN_CONFLUENCE

    def __post_init__(self):
        """Validate the exchange identifier, setup-version flag, and validator floor."""
        if self.exchange not in EXCHANGE_META:
            raise ValueError(
                f"Unknown exchange: '{self.exchange}'. "
                f"Supported: {', '.join(EXCHANGE_META)}"
            )
        if self.setup_version not in SETUP_VERSIONS:
            raise ValueError(
                f"Unknown setup_version: '{self.setup_version}'. Supported: {', '.join(SETUP_VERSIONS)}"
            )
        if self.shadow_mode and self.setup_version != "v2":
            raise ValueError(
                f"shadow_mode requires setup_version='v2' (got '{self.setup_version}') — "
                "shadow capture is the v2-only +R-gate path (SPEC #41)"
            )
        # Floor < 1 would silently disable the gate (every signal has ≥1 confluence
        # factor). 1 is the valid inert setting; 2+ is active. (SPEC #33; arch-pass R1.)
        if self.v2_min_confluence < 1:
            raise ValueError(
                f"v2_min_confluence must be >= 1 (got {self.v2_min_confluence}); "
                "a floor below 1 silently disables the validator."
            )
