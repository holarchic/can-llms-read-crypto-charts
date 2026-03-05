"""Execution configuration — dataclass + loader."""

import os
from dataclasses import dataclass

from .secrets import get_secret_or_env


@dataclass
class ExecutionConfig:
    """All execution parameters. Defaults are safe (disabled, testnet)."""

    # Master switch
    enabled: bool = False
    mode: str = "testnet"  # "testnet" | "live"

    # Exchange credentials
    exchange: str = "hyperliquid"
    private_key: str = ""
    wallet_address: str = ""

    # Risk parameters
    risk_pct: float = 1.0
    max_concurrent_positions: int = 100
    max_open_longs: int = 100
    max_portfolio_exposure_pct: float = 100.0

    # Sanity thresholds
    entry_price_tolerance_pct: float = 3.0
    entry_zone_buffer_pct: float = 2.0  # direction-aware buffer beyond entry zone
    max_price_deviation_pct: float = 30.0
    min_rr_ratio: float = 0.5
    max_rr_ratio: float = 10.0
    min_atr_stop_mult: float = 0.5
    max_atr_stop_mult: float = 5.0

    # BTC volatility gate
    btc_vol_gate_enabled: bool = True
    btc_vol_gate_mult: float = 2.0  # block when BTC 4H range > Nx ATR

    # GCP
    gcp_project_id: str = ""

    # Discord
    discord_execution_webhook_url: str = ""

    def __post_init__(self):
        if self.mode not in ("testnet", "live"):
            raise ValueError(f"Invalid execution mode: '{self.mode}'. Must be 'testnet' or 'live'.")
        if self.risk_pct <= 0 or self.risk_pct > 100:
            raise ValueError(f"risk_pct must be in (0, 100], got {self.risk_pct}")


def load_execution_config() -> ExecutionConfig:
    """Build ExecutionConfig from GCP Secret Manager / env vars."""
    gcp_project = os.getenv("GCP_PROJECT_ID", "")

    return ExecutionConfig(
        enabled=os.getenv("EXECUTION_ENABLED", "false").lower() in ("true", "1", "yes"),
        mode=os.getenv("EXECUTION_MODE", "testnet"),
        exchange=os.getenv("EXECUTION_EXCHANGE", "hyperliquid"),
        private_key=get_secret_or_env("HL_PRIVATE_KEY", "HL_PRIVATE_KEY", gcp_project),
        wallet_address=get_secret_or_env("HL_WALLET_ADDRESS", "HL_WALLET_ADDRESS", gcp_project),
        risk_pct=float(os.getenv("EXECUTION_RISK_PCT", "1.0")),
        max_concurrent_positions=int(os.getenv("EXECUTION_MAX_POSITIONS", "100")),
        max_open_longs=int(os.getenv("EXECUTION_MAX_LONGS", "100")),
        max_portfolio_exposure_pct=float(os.getenv("EXECUTION_MAX_EXPOSURE_PCT", "100.0")),
        gcp_project_id=gcp_project,
        discord_execution_webhook_url=os.getenv("DISCORD_EXECUTION_WEBHOOK_URL", ""),
    )
