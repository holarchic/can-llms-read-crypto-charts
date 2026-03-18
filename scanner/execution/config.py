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

    # Capital bounds
    max_notional_usd: float = 0.0  # 0 = disabled; max notional per trade

    # Sanity thresholds
    entry_price_tolerance_pct: float = 5.0
    entry_zone_buffer_pct: float = 2.0  # direction-aware buffer beyond entry zone
    min_rr_ratio: float = 0.5
    max_rr_ratio: float = 10.0
    min_atr_stop_mult: float = 0.5
    max_atr_stop_mult: float = 5.0

    # BTC volatility gate
    btc_vol_gate_enabled: bool = True
    btc_vol_gate_mult: float = 2.0  # block when BTC 4H range > Nx ATR

    # GCP
    gcp_project_id: str = ""

    # Position timeout
    position_timeout_hours: int = 72  # 0 = disabled

    # Discord
    discord_execution_webhook_url: str = ""

    def __post_init__(self):
        if self.mode not in ("testnet", "live"):
            raise ValueError(f"Invalid execution mode: '{self.mode}'. Must be 'testnet' or 'live'.")
        if self.risk_pct <= 0 or self.risk_pct > 100:
            raise ValueError(f"risk_pct must be in (0, 100], got {self.risk_pct}")
        if self.enabled and not self.private_key:
            raise ValueError("enabled=True requires private_key")
        if self.enabled and not self.wallet_address:
            raise ValueError("enabled=True requires wallet_address")
        if self.position_timeout_hours < 0:
            raise ValueError(f"position_timeout_hours must be >= 0, got {self.position_timeout_hours}")


def load_execution_config() -> ExecutionConfig:
    """Build ExecutionConfig from GCP Secret Manager / env vars."""
    gcp_project = os.getenv("GCP_PROJECT_ID", "")
    is_live = os.getenv("EXECUTION_MODE", "testnet") == "live"
    require_sm = is_live and bool(gcp_project)

    return ExecutionConfig(
        enabled=os.getenv("EXECUTION_ENABLED", "false").lower() in ("true", "1", "yes"),
        mode=os.getenv("EXECUTION_MODE", "testnet"),
        exchange=os.getenv("EXECUTION_EXCHANGE", "hyperliquid"),
        private_key=get_secret_or_env(
            "LT_HL_PRIVATE_KEY", "HL_PRIVATE_KEY", gcp_project, require=require_sm,
        ),
        wallet_address=get_secret_or_env(
            "LT_HL_WALLET_ADDRESS", "HL_WALLET_ADDRESS", gcp_project, require=require_sm,
        ),
        risk_pct=float(os.getenv("EXECUTION_RISK_PCT", "1.0")),
        max_notional_usd=float(os.getenv("EXECUTION_MAX_NOTIONAL_USD", "0.0")),
        max_concurrent_positions=int(os.getenv("EXECUTION_MAX_POSITIONS", "100")),
        max_open_longs=int(os.getenv("EXECUTION_MAX_LONGS", "100")),
        max_portfolio_exposure_pct=float(os.getenv("EXECUTION_MAX_EXPOSURE_PCT", "100.0")),
        gcp_project_id=gcp_project,
        position_timeout_hours=int(os.getenv("EXECUTION_POSITION_TIMEOUT_HOURS", "72")),
        discord_execution_webhook_url=os.getenv("DISCORD_EXECUTION_WEBHOOK_URL", ""),
    )
