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

    # Order limits
    max_open_orders: int = 200  # 0 = disabled; env: EXECUTION_MAX_OPEN_ORDERS

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

    # Drift invalidation (hybrid_inv strategy on 4H candles)
    drift_enabled: bool = False
    drift_grace_hours: float = 12.0           # hours after entry before any trigger fires
    drift_consec_threshold: int = 6           # consecutive 4H closes in drift zone for stagnation
    drift_momentum_check_hours: float = 24.0  # hours after entry for no_momentum one-shot check
    drift_momentum_pct: float = 40.0          # min % progress toward TP1 at momentum check
    drift_reversal_best_pct: float = 50.0     # best progress threshold for reversal trigger
    drift_reversal_current_pct: float = 15.0  # current progress threshold for reversal trigger

    # Conditional entry (monitor 1H candle closes for trigger/invalidation)
    conditional_entry_enabled: bool = False
    conditional_entry_max_expiry: int = 24    # max 1H candles before setup expires

    # Counter-trend adjustments (ruleset 6.3)
    counter_trend_risk_mult: float = 0.5      # multiply risk_pct for counter-trend sizing
    counter_trend_min_rr: float = 3.0         # minimum R:R for counter-trend trades
    counter_trend_timeout_hours: int = 36     # shorter timeout (vs position_timeout_hours)

    # Trailing stop (trail SL to swing points after TP1 fill)
    trailing_sl_enabled: bool = False
    trailing_sl_lookback: int = 3             # swing detection lookback (4H candles each side)

    # Discord
    discord_ops_webhook_url: str = ""
    discord_urgent_webhook_url: str = ""
    discord_errors_webhook_url: str = ""

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
        max_open_orders=int(os.getenv("EXECUTION_MAX_OPEN_ORDERS", "200")),
        max_notional_usd=float(os.getenv("EXECUTION_MAX_NOTIONAL_USD", "0.0")),
        max_concurrent_positions=int(os.getenv("EXECUTION_MAX_POSITIONS", "100")),
        max_open_longs=int(os.getenv("EXECUTION_MAX_LONGS", "100")),
        max_portfolio_exposure_pct=float(os.getenv("EXECUTION_MAX_EXPOSURE_PCT", "100.0")),
        gcp_project_id=gcp_project,
        position_timeout_hours=int(os.getenv("EXECUTION_POSITION_TIMEOUT_HOURS", "72")),
        drift_enabled=os.getenv("DRIFT_ENABLED", "false").lower() in ("true", "1", "yes"),
        conditional_entry_enabled=os.getenv("CONDITIONAL_ENTRY_ENABLED", "false").lower() in ("true", "1", "yes"),
        conditional_entry_max_expiry=int(os.getenv("CONDITIONAL_ENTRY_MAX_EXPIRY", "24")),
        trailing_sl_enabled=os.getenv("TRAILING_SL_ENABLED", "false").lower() in ("true", "1", "yes"),
        discord_ops_webhook_url=os.getenv("DISCORD_OPS_WEBHOOK_URL", ""),
        discord_urgent_webhook_url=os.getenv("DISCORD_URGENT_WEBHOOK_URL", ""),
        discord_errors_webhook_url=os.getenv("DISCORD_ERRORS_WEBHOOK_URL", ""),
    )
