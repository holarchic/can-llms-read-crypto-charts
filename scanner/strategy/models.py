"""Strategy dataclass — full parameter set for reproducible scanning."""

from dataclasses import dataclass

from ..config import (
    DEFAULT_MAX_SETUPS_PER_COIN,
    DEFAULT_PROXIMITY_ATR_MULT,
    DEFAULT_SR_CLUSTER_PCT,
    DEFAULT_SWING_LOOKBACK,
    ScannerConfig,
)


@dataclass
class Strategy:
    strategy_id: str
    label: str
    # Screener params mirror scanner.config defaults (T-016 Phase 3, SPEC #26 a
    # — single source in scanner/config.py, not re-declared literals here).
    swing_lookback: int = DEFAULT_SWING_LOOKBACK
    sr_cluster_pct: float = DEFAULT_SR_CLUSTER_PCT
    proximity_atr_mult: float = DEFAULT_PROXIMITY_ATR_MULT
    max_setups_per_coin: int = DEFAULT_MAX_SETUPS_PER_COIN
    atr_stop_mult: float = 1.5
    atr_target_mult: float = 3.0
    ruleset_version: str = ""
    is_active: bool = False
    disabled_types: str = ""  # comma-separated, e.g. "A,E"
    notes: str = ""
    created_at: str = ""

    @property
    def disabled_types_set(self) -> set[str]:
        if not self.disabled_types:
            return set()
        return {t.strip().upper() for t in self.disabled_types.split(",") if t.strip()}

    def to_scanner_config(self, **overrides) -> ScannerConfig:
        """Build a ScannerConfig from this strategy's params."""
        return ScannerConfig(
            swing_lookback=self.swing_lookback,
            sr_cluster_pct=self.sr_cluster_pct,
            proximity_atr_mult=self.proximity_atr_mult,
            max_setups_per_coin=self.max_setups_per_coin,
            **overrides,
        )
