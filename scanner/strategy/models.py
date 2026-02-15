"""Strategy dataclass — full parameter set for reproducible scanning."""

from dataclasses import dataclass

from ..config import ScannerConfig


@dataclass
class Strategy:
    strategy_id: str
    label: str
    swing_lookback: int = 5
    sr_cluster_pct: float = 0.015
    proximity_atr_mult: float = 1.5
    max_setups_per_coin: int = 3
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
