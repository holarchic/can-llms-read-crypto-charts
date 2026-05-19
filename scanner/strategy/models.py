"""Strategy dataclass — disabled setup-detector types per strategy_id.

The live scan reads exactly `get_strategy(strategy_id).disabled_types_set`
(`scanner/pipeline.py`) to skip setup detectors. Screener/resolution params
are single-owned in `scanner/config.py` (decision #26); they are not carried
here.
"""

from dataclasses import dataclass


@dataclass
class Strategy:
    strategy_id: str
    disabled_types: str = ""  # comma-separated, e.g. "A,E"

    @property
    def disabled_types_set(self) -> set[str]:
        if not self.disabled_types:
            return set()
        return {t.strip().upper() for t in self.disabled_types.split(",") if t.strip()}
