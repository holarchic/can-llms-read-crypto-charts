"""Signal-quality backtesting for the scanner pipeline."""

from .metrics import MetricsReport, compute_metrics
from .resolver import TradeOutcome, TradeSetup, fetch_and_resolve, parse_price, resolve_signal

__all__ = [
    "TradeSetup",
    "TradeOutcome",
    "resolve_signal",
    "fetch_and_resolve",
    "parse_price",
    "MetricsReport",
    "compute_metrics",
]
