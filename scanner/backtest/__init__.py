"""Signal-quality backtesting for the scanner pipeline."""

from .resolver import TradeSetup, TradeOutcome, resolve_signal, fetch_and_resolve, parse_price
from .metrics import MetricsReport, compute_metrics

__all__ = [
    "TradeSetup",
    "TradeOutcome",
    "resolve_signal",
    "fetch_and_resolve",
    "parse_price",
    "MetricsReport",
    "compute_metrics",
]
