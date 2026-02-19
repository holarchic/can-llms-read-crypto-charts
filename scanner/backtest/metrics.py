"""Aggregation: win rate, expectancy, profit factor, slicing by dimension."""

import random
from dataclasses import dataclass, field

from .resolver import TradeOutcome


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval on expectancy."""

    ci_low: float
    ci_high: float
    confidence: float
    excludes_zero: bool


def bootstrap_expectancy(
    outcomes: list[TradeOutcome],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BootstrapCI | None:
    """Bootstrap CI on expectancy (mean R-multiple).

    Returns None if fewer than 5 outcomes (bootstrap meaningless).
    """
    if len(outcomes) < 5:
        return None

    r_values = [o.r_achieved for o in outcomes]
    n = len(r_values)
    rng = random.Random(seed)

    boot_means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(r_values) for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    alpha = 1 - confidence
    lo_idx = int(n_bootstrap * alpha / 2)
    hi_idx = int(n_bootstrap * (1 - alpha / 2)) - 1
    ci_low = boot_means[lo_idx]
    ci_high = boot_means[hi_idx]

    return BootstrapCI(
        ci_low=round(ci_low, 4),
        ci_high=round(ci_high, 4),
        confidence=confidence,
        excludes_zero=ci_low > 0 or ci_high < 0,
    )


@dataclass
class MetricsReport:
    """Aggregated metrics for a set of trade outcomes."""

    label: str
    sample_size: int
    wins: int
    losses: int
    expired: int
    win_rate: float
    avg_r: float
    expectancy: float
    profit_factor: float
    total_r: float
    avg_bars_to_exit: float
    avg_mae: float
    avg_mfe: float
    avg_win_r: float
    avg_loss_r: float
    bootstrap_ci: BootstrapCI | None = None


def compute_metrics(outcomes: list[TradeOutcome], label: str = "All") -> MetricsReport:
    """Compute aggregate metrics from a list of trade outcomes."""
    n = len(outcomes)
    if n == 0:
        return MetricsReport(
            label=label, sample_size=0, wins=0, losses=0, expired=0,
            win_rate=0.0, avg_r=0.0, expectancy=0.0, profit_factor=0.0,
            total_r=0.0, avg_bars_to_exit=0.0, avg_mae=0.0, avg_mfe=0.0,
            avg_win_r=0.0, avg_loss_r=0.0,
        )

    wins = [o for o in outcomes if o.result == "WIN"]
    losses = [o for o in outcomes if o.result == "LOSS"]
    expired = [o for o in outcomes if o.result == "EXPIRED"]

    total_r = sum(o.r_achieved for o in outcomes)
    avg_r = total_r / n

    win_r_values = [o.r_achieved for o in wins]
    loss_r_values = [o.r_achieved for o in losses]

    avg_win_r = sum(win_r_values) / len(win_r_values) if win_r_values else 0.0
    avg_loss_r = sum(loss_r_values) / len(loss_r_values) if loss_r_values else 0.0

    win_rate = len(wins) / n if n > 0 else 0.0

    # Expectancy: (win_rate * avg_win) + (loss_rate * avg_loss)
    # avg_loss is already negative
    loss_rate = len(losses) / n if n > 0 else 0.0
    expectancy = (win_rate * avg_win_r) + (loss_rate * avg_loss_r)

    # Profit factor: gross wins / gross losses (absolute)
    gross_wins = sum(o.r_achieved for o in wins)
    gross_losses = abs(sum(o.r_achieved for o in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 0.0

    avg_bars = sum(o.bars_to_exit for o in outcomes) / n
    avg_mae = sum(o.mae for o in outcomes) / n
    avg_mfe = sum(o.mfe for o in outcomes) / n

    ci = bootstrap_expectancy(outcomes) if n >= 5 else None

    return MetricsReport(
        label=label,
        sample_size=n,
        wins=len(wins),
        losses=len(losses),
        expired=len(expired),
        win_rate=round(win_rate, 4),
        avg_r=round(avg_r, 4),
        expectancy=round(expectancy, 4),
        profit_factor=round(profit_factor, 4),
        total_r=round(total_r, 4),
        avg_bars_to_exit=round(avg_bars, 1),
        avg_mae=round(avg_mae, 4),
        avg_mfe=round(avg_mfe, 4),
        avg_win_r=round(avg_win_r, 4),
        avg_loss_r=round(avg_loss_r, 4),
        bootstrap_ci=ci,
    )


# --- Slicers ---

def _confidence_bucket(confidence: float) -> str:
    if confidence >= 0.7:
        return "high"
    elif confidence >= 0.5:
        return "med"
    else:
        return "low"


DEFAULT_SLICERS = {
    "setup_type": lambda o: o.setup.setup_type if o.setup else "unknown",
    "direction": lambda o: o.setup.direction if o.setup else "unknown",
    "structure": lambda o: o.setup.structure if o.setup else "unknown",
    "confidence": lambda o: _confidence_bucket(o.setup.confidence) if o.setup else "unknown",
    "sent_to_llm": lambda o: "yes" if (o.setup and o.setup.sent_to_llm) else "no",
    "regime": lambda o: o.setup.regime if (o.setup and o.setup.regime) else "unknown",
}


def slice_metrics(
    outcomes: list[TradeOutcome],
    slicers: dict | None = None,
) -> dict[str, list[MetricsReport]]:
    """Slice outcomes by multiple dimensions, compute metrics for each slice.

    Returns {dimension_name: [MetricsReport per slice value]}.
    """
    if slicers is None:
        slicers = DEFAULT_SLICERS

    result = {}
    for dim_name, key_fn in slicers.items():
        buckets: dict[str, list[TradeOutcome]] = {}
        for o in outcomes:
            key = str(key_fn(o))
            buckets.setdefault(key, []).append(o)

        reports = []
        for key in sorted(buckets.keys()):
            reports.append(compute_metrics(buckets[key], label=key))
        result[dim_name] = reports

    return result
