"""Expectancy readout for the shadow tape (T-020.1, SPEC #41) — emits the Acceptance-#10
cutover verdict.

Pure. Operates on the per-trade net-R sample (`store.resolved_r_values`): 0R outcomes
stay in the sample; non-trade statuses were already excluded upstream. The gate is the
project-standard one (Acceptance #2/#8): **n ≥ min_n AND the bootstrap 95% CI lower bound
> 0** — "excludes zero" means the CI bound, NOT dropping zero-R trades.
"""

import random
from dataclasses import dataclass

__all__ = ["ExpectancyReport", "bootstrap_ci", "expectancy_report"]


@dataclass(frozen=True)
class ExpectancyReport:
    n: int
    win_rate: float
    avg_r: float
    profit_factor: float
    ci_lo: float
    ci_hi: float
    gate_pass: bool


def bootstrap_ci(
    samples: list[float], *, iters: int = 10000, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the MEAN per-trade R. Seeded for reproducibility."""
    if not samples:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(samples)
    means = sorted(
        sum(samples[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(iters)
    )
    lo = means[int((alpha / 2) * iters)]
    hi = means[min(int((1 - alpha / 2) * iters), iters - 1)]
    return (lo, hi)


def expectancy_report(
    realized_rs: list[float], *, min_n: int = 30, iters: int = 10000, seed: int = 0,
) -> ExpectancyReport:
    """n / win% / avg R / profit factor / bootstrap CI + the Gate-#10 pass flag."""
    rs = list(realized_rs)
    n = len(rs)
    if n == 0:
        return ExpectancyReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, False)
    wins = [r for r in rs if r > 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(r for r in rs if r < 0))
    avg_r = sum(rs) / n
    win_rate = len(wins) / n
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
    ci_lo, ci_hi = bootstrap_ci(rs, iters=iters, seed=seed)
    return ExpectancyReport(
        n=n, win_rate=win_rate, avg_r=avg_r, profit_factor=profit_factor,
        ci_lo=ci_lo, ci_hi=ci_hi, gate_pass=(n >= min_n and ci_lo > 0),
    )
