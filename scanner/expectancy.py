"""Per-trade-R cohort expectancy — the shared statistic behind both validation gates.

Single owner of "a list of per-trade R → n / win% / avg R / profit factor / bootstrap 95% CI
/ pass-flag". Used by the shadow cutover gate (Acceptance #10, `scanner/shadow`) AND the live
per-version cohort surface (Acceptance #8, `scanner/execution/analytics.per_version_expectancy`).

Pure + stdlib-only (no `scanner.execution`/`scanner.shadow`/`scanner.models` import) so it is a
true leaf both subsystems depend on without inverting the layering. Operates on the per-trade
net-R sample: 0R outcomes stay in; non-trade statuses are excluded upstream. The gate is the
project-standard one (Acceptance #2/#8/#10): **n ≥ min_n AND the bootstrap 95% CI lower bound
> 0** — "excludes zero" means the CI bound, NOT dropping zero-R trades.
"""

import random
from dataclasses import dataclass

__all__ = ["DEFAULT_MIN_N", "ExpectancyReport", "bootstrap_ci", "expectancy_report"]

# Single owner of the cohort-size floor (Acceptance #2/#8/#10) — `expectancy_report`'s
# `min_n` default AND the CLI's PASS/PENDING/FAIL split read this, so it lives in one place.
DEFAULT_MIN_N = 30


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
    realized_rs: list[float], *, min_n: int = DEFAULT_MIN_N, iters: int = 10000, seed: int = 0,
) -> ExpectancyReport:
    """n / win% / avg R / profit factor / bootstrap CI + the gate pass flag (n≥min_n ∧ ci_lo>0)."""
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
