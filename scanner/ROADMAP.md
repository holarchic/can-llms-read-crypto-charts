# Scanner Roadmap

## Signal Ranking Evolution

### Tier 1: Composite scoring (SHIPPED)
Weighted composite replaces static per-setup-type confidence.
Factors: base prior (0.30), structure alignment (0.25), level strength (0.15), proximity (0.15), confluence (0.15).

### Tier 2: Diminishing returns penalty
**When**: If clustering appears in production (e.g. 15/20 signals are the same setup type).
**What**: Greedy selection with decay — `effective_score = composite * 0.85^n` where `n` = count of same setup type already selected. Replaces the simple sort+slice in `pipeline.py:stage1_screen`.
**Scope**: ~20 lines in `pipeline.py`. No model/screener changes.

### Tier 3: Feedback loop from historical results
**When**: 50+ LLM results per setup type accumulated in `scanner.db`.
**What**: Query historical CONFIRM rate by setup type from storage. Blend empirical hit rate as a 6th scoring factor in `score_signal()`. The LLM's own judgment feeds back into which signals get prioritized — the system learns which setups are worth spending tokens on.

**Implementation sketch**:
1. `storage.py` — add `confirm_rate_by_setup(conn, min_samples=50) -> dict[str, float]` querying `scan_results` table.
2. `screener.py` — add `W_EMPIRICAL` weight (steal from base prior). When data is insufficient, weight falls back to 0 and gets redistributed to base.
3. `pipeline.py` — pass rates into `detect_all_setups()` or post-score adjustment.

**Why wait**: Premature optimization. Need real data to validate. Bad early data would create a self-reinforcing bias (low-sample setups get deprioritized → never get sampled → stay deprioritized).
