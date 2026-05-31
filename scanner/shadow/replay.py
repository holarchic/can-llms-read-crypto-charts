"""Anchor planning for the T-020.15 historical-replay shadow scan (SPEC #53).

Pure helpers behind `scripts/shadow-replay.py`. A hyphenated script can't be imported, so the
logic that deserves unit tests (duration parsing + anchor planning + the spacing floor) lives
here — the `runner.py` precedent. The CLI sets `LLM_TRADER_DB` (isolated `data/replay.db`) then
loops `run_scan(as_of=T)` over `plan_anchors(...)`.
"""

from datetime import datetime, timedelta

__all__ = ["MIN_SPACING", "parse_duration", "plan_anchors"]

# The live historical-dedup window (`scan_queries.is_duplicate`, 12h). Replay skips that gate, so
# anchors closer than this would over-capture a persistent setup live dedups to one — the floor.
MIN_SPACING = timedelta(hours=12)

_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def parse_duration(s: str) -> timedelta:
    """`'24h'` / `'3d'` / `'90m'` / `'2w'` → timedelta; a bare number ⇒ hours."""
    s = s.strip().lower()
    if s and s[-1] in _UNITS:
        return timedelta(**{_UNITS[s[-1]]: float(s[:-1])})
    return timedelta(hours=float(s))


def plan_anchors(now: datetime, anchors: int, spacing: timedelta, min_age: timedelta) -> list[datetime]:
    """The `anchors` past scan timestamps, NEWEST-first: `now - min_age - i*spacing`, i in 0..n-1.

    Newest-first so a `daily_cost_cap` cutoff drops the stalest (highest training-cutoff-hindsight)
    anchors, not the freshest. Raises if `anchors < 1` or `spacing < MIN_SPACING` (below the 12h
    dedup window the skipped-dedup replay over-captures persistent setups).
    """
    if anchors < 1:
        raise ValueError(f"anchors must be >= 1, got {anchors}")
    if spacing < MIN_SPACING:
        raise ValueError(f"spacing {spacing} < MIN_SPACING {MIN_SPACING} (the 12h dedup window)")
    return [now - min_age - i * spacing for i in range(anchors)]
