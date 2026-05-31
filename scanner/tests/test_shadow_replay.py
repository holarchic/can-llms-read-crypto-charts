"""T-020.15 replay anchor planning (SPEC #53) — `scanner.shadow.replay`.

The pure logic behind the hyphenated `scripts/shadow-replay.py` CLI: duration parsing, the
newest-first anchor schedule, and the >=12h spacing floor (below the dedup window the
skipped-dedup replay over-captures persistent setups)."""

from datetime import datetime, timedelta, timezone

import pytest

from scanner.shadow.replay import MIN_SPACING, parse_duration, plan_anchors

_NOW = datetime(2026, 5, 31, 0, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("s, td", [
    ("24h", timedelta(hours=24)),
    ("3d", timedelta(days=3)),
    ("90m", timedelta(minutes=90)),
    ("2w", timedelta(weeks=2)),
    ("12", timedelta(hours=12)),   # bare number ⇒ hours
])
def test_parse_duration(s, td):
    assert parse_duration(s) == td


def test_plan_anchors_newest_first_and_spacing():
    anchors = plan_anchors(_NOW, anchors=4, spacing=timedelta(days=1), min_age=timedelta(days=3))
    assert len(anchors) == 4
    # newest-first: index 0 is the freshest (now - min_age); each subsequent is one spacing older
    assert anchors[0] == _NOW - timedelta(days=3)
    assert anchors[-1] == _NOW - timedelta(days=3) - timedelta(days=3)  # 3 spacings older
    assert anchors == sorted(anchors, reverse=True)  # strictly newest → oldest


def test_plan_anchors_rejects_subwindow_spacing():
    """Spacing below the 12h dedup window would over-capture a persistent setup (skipped dedup)."""
    with pytest.raises(ValueError, match="MIN_SPACING"):
        plan_anchors(_NOW, anchors=5, spacing=timedelta(hours=6), min_age=timedelta(days=3))
    # exactly the floor is allowed
    assert len(plan_anchors(_NOW, anchors=2, spacing=MIN_SPACING, min_age=timedelta(days=3))) == 2


def test_plan_anchors_rejects_nonpositive_count():
    with pytest.raises(ValueError, match="anchors must be"):
        plan_anchors(_NOW, anchors=0, spacing=timedelta(days=1), min_age=timedelta(days=3))
