"""Tests for scanner/prompt_select.py + version-gated prompt persistence (T-019 7b.3)."""

from datetime import datetime, timezone

import pytest

from scanner.config import STRATEGY_PATH, STRATEGY_PATH_V2
from scanner.db import get_connection
from scanner.llm import SCANNER_MODE_SUFFIX, SYSTEM_PROMPT, load_strategy
from scanner.models import (
    BoundarySignal,
    Direction,
    ScanResult,
    SetupType,
    SpineContext,
    Structure,
)
from scanner.prompt_select import select_prompt
from scanner.scan_store import save_result


def test_prompt_versions():
    assert select_prompt("v1").prompt_version == "9.11"
    assert select_prompt("v2").prompt_version == "10.00"


def test_v1_system_text_is_systemprompt_plus_suffix():
    """The selector assembles exactly SYSTEM_PROMPT.format(ruleset=v1) + SCANNER_MODE_SUFFIX
    — the suffix being the hash-verified verbatim relocation of the pre-7b.3 inline string,
    so the v1 prompt is byte-identical to before the seam was extracted."""
    spec = select_prompt("v1")
    assert spec.system_text == SYSTEM_PROMPT.format(ruleset=load_strategy(STRATEGY_PATH)) + SCANNER_MODE_SUFFIX
    assert "## SCANNER MODE" in spec.system_text
    assert "## MACHINE BLOCK (CONFIRM only)" in spec.system_text


def test_v2_injects_v2_ruleset_not_v1():
    v1, v2 = select_prompt("v1"), select_prompt("v2")
    assert v2.system_text != v1.system_text
    assert load_strategy(STRATEGY_PATH_V2) in v2.system_text       # the v2 ruleset is what's injected
    assert "v10" in v2.system_text or "the framework" in v2.system_text


def test_load_strategy_paths():
    assert load_strategy(STRATEGY_PATH) == STRATEGY_PATH.read_text()
    assert load_strategy(STRATEGY_PATH_V2) == STRATEGY_PATH_V2.read_text()
    assert STRATEGY_PATH_V2.exists()                    # v2 stub shipped → fail-closed-safe


def test_unknown_version_fails_closed():
    # config validates the flag upstream; an unknown version is a programmer error and
    # raises (fail-closed), never silently the v1 cohort.
    with pytest.raises(KeyError):
        select_prompt("v3")


def test_prompt_version_persists_per_result(tmp_path):
    """The per-result prompt_version reaches the scans column (pipeline persists
    result.prompt_version rather than a module constant)."""
    conn = get_connection(tmp_path / "t.db")
    spine = SpineContext(structure=Structure.RANGING, sr_levels=[], range_bounds=None,
                         atr=1.0, current_price=100.0, fib_levels=None)
    sig = BoundarySignal(coin="BTC", setup_type=SetupType.C, direction=Direction.LONG,
                         confidence=0.7, key_level=100.0, context=spine, description="x", setup_tf="4h")
    result = ScanResult(scan_id="abcd1234", coin="BTC", timestamp=datetime.now(timezone.utc),
                        setup_signal=sig, decision="REJECT", confluence_score=None,
                        analysis_text="", parsed_decision={}, chart_paths=[], cost_usd=0.0,
                        prompt_version="10.00")
    save_result(result, conn)  # persists result.prompt_version (no override param)
    row = conn.execute("SELECT prompt_version FROM scans WHERE scan_id='abcd1234'").fetchone()
    conn.close()
    assert row["prompt_version"] == "10.00"
