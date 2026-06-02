"""Tests for scanner/prompt_select.py + version-gated prompt persistence (T-019 7b.3)."""

import json
import re
from datetime import datetime, timezone

import pytest

from scanner.config import (
    SETUP_VERSION_REGISTRY,
    STRATEGY_PATH,
    STRATEGY_PATH_V2,
    require_setupversion_keys,
)
from scanner.db import get_connection
from scanner.llm import (
    SCANNER_MODE_SUFFIX,
    SCANNER_MODE_SUFFIX_V2,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_V2,
    load_strategy,
)
from scanner.models import (
    BoundarySignal,
    Direction,
    MachineBlock,
    ScanResult,
    SetupType,
    SetupTypeV2,
    SpineContext,
    Structure,
)
from scanner.prompt_select import _PROMPT_PROSE, select_prompt
from scanner.scan_store import save_result


def test_prompt_versions():
    assert select_prompt("v1").prompt_version == "9.11"
    assert select_prompt("v2").prompt_version == "10.01"  # T-020.4 bump (veto collapse, SPEC #44)


def test_version_registries_keyed_by_setupversion():
    """Both live maps are keyed by the closed `SetupVersion` type, never raw `str` (T-027.2)."""
    require_setupversion_keys("_PROMPT_PROSE", _PROMPT_PROSE)              # the real import-time guard
    require_setupversion_keys("SETUP_VERSION_REGISTRY", SETUP_VERSION_REGISTRY)


def test_require_setupversion_keys_trips_on_raw_str():
    """The guard catches a raw-string-keyed regression that set-equality is blind to: a `str` key and
    a `SetupVersion` key are set- AND hash-equal (`{"v1","v2"} == set(SETUP_VERSION_REGISTRY)`), so the
    type check is the only thing that fires."""
    assert {"v1", "v2"} == set(SETUP_VERSION_REGISTRY)                    # the blind spot it backstops
    with pytest.raises(RuntimeError, match="must all be SetupVersion"):
        require_setupversion_keys("bad_map", {"v1": 0, "v2": 0})


def test_version_registries_are_read_only():
    """Both maps are frozen (`MappingProxyType`) so they can't be mutated after the import-time guards
    prove them well-formed (T-027.1)."""
    with pytest.raises(TypeError):
        SETUP_VERSION_REGISTRY[next(iter(SETUP_VERSION_REGISTRY))] = None  # type: ignore[index]
    with pytest.raises(TypeError):
        _PROMPT_PROSE[next(iter(_PROMPT_PROSE))] = None  # type: ignore[index]


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


def test_v2_system_text_is_v2prompt_plus_v2suffix():
    """The v2 selector assembles exactly SYSTEM_PROMPT_V2.format(ruleset=v2) +
    SCANNER_MODE_SUFFIX_V2 — all three taxonomy-bearing surfaces are the v2 prose (Chunk 8d)."""
    spec = select_prompt("v2")
    assert spec.system_text == (
        SYSTEM_PROMPT_V2.format(ruleset=load_strategy(STRATEGY_PATH_V2)) + SCANNER_MODE_SUFFIX_V2
    )


def test_v2_three_surface_vocabulary_coherent():
    """All 6 v2 setups appear across the assembled v2 surfaces; no v1 A–G setup tokens leak in;
    deviation_reclaim appears as a setup, never as an entry_trigger enum value."""
    text = select_prompt("v2").system_text
    for st in SetupTypeV2:
        assert st.value in text, st.value
    # v1 A–G decision-criteria / setup-specific tokens must not survive into v2
    assert "A-G per ruleset" not in text
    assert "Setup B (breakout)" not in text and "Setup F (deviation)" not in text
    # deviation_reclaim is a SETUP, not a trigger: the entry_trigger enum line excludes it
    trigger_line = next(ln for ln in text.splitlines() if ln.strip().startswith("- entry_trigger:"))
    assert "deviation_reclaim" not in trigger_line
    # T-020.4: the setup_quality self-label is gone from the v2 prompt; the veto framing replaces it
    assert "setup_quality" not in text
    assert "veto" in text.lower()


def test_v2_keeps_numeric_confidence_required():
    """Demoting confidence in the v2 prose must not let the LLM drop the still-required field:
    the visible format keeps **LLM Confidence:**, the machine-block surface keeps "confidence"."""
    sys_text = SYSTEM_PROMPT_V2  # the system-prompt surface
    assert "**LLM Confidence:**" in sys_text
    assert '"confidence":' in SCANNER_MODE_SUFFIX_V2          # the example JSON + field rule
    # confidence present in the assembled whole too
    assert '"confidence"' in select_prompt("v2").system_text


def test_v1_free_of_v2_vocabulary():
    """The v1 default path must stay clean of v2 vocabulary (byte separation undergirds the
    flag-reversible cutover). `deviation_reclaim` is the known cross-namespace collision — it
    appears in v1 as the *deprecated entry-trigger* name (the Setup-F rules forbid it), NOT as a
    v2 setup — so it is excluded from this check; all other v2 vocabulary must be absent."""
    text = select_prompt("v1").system_text
    assert "setup_quality" not in text
    v2_exclusive = [st.value for st in SetupTypeV2 if st is not SetupTypeV2.DEVIATION_RECLAIM]
    for tok in v2_exclusive:
        assert tok not in text, tok


def test_v2_example_machine_block_parses_and_validates():
    """The MACHINE_BLOCK example embedded in the v2 scanner suffix must be a valid block (so the
    LLM has a correct template) — v2 setup_type, and NO setup_quality (T-020.4 dropped it from the
    prompt; the model field is inert)."""
    text = select_prompt("v2").system_text
    m = re.search(r"<!--\s*MACHINE_BLOCK\s*\n(.+?)\nMACHINE_BLOCK\s*-->", text, re.DOTALL)
    assert m is not None
    mb = MachineBlock.from_raw(json.loads(m.group(1).strip()))
    assert mb is not None
    assert mb.setup_quality is None  # dropped from the v2 example (T-020.4)
    assert mb.setup_type == "trend_continuation"


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
