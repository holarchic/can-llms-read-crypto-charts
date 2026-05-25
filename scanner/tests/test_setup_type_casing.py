"""v1/v2 setup_type casing compatibility (T-019 Chunk 7b.1).

v1 setup types are UPPERCASE ``A``–``G``; v2 (v10) are lowercase snake_case.
The shared ``normalize_setup_type`` canonicalizes per vocabulary so every reader
tolerates both during cutover (``from_raw`` validates against the union). Default
``setup_version="v1"`` keeps the live path byte-identical until 7b.2b dispatch.
"""

from scanner.models import SETUP_NAMES, SetupType, SetupTypeV2, normalize_setup_type
from scanner.strategy.models import Strategy


def test_normalize_v1_uppercases_any_case():
    assert normalize_setup_type("a") == "A"
    assert normalize_setup_type(" b ") == "B"
    assert normalize_setup_type("G") == "G"


def test_normalize_v2_lowercases_any_case():
    assert normalize_setup_type("TREND_CONTINUATION") == "trend_continuation"
    assert normalize_setup_type("gap_fill") == "gap_fill"


def test_normalize_unknown_is_harmless_lower():
    # not a member of either vocabulary → lowercased; never matches a real type
    assert normalize_setup_type("bogus") == "bogus"


def test_setup_names_covers_both_vocabularies():
    for t in SetupType:
        assert t in SETUP_NAMES
    for t in SetupTypeV2:
        assert t in SETUP_NAMES


def test_disabled_types_set_mixed_vocab():
    # v1 letters → canonical uppercase (unchanged); v2 tokens preserved lowercase
    s = Strategy(strategy_id="x", disabled_types="a, e, gap_fill").disabled_types_set
    assert s == {"A", "E", "gap_fill"}
