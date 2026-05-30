"""T-016 Phase 1a — MachineBlock equivalence battery (C1, pure).

The behaviour-preservation gate for the typed-MachineBlock migration. Proves
`MachineBlock.from_raw`/`to_raw` is byte-for-behaviour identical to the prior
raw-dict path (`scanner.llm.parse_machine_block`) on a corpus of 706 real
production rows (scans + pending_entries, snapshot 2026-05-14), and that
`reanchored()` reproduces the old in-place `_reconstruct_scan_result`
arithmetic exactly. No consumer wiring is exercised here (that lands C2-C5);
this is the pure boundary contract.
"""

import copy
import json
import re
from pathlib import Path

import pytest

from scanner.llm import parse_machine_block
from scanner.models import MachineBlock


def _legacy_validate_execution_params(params: dict, max_leverage=None):
    """FROZEN verbatim copy of validate_execution_params as it was BEFORE
    T-016 C3 (dict-accepting, .get-based). C3 migrates the live function to
    typed MachineBlock access; the behaviour baseline must not drift with it,
    so the frozen legacy parser calls THIS, not the live validator. Do not
    tidy or share — its value is immutability.
    """
    errors: list[str] = []
    direction = params.get("direction", "").upper()
    if direction not in ("LONG", "SHORT"):
        errors.append(f"invalid direction: {params.get('direction')}")
    entry_low = params.get("entry_low", 0)
    entry_high = params.get("entry_high", 0)
    stop = params.get("stop_loss", 0)
    tp1 = params.get("tp1", 0)
    confidence = params.get("confidence", 0)
    leverage = params.get("leverage", 0)
    for name, val in [("entry_low", entry_low), ("entry_high", entry_high),
                      ("stop_loss", stop), ("tp1", tp1)]:
        if val <= 0:
            errors.append(f"{name} must be positive: {val}")
    if entry_low > entry_high:
        errors.append(f"entry_low ({entry_low}) > entry_high ({entry_high})")
    if direction == "LONG" and stop > 0 and entry_low > 0:
        if stop >= entry_low:
            errors.append(f"LONG stop ({stop}) must be below entry_low ({entry_low})")
        if tp1 > 0 and tp1 <= entry_high:
            errors.append(f"LONG tp1 ({tp1}) must be above entry_high ({entry_high})")
    elif direction == "SHORT" and stop > 0 and entry_high > 0:
        if stop <= entry_high:
            errors.append(f"SHORT stop ({stop}) must be above entry_high ({entry_high})")
        if tp1 > 0 and tp1 >= entry_low:
            errors.append(f"SHORT tp1 ({tp1}) must be below entry_low ({entry_low})")
    tp2 = params.get("tp2")
    if tp2 is not None and tp1 > 0:
        if direction == "LONG" and tp2 <= tp1:
            errors.append(f"LONG tp2 ({tp2}) must be above tp1 ({tp1})")
        elif direction == "SHORT" and tp2 >= tp1:
            errors.append(f"SHORT tp2 ({tp2}) must be below tp1 ({tp1})")
    tp1_pct = params.get("tp1_exit_pct", 0)
    tp2_pct = params.get("tp2_exit_pct", 0)
    if tp1_pct is not None and tp1_pct <= 0:
        errors.append(f"tp1_exit_pct must be positive: {tp1_pct}")
    if tp2_pct < 0:
        errors.append(f"tp2_exit_pct must be non-negative: {tp2_pct}")
    if tp1_pct + tp2_pct > 100:
        errors.append(f"exit pcts sum ({tp1_pct} + {tp2_pct} = {tp1_pct + tp2_pct}) > 100")
    if not (1 <= confidence <= 10):
        errors.append(f"confidence ({confidence}) must be 1-10")
    if leverage <= 0:
        errors.append(f"leverage must be positive: {leverage}")
    if max_leverage is not None and leverage > max_leverage:
        errors.append(f"leverage ({leverage}) exceeds max ({max_leverage})")
    trigger = params.get("entry_trigger")
    trigger_price = params.get("entry_trigger_price")
    inv_price = params.get("invalidation_price")
    has_any = any(v is not None for v in (trigger, trigger_price, inv_price))
    has_all = all(v is not None for v in (trigger, trigger_price, inv_price))
    if has_any and not has_all:
        errors.append("conditional entry: entry_trigger, entry_trigger_price, and invalidation_price must all be present together")
    elif has_all and direction in ("LONG", "SHORT"):
        if direction == "LONG" and inv_price < stop:
            errors.append(f"LONG invalidation_price ({inv_price}) must be at or above stop_loss ({stop})")
        if direction == "SHORT" and inv_price > stop:
            errors.append(f"SHORT invalidation_price ({inv_price}) must be at or below stop_loss ({stop})")
    return (len(errors) == 0, errors)


def _legacy_parse_machine_block(text: str):
    """FROZEN verbatim copy of parse_machine_block as it was BEFORE T-016
    Phase 1a (commit 83ddd7a), returning a plain dict | None.

    This is the behaviour-preservation baseline. The battery must compare
    against this frozen reference, NOT the live parse_machine_block — C2
    rewrites the live function to delegate to MachineBlock.from_raw, so
    comparing against it would be circular (from_raw vs itself). Do not
    "tidy" this to call shared code; its whole value is that it cannot drift.
    """
    m = re.search(
        r"<!--\s*MACHINE_BLOCK\s*\n(.+?)\nMACHINE_BLOCK\s*-->", text, re.DOTALL,
    )
    if not m:
        return None
    try:
        data = json.loads(m.group(1).strip())
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    required = {
        "entry_low": (int, float), "entry_high": (int, float),
        "stop_loss": (int, float), "tp1": (int, float),
        "tp1_exit_pct": (int, float), "direction": str,
        "confidence": (int, float), "leverage": (int, float),
    }
    for fld, types in required.items():
        if fld not in data or not isinstance(data[fld], types):
            return None
    for fld in ("entry_low", "entry_high", "stop_loss", "tp1", "tp1_exit_pct",
                "confidence", "leverage"):
        data[fld] = float(data[fld])
    if "tp2" in data:
        if not isinstance(data["tp2"], (int, float)):
            return None
        data["tp2"] = float(data["tp2"])
    if "tp2_exit_pct" in data:
        if not isinstance(data["tp2_exit_pct"], (int, float)):
            return None
        data["tp2_exit_pct"] = float(data["tp2_exit_pct"])
    _VT = ("candle_close_above", "candle_close_below", "limit_at")
    if "entry_trigger" in data and data["entry_trigger"] not in _VT:
        return None
    for opt_num in ("entry_trigger_price", "invalidation_price"):
        if opt_num in data:
            if not isinstance(data[opt_num], (int, float)):
                return None
            data[opt_num] = float(data[opt_num])
    if "expiry_candles" in data:
        if not isinstance(data["expiry_candles"], (int, float)):
            return None
        data["expiry_candles"] = min(int(data["expiry_candles"]), 24)
    _VST = ("A", "B", "C", "D", "E", "F", "G")
    if "setup_type" in data:
        if not isinstance(data["setup_type"], str):
            return None
        st = data["setup_type"].strip().upper()
        if st not in _VST:
            return None
        data["setup_type"] = st
    if "confluence_score" in data:
        if not isinstance(data["confluence_score"], (int, float)):
            return None
        score = int(data["confluence_score"])
        if not 0 <= score <= 15:
            return None
        data["confluence_score"] = score
    if "confirming_event" in data:
        if not isinstance(data["confirming_event"], str):
            return None
        ev = data["confirming_event"].strip()
        if not 10 <= len(ev) <= 200:
            return None
        data["confirming_event"] = ev
    valid, _e = _legacy_validate_execution_params(data)
    if not valid:
        return None
    return data

_CORPUS = json.loads(
    (Path(__file__).parent / "fixtures" / "machine_block_corpus.json").read_text()
)
_ROWS = [(t, r) for t in ("scans", "pending_entries") for r in _CORPUS[t]]


def _wrap(raw_json: str) -> str:
    """Embed a machine_block JSON string as parse_machine_block expects."""
    return f"<!-- MACHINE_BLOCK\n{raw_json}\nMACHINE_BLOCK -->"


def _old_reanchor_inplace(mb: dict, trigger_close: float) -> dict:
    """Frozen verbatim copy of the pre-Phase-1a in-place reanchor arithmetic
    (formerly entry_monitor._reconstruct_scan_result; that function was
    deleted in Phase 1b, SPEC #21). This is the immutable reference
    implementation MachineBlock.reanchored must stay byte-identical to.
    """
    mb = copy.deepcopy(mb)
    orig_entry_low = float(mb.get("entry_low", 0))
    orig_entry_high = float(mb.get("entry_high", 0))
    orig_entry_mid = (
        (orig_entry_low + orig_entry_high) / 2 if orig_entry_low else trigger_close
    )
    delta = trigger_close - orig_entry_mid
    if mb.get("stop_loss") is not None:
        mb["stop_loss"] = float(mb["stop_loss"]) + delta
    if mb.get("tp1") is not None:
        mb["tp1"] = float(mb["tp1"]) + delta
    if mb.get("tp2") is not None:
        mb["tp2"] = float(mb["tp2"]) + delta
    buffer = trigger_close * 0.005
    mb["entry_low"] = trigger_close - buffer
    mb["entry_high"] = trigger_close + buffer
    return mb


def test_corpus_loaded():
    assert len(_ROWS) == 706, "corpus must be the full 556 scans + 150 pending"


# --- Reject-parity + to_raw semantic identity over the real corpus ---------

@pytest.mark.parametrize("src,raw", _ROWS, ids=[f"{t}:{i}" for t in ("scans", "pending_entries") for i in range(len(_CORPUS[t]))])
def test_from_raw_matches_parse_machine_block(src, raw):
    d = json.loads(raw)
    old = _legacy_parse_machine_block(_wrap(raw))
    mb = MachineBlock.from_raw(d)

    # Reject parity: from_raw is None on EXACTLY the inputs parse rejects.
    assert (mb is None) == (old is None), (
        f"reject divergence ({src}): from_raw={'None' if mb is None else 'ok'} "
        f"parse={'None' if old is None else 'ok'}"
    )
    if old is None:
        return

    # Semantic JSON identity: to_raw() == parse_machine_block's dict.
    assert mb.to_raw() == old, f"to_raw != parse output ({src})"
    # Round-trip idempotence.
    assert MachineBlock.from_raw(mb.to_raw()) == mb, f"round-trip not idempotent ({src})"


# --- reanchored() == old in-place arithmetic (mutation -> pure) ------------

@pytest.mark.parametrize("src,raw", _ROWS, ids=[f"{t}:{i}" for t in ("scans", "pending_entries") for i in range(len(_CORPUS[t]))])
def test_reanchored_matches_old_inplace(src, raw):
    mb = MachineBlock.from_raw(json.loads(raw))
    if mb is None:
        pytest.skip("invalid row — reanchor-of-invalid is a C4 battery item")
    coerced = mb.to_raw()
    for tc in (mb.entry_low * 0.97, mb.entry_high * 1.05, mb.tp1, mb.stop_loss):
        expected = _old_reanchor_inplace(coerced, tc)
        got = mb.reanchored(tc).to_raw()
        assert got == expected, f"reanchor divergence ({src}) tc={tc}"


# --- unknown/future-key passthrough (extra) -------------------------------

def _valid_block(**over):
    b = {
        "entry_low": 100.0, "entry_high": 101.0, "stop_loss": 98.0,
        "tp1": 105.0, "tp1_exit_pct": 50.0, "direction": "LONG",
        "confidence": 6.0, "leverage": 3.0,
    }
    b.update(over)
    return b


def test_unknown_keys_round_trip_via_extra():
    raw = _valid_block(future_field="v9.3-thing", nested={"a": [1, 2]})
    old = _legacy_parse_machine_block(_wrap(json.dumps(raw)))
    mb = MachineBlock.from_raw(raw)
    assert old is not None and mb is not None
    assert mb.extra == {"future_field": "v9.3-thing", "nested": {"a": [1, 2]}}
    assert mb.to_raw() == old  # parse keeps unknown keys; to_raw must too
    # extra is defensively isolated (no shared mutable state)
    mb.to_raw()["nested"]["a"].append(99)
    assert mb.extra["nested"]["a"] == [1, 2]


def test_reanchored_owns_independent_extra():
    """Phase-review (architecture R2): frozen=True does not deep-freeze the
    `extra` dict, and dataclasses.replace would alias it into the reanchored
    block. reanchored() must deep-copy so each MachineBlock genuinely owns
    its `extra` — no shared mutable side channel (SPEC #20 immutability).
    """
    mb = MachineBlock.from_raw(_valid_block(future_field="x", nested={"a": [1]}))
    assert mb is not None
    ra = mb.reanchored(123.45)
    assert ra.extra == mb.extra
    assert ra.extra is not mb.extra
    assert ra.extra["nested"] is not mb.extra["nested"]
    ra.extra["nested"]["a"].append(99)
    assert mb.extra["nested"]["a"] == [1]  # original unaffected


# --- malformed-rejection parity (synthetic) -------------------------------

@pytest.mark.parametrize("bad", [
    {k: v for k, v in _valid_block().items() if k != "stop_loss"},   # missing required
    _valid_block(direction="SIDEWAYS"),                              # bad direction (business rule)
    _valid_block(entry_trigger="teleport"),                          # invalid trigger
    _valid_block(confidence="high"),                                 # wrong type
    _valid_block(setup_type="Z"),                                    # bad observability enum
    _valid_block(confluence_score=99),                               # observability range
    _valid_block(confirming_event="short"),                          # observability length
    _valid_block(stop_loss=110.0),                                   # LONG SL above entry (business rule)
])
def test_malformed_rejected_like_parser(bad):
    assert (MachineBlock.from_raw(bad) is None) == (
        _legacy_parse_machine_block(_wrap(json.dumps(bad))) is None
    )
    assert MachineBlock.from_raw(bad) is None


def test_non_dict_input_rejected():
    assert MachineBlock.from_raw(None) is None
    assert MachineBlock.from_raw([1, 2]) is None
    assert MachineBlock.from_raw("not a dict") is None


def test_from_raw_accepts_v2_setup_type():
    # v2 (v10) lowercase setup_type passes the union valid-set (7b.1 casing)
    mb = MachineBlock.from_raw(_valid_block(setup_type="trend_continuation"))
    assert mb is not None and mb.setup_type == "trend_continuation"


def test_from_raw_v1_setup_type_canonical_uppercase():
    # a v1 letter normalizes to canonical uppercase regardless of input case
    mb = MachineBlock.from_raw(_valid_block(setup_type="a"))
    assert mb is not None and mb.setup_type == "A"


# --- setup_quality (inert back-compat field post-T-020.4 / SPEC #44; parse still works) ------

@pytest.mark.parametrize("val", ["textbook", "variant"])
def test_setup_quality_round_trips(val):
    # the v2 primary judgment parses + round-trips through to_raw
    mb = MachineBlock.from_raw(_valid_block(setup_quality=val))
    assert mb is not None and mb.setup_quality == val
    assert mb.to_raw()["setup_quality"] == val
    assert MachineBlock.from_raw(mb.to_raw()) == mb  # idempotent


def test_setup_quality_case_insensitive():
    # "Textbook" / "VARIANT" canonicalize to lowercase
    mb = MachineBlock.from_raw(_valid_block(setup_quality="Textbook"))
    assert mb is not None and mb.setup_quality == "textbook"


@pytest.mark.parametrize("bad", ["strong", "", 7, None, "textbookish"])
def test_setup_quality_invalid_coerced_not_rejected(bad):
    # LENIENT-coerce (SPEC #35): an invalid value on this field must NOT downgrade an otherwise-
    # valid CONFIRM to None at the parser — the block survives, the field drops to None, and
    # to_raw omits it (distinct from the strict-reject capital-path fields). The post-LLM
    # setup_validator then demotes the None-quality CONFIRM with REJECT_NON_TEXTBOOK (SPEC #33).
    mb = MachineBlock.from_raw(_valid_block(setup_quality=bad))
    assert mb is not None
    assert mb.setup_quality is None
    assert "setup_quality" not in mb.to_raw()


def test_v1_block_byte_identical_without_setup_quality():
    # a v1 block (no setup_quality emitted) round-trips with no new key — the byte-identical guarantee
    mb = MachineBlock.from_raw(_valid_block())
    assert mb is not None and mb.setup_quality is None
    assert "setup_quality" not in mb.to_raw()


# --- compat shim parity with a plain dict (removed in C6) -----------------

def test_no_mapping_shim_C6_enforcement():
    """T-016 C6: the temporary expand->migrate->contract compat shim is
    DELETED and must never return. This is the enforcement guard — the type
    itself physically refuses mapping access, so any reintroduced
    `mb["x"]` / `mb.get(...)` / `"x" in mb` on a MachineBlock fails loudly
    in the suite (stronger and false-positive-free vs an AST/grep scan).
    """
    mb = MachineBlock.from_raw(_valid_block(tp2=110.0, tp2_exit_pct=50.0))
    assert mb is not None
    for attr in ("__getitem__", "get", "__contains__", "_shim_lookup"):
        assert not hasattr(MachineBlock, attr), f"shim leaked back: {attr}"
    with pytest.raises(TypeError):
        _ = mb["entry_low"]
    with pytest.raises(TypeError):
        _ = "entry_low" in mb
    # typed access is the only contract
    assert mb.entry_low == 100.0
    assert mb.tp2 == 110.0
    assert mb.confirming_event is None


# --- C2: live parse_machine_block delegates correctly to from_raw ----------

@pytest.mark.parametrize("src,raw", _ROWS, ids=[f"{t}:{i}" for t in ("scans", "pending_entries") for i in range(len(_CORPUS[t]))])
def test_live_parse_delegates_to_from_raw(src, raw):
    """C2 contract: the rewritten parse_machine_block returns a typed
    MachineBlock equal to from_raw, with to_raw matching the frozen legacy
    dict, over the full real corpus.
    """
    live = parse_machine_block(_wrap(raw))
    legacy = _legacy_parse_machine_block(_wrap(raw))
    if legacy is None:
        assert live is None
        return
    assert isinstance(live, MachineBlock)
    assert live == MachineBlock.from_raw(json.loads(raw))
    assert live.to_raw() == legacy


# --- C3: typed sizing/sanity/validate reproduce the pre-C3 snapshot --------
# scanner/tests/fixtures/c3_pre_snapshot.json was captured by running the
# DICT-era calculate_position / check_price_sanity / reanchor_exit_prices /
# validate_execution_params over the full corpus BEFORE the C3 typed rewrite.
# The typed implementations must reproduce it byte-for-byte. This is the
# real-money behaviour-preservation gate for C3.

from dataclasses import asdict  # noqa: E402

from scanner.execution.config import ExecutionConfig  # noqa: E402
from scanner.execution.sanity import check_price_sanity  # noqa: E402
from scanner.execution.sizing import (  # noqa: E402
    calculate_position,
    reanchor_exit_prices,
)
from scanner.llm import validate_execution_params  # noqa: E402

_C3_SNAP = json.loads(
    (Path(__file__).parent / "fixtures" / "c3_pre_snapshot.json").read_text()
)
_C3_CFG = ExecutionConfig(
    enabled=True, mode="testnet", private_key="0x", wallet_address="0x",
)
# Exact input grids the snapshot was generated with — must not change.
_CALC_PARAMS = [(1000.0, 1.0, None), (500.0, 0.5, 10), (10000.0, 2.0, 3)]


@pytest.mark.parametrize(
    "rec", _C3_SNAP,
    ids=[f"row{i}" for i in range(len(_C3_SNAP))],
)
def test_c3_typed_matches_pre_snapshot(rec):
    d = json.loads(rec["raw"])
    mb = MachineBlock.from_raw(d)

    if mb is None:
        # from_raw rejects exactly the inputs the pre-C3 path rejected, so the
        # snapshot's (legacy) validate must also be invalid. Invalid rows never
        # reach calc/sanity/reanchor in production (rejected at the boundary);
        # reject-parity is proven against the frozen legacy parser elsewhere.
        assert rec["validate"][0] is False, (
            f"from_raw->None but pre-C3 validate passed: {rec['validate']}"
        )
        return

    # validate: typed result == the pre-C3 dict-era result, every valid row.
    v, errs = validate_execution_params(mb)
    assert [v, errs] == rec["validate"], (
        f"validate divergence: typed={[v, errs]} snap={rec['validate']}"
    )

    em = (mb.entry_low + mb.entry_high) / 2 or 1.0

    for (bal, risk, ml), snap in zip(_CALC_PARAMS, rec["calc"]):
        ps = calculate_position(mb, bal, risk, ml)
        got = None if ps is None else asdict(ps)
        assert got == snap, f"calc divergence bal={bal} risk={risk} ml={ml}"

    for (price, atr), snap in zip(
        [(em, 0.0), (em * 1.03, em * 0.01), (em * 0.9, em * 0.02)], rec["sanity"]
    ):
        r = check_price_sanity(mb, price, atr, _C3_CFG)
        assert [r.passed, r.errors] == snap, f"sanity divergence price={price} atr={atr}"

    for fp, snap in zip([em, em * 1.01, em * 0.985], rec["reanchor"]):
        assert reanchor_exit_prices(mb, fp) == snap, f"reanchor divergence fp={fp}"
