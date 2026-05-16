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
    from scanner.llm import validate_execution_params
    valid, _e = validate_execution_params(data)
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
    """Verbatim copy of entry_monitor._reconstruct_scan_result:192-206 — the
    pre-Phase-1a in-place reanchor arithmetic. The reference implementation.
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


# --- compat shim parity with a plain dict (removed in C6) -----------------

def test_shim_matches_dict_semantics():
    # tp2 present (LONG: tp2>tp1 ✓); conditional/observability fields absent.
    raw = _valid_block(tp2=110.0, tp2_exit_pct=50.0)
    mb = MachineBlock.from_raw(raw)
    assert mb is not None
    d = mb.to_raw()
    for k in ("entry_low", "direction", "tp2", "tp2_exit_pct", "leverage"):
        assert mb[k] == d[k]
        assert (k in mb) == (k in d)
        assert mb.get(k) == d.get(k)
    # absent optional must behave exactly like a missing dict key
    assert "confirming_event" not in mb
    assert mb.get("confirming_event") is None
    assert mb.get("expiry_candles", 12) == 12
    with pytest.raises(KeyError):
        _ = mb["confirming_event"]
    # unknown key absent
    assert "totally_unknown" not in mb


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
