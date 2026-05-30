"""`scripts/shadow-resolve.py` --exit-policy seam (T-020.11).

The seam is the money-path boundary that plan-review flagged: `--exit-policy` must reach
EVERY policy-keyed site identically (load / resolve / save / report) or a load-vs-save key
mismatch silently produces a false Gate-#10 sample. These tests load the hyphenated script
via importlib, stub its deps, and assert the policy string seen at each boundary.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "shadow-resolve.py"


def _load():
    """Fresh import of the hyphenated CLI script as a module."""
    spec = importlib.util.spec_from_file_location("shadow_resolve_cli", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _wire(mod):
    """Replace the CLI's module-level deps with stubs; return the wired module.

    One captured signal is returned so the per-signal resolve/save path is exercised
    (else `resolve_signal`/`save_resolution` would never be called and the seam unproven).
    """
    sig = SimpleNamespace(coin="BTC", exchange="hyperliquid", id=1,
                          machine_block=object(), anchor_ts=object())
    mod.get_connection = MagicMock(return_value=MagicMock())
    mod.load_unresolved_signals = MagicMock(return_value=[sig])
    mod.fetch_candles = MagicMock(return_value=object())
    mod.get_exchange_meta = MagicMock(return_value={"fee_rate": 0.0003})
    mod.resolve_signal = MagicMock(return_value=SimpleNamespace(status="RESOLVED"))
    mod.save_resolution = MagicMock()
    mod.resolved_r_values = MagicMock(return_value=[1.0, -0.5])
    mod.expectancy_report = MagicMock(return_value=SimpleNamespace(
        n=2, win_rate=0.5, avg_r=0.25, profit_factor=1.5, ci_lo=-0.1, ci_hi=0.6, gate_pass=False))
    return mod


def _run(mod, argv, capsys):
    old = sys.argv
    sys.argv = argv
    try:
        mod.main()
    finally:
        sys.argv = old
    return capsys.readouterr().out


def test_cli_threads_v2_to_every_policy_keyed_site(capsys):
    mod = _wire(_load())
    out = _run(mod, ["shadow-resolve.py", "--exit-policy", "v2"], capsys)
    # load_unresolved_signals(conn, exit_policy, entry_policy)
    assert mod.load_unresolved_signals.call_args.args[1] == "v2"
    # resolve_signal(..., exit_policy=...)
    assert mod.resolve_signal.call_args.kwargs["exit_policy"] == "v2"
    # save_resolution(conn, id, outcome, exit_policy, entry_policy, now)
    assert mod.save_resolution.call_args.args[3] == "v2"
    # resolved_r_values(conn, exit_policy, entry_policy)
    assert mod.resolved_r_values.call_args.args[1] == "v2"
    assert "exit=v2" in out


def test_cli_defaults_to_v1(capsys):
    mod = _wire(_load())
    out = _run(mod, ["shadow-resolve.py"], capsys)
    assert mod.load_unresolved_signals.call_args.args[1] == "v1"
    assert mod.resolve_signal.call_args.kwargs["exit_policy"] == "v1"
    assert mod.save_resolution.call_args.args[3] == "v1"
    assert mod.resolved_r_values.call_args.args[1] == "v1"
    assert "exit=v1" in out


def test_cli_rejects_unknown_policy(capsys):
    mod = _wire(_load())
    with pytest.raises(SystemExit):  # argparse `choices` rejects an unknown policy
        _run(mod, ["shadow-resolve.py", "--exit-policy", "v3"], capsys)
