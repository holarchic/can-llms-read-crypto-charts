"""scanner/shadow/runner.py::run_resolution money-path seam (T-020.12).

The single `exit_policy` argument must reach ALL FOUR policy-keyed calls
(`load_unresolved_signals` / `resolve_signal` / `save_resolution` / `resolved_r_values`) and
`setup_tf` must ride from the loaded signal into the resolver — else a cross-keyed tape would
produce a false Gate-#10 sample. Patched at the runner module (a real package module — no
importlib of the hyphenated script), so the seam is asserted directly, not via downstream output.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from scanner.shadow import runner


def _sig(coin="BTC", setup_tf="4h"):
    return SimpleNamespace(coin=coin, exchange="hyperliquid", id=1,
                           machine_block=object(), anchor_ts=object(), setup_tf=setup_tf)


def _run(exit_policy):
    now = datetime(2026, 5, 29, tzinfo=timezone.utc)
    fetch = MagicMock(return_value=object())
    with patch.object(runner, "load_unresolved_signals", return_value=[_sig()]) as load, \
         patch.object(runner, "resolve_signal", return_value=SimpleNamespace(status="RESOLVED")) as resolve, \
         patch.object(runner, "save_resolution") as save, \
         patch.object(runner, "resolved_r_values", return_value=[1.0, -0.5]) as rvals, \
         patch.object(runner, "get_exchange_meta", return_value={"fee_rate": 0.0003}), \
         patch.object(runner, "expectancy_report", return_value=SimpleNamespace(n=2)):
        result = runner.run_resolution(
            MagicMock(), exit_policy=exit_policy, bars=400, fetch=fetch, now=now, log=lambda *a: None)
    return load, resolve, save, rvals, result


def test_run_resolution_threads_policy_to_all_four_keyed_calls():
    load, resolve, save, rvals, result = _run("v2")
    assert load.call_args.args[1] == "v2"               # load_unresolved_signals(conn, exit, entry)
    assert resolve.call_args.kwargs["exit_policy"] == "v2"
    assert save.call_args.args[3] == "v2"               # save_resolution(conn, id, outcome, exit, entry, now)
    assert rvals.call_args.args[1] == "v2"              # resolved_r_values(conn, exit, entry)
    assert result.by_status == {"RESOLVED": 1}          # named record, not a positional tuple
    assert result.report.n == 2


def test_run_resolution_threads_setup_tf_into_resolver():
    _, resolve, _, _, _ = _run("v2")
    assert resolve.call_args.kwargs["setup_tf"] == "4h"


def test_run_resolution_v1_keys_every_call_v1():
    load, resolve, save, rvals, _ = _run("v1")
    assert load.call_args.args[1] == "v1"
    assert resolve.call_args.kwargs["exit_policy"] == "v1"
    assert save.call_args.args[3] == "v1"
    assert rvals.call_args.args[1] == "v1"
