"""scripts/shadow-resolve.py CLI render contract (T-020.12).

`main()` is now argparse-dispatch over `run_resolution`; the money-path policy threading is
tested directly in `test_shadow_runner.py`. This only pins that the script still passes the
parsed `--exit-policy` through AND renders the returned `(report, by_status)` readout — so the
extraction can't silently drop printing, swap the tuple order, or omit the header (codex r3).
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "shadow-resolve.py"


def _load():
    spec = importlib.util.spec_from_file_location("shadow_resolve_cli", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(mod, argv, capsys):
    report = SimpleNamespace(n=7, win_rate=0.5, avg_r=0.25, profit_factor=1.5,
                             ci_lo=-0.1, ci_hi=0.6, gate_pass=False)
    by_status = {"RESOLVED": 7, "OPEN": 2}
    mod.get_connection = MagicMock(return_value=MagicMock())
    mod.run_resolution = MagicMock(return_value=(report, by_status))
    old = sys.argv
    sys.argv = argv
    try:
        mod.main()
    finally:
        sys.argv = old
    return mod.run_resolution, capsys.readouterr().out


def test_cli_renders_report_and_breakdown_keyed_by_exit_policy(capsys):
    mod = _load()
    run, out = _run(mod, ["shadow-resolve.py", "--exit-policy", "v2"], capsys)
    assert run.call_args.kwargs["exit_policy"] == "v2"   # parsed policy reaches run_resolution
    assert "RESOLVED" in out and "OPEN" in out           # by_status breakdown rendered
    assert "exit=v2" in out                              # header keyed by exit_policy
    assert "n=7" in out and "GATE #10" in out            # report fields + gate line rendered


def test_cli_defaults_exit_policy_to_v1(capsys):
    mod = _load()
    run, out = _run(mod, ["shadow-resolve.py"], capsys)
    assert run.call_args.kwargs["exit_policy"] == "v1"
    assert "exit=v1" in out
