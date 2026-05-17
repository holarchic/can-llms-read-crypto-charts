"""Regression for the `smoke` execution subcommand.

T-016 phase-review (correctness R3): `_cmd_smoke()` passed a raw dict into
the now-typed `validate_execution_params` and raised
`AttributeError: 'dict' object has no attribute 'direction'` before
preflight — a real broken CLI path (`python -m scanner.execution smoke`)
that no test covered. This pins it.
"""

from scanner.execution.__main__ import _cmd_smoke
from scanner.execution.config import ExecutionConfig


def test_smoke_runs_without_typed_machine_block_crash(capsys):
    # enabled=False → preflight skipped (no exchange creds needed); the bug
    # was upstream of preflight, at the param-validation step.
    _cmd_smoke(ExecutionConfig(enabled=False))
    out = capsys.readouterr().out
    assert "Param validation: OK" in out
    assert "Smoke: ALL PASS" in out
