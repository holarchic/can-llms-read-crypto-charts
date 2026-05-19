"""Smoke tests for the strategy CLI — `list` + `disable`.

The subsystem is cut to `disabled_types` (decision #31): create/activate and
the param-set surface no longer exist.
"""

import pytest

from scanner.db import get_connection
from scanner.strategy.cli import build_parser, cmd_disable, cmd_list
from scanner.strategy.store import ensure_baseline, get_strategy, set_disabled_types


@pytest.fixture
def patched_conn(monkeypatch, tmp_path):
    """Point cli.get_connection() at a tmp_path DB.

    The cmd_xxx functions own their connection (open + close per call), so
    the patched factory hands each invocation a fresh conn to the same file.
    The yielded `verify` conn is used by tests to read state back.
    """
    db_path = tmp_path / "test.db"
    get_connection(db_path).close()  # initialise schema once

    import scanner.strategy.cli as cli_mod
    monkeypatch.setattr(cli_mod, "get_connection", lambda: get_connection(db_path))

    verify = get_connection(db_path)
    yield verify
    verify.close()


class TestParser:
    def test_list_parses(self):
        assert build_parser().parse_args(["list"]).command == "list"

    def test_disable_parses_positional(self):
        args = build_parser().parse_args(["disable", "v0", "A,E"])
        assert args.command == "disable"
        assert args.strategy_id == "v0"
        assert args.types == "A,E"

    def test_disable_parses_none(self):
        assert build_parser().parse_args(["disable", "v0", "none"]).types == "none"

    def test_removed_subcommands_rejected(self):
        """create/activate (and the long-gone replay/sweep/...) must not parse."""
        parser = build_parser()
        for cmd in ["create", "activate", "replay", "sweep", "calibrate", "report", "backfill"]:
            with pytest.raises(SystemExit):
                parser.parse_args([cmd, "v0"])


class TestCmdList:
    def test_ensure_baseline_plants_v0(self, patched_conn, capsys):
        cmd_list(build_parser().parse_args(["list"]))
        assert "v0" in capsys.readouterr().out


class TestCmdDisable:
    def test_sets_disabled_types(self, patched_conn):
        ensure_baseline(patched_conn)
        cmd_disable(build_parser().parse_args(["disable", "v0", "A,E"]))
        assert get_strategy("v0", patched_conn).disabled_types == "A,E"

    def test_normalizes_case_and_whitespace(self, patched_conn):
        ensure_baseline(patched_conn)
        cmd_disable(build_parser().parse_args(["disable", "v0", " a, e , f "]))
        assert get_strategy("v0", patched_conn).disabled_types == "A,E,F"

    def test_none_clears(self, patched_conn):
        ensure_baseline(patched_conn)
        set_disabled_types("v0", "A,E", patched_conn)
        cmd_disable(build_parser().parse_args(["disable", "v0", "none"]))
        assert get_strategy("v0", patched_conn).disabled_types == ""

    def test_missing_strategy_exits(self, patched_conn):
        with pytest.raises(SystemExit):
            cmd_disable(build_parser().parse_args(["disable", "does-not-exist", "A"]))
