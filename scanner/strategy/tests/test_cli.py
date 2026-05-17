"""Smoke tests for the surviving strategy CLI subcommands.

Covers argparse parsing + the cmd_xxx functions for create/list/activate/disable.
The replay/sweep/calibrate/report/backfill subcommands were removed in T-001;
they're not tested because they no longer exist.
"""

import pytest

from scanner.db import get_connection
from scanner.strategy.cli import build_parser, cmd_activate, cmd_create, cmd_disable, cmd_list
from scanner.strategy.store import get_strategy, save_strategy


@pytest.fixture
def patched_conn(monkeypatch, tmp_path):
    """Point cli.get_connection() at a tmp_path DB.

    The cmd_xxx functions own their connection (open + close per call), so
    the patched factory hands each invocation a fresh conn to the same file.
    The yielded `verify` conn is used by tests to read state back.
    """
    db_path = tmp_path / "test.db"
    # Initialize schema once
    get_connection(db_path).close()

    import scanner.strategy.cli as cli_mod
    monkeypatch.setattr(cli_mod, "get_connection", lambda: get_connection(db_path))

    verify = get_connection(db_path)
    yield verify
    verify.close()


class TestParser:
    """argparse build doesn't crash and surviving subcommands are reachable."""

    def test_create_parses(self):
        args = build_parser().parse_args(["create", "--label", "test", "--id", "v-test"])
        assert args.command == "create"
        assert args.label == "test"
        assert args.id == "v-test"
        assert args.swing_lookback == 5

    def test_list_parses(self):
        args = build_parser().parse_args(["list"])
        assert args.command == "list"

    def test_activate_parses(self):
        args = build_parser().parse_args(["activate", "v0"])
        assert args.command == "activate"
        assert args.strategy_id == "v0"

    def test_disable_parses_positional(self):
        args = build_parser().parse_args(["disable", "v0", "A,E"])
        assert args.command == "disable"
        assert args.strategy_id == "v0"
        assert args.types == "A,E"

    def test_disable_parses_none(self):
        args = build_parser().parse_args(["disable", "v0", "none"])
        assert args.types == "none"

    def test_deleted_subcommand_rejected(self):
        """replay/sweep/calibrate/report/backfill must no longer parse."""
        parser = build_parser()
        for cmd in ["replay", "sweep", "calibrate", "report", "backfill"]:
            with pytest.raises(SystemExit):
                parser.parse_args([cmd, "v0"])


class TestCmdCreate:
    def test_inserts_strategy(self, patched_conn, capsys):
        args = build_parser().parse_args([
            "create", "--label", "smoke", "--id", "v-smoke",
            "--atr-stop", "2.0", "--atr-target", "4.0",
        ])
        cmd_create(args)
        out = capsys.readouterr().out
        assert "v-smoke" in out
        got = get_strategy("v-smoke", patched_conn)
        assert got is not None
        assert got.label == "smoke"
        assert got.atr_stop_mult == 2.0
        assert got.atr_target_mult == 4.0


class TestCmdList:
    def test_runs_against_empty_db(self, patched_conn, capsys):
        """ensure_baseline plants v0 — so list should always show at least one row."""
        args = build_parser().parse_args(["list"])
        cmd_list(args)
        out = capsys.readouterr().out
        assert "v0" in out


class TestCmdActivate:
    def test_flips_is_active(self, patched_conn, baseline_strategy, custom_strategy):
        save_strategy(baseline_strategy, patched_conn)
        save_strategy(custom_strategy, patched_conn)
        args = build_parser().parse_args(["activate", custom_strategy.strategy_id])
        cmd_activate(args)
        got = get_strategy(custom_strategy.strategy_id, patched_conn)
        assert got.is_active

    def test_missing_strategy_exits(self, patched_conn):
        args = build_parser().parse_args(["activate", "does-not-exist"])
        with pytest.raises(SystemExit):
            cmd_activate(args)


class TestCmdDisable:
    def test_sets_disabled_types(self, patched_conn, baseline_strategy):
        save_strategy(baseline_strategy, patched_conn)
        args = build_parser().parse_args(["disable", "v0", "A,E"])
        cmd_disable(args)
        got = get_strategy("v0", patched_conn)
        assert got.disabled_types == "A,E"

    def test_normalizes_case_and_whitespace(self, patched_conn, baseline_strategy):
        save_strategy(baseline_strategy, patched_conn)
        args = build_parser().parse_args(["disable", "v0", " a, e , f "])
        cmd_disable(args)
        got = get_strategy("v0", patched_conn)
        assert got.disabled_types == "A,E,F"

    def test_none_clears(self, patched_conn, baseline_strategy):
        baseline_strategy.disabled_types = "A,E"
        save_strategy(baseline_strategy, patched_conn)
        args = build_parser().parse_args(["disable", "v0", "none"])
        cmd_disable(args)
        got = get_strategy("v0", patched_conn)
        assert got.disabled_types == ""

    def test_missing_strategy_exits(self, patched_conn):
        args = build_parser().parse_args(["disable", "does-not-exist", "A"])
        with pytest.raises(SystemExit):
            cmd_disable(args)
