"""Tests for remote kill switch."""

from unittest.mock import MagicMock, patch

import pytest

from scanner.execution.config import ExecutionConfig
from scanner.execution.kill_switch import is_killed, set_kill_switch


@pytest.fixture
def config():
    return ExecutionConfig(gcp_project_id="test-project")


@pytest.fixture
def config_no_gcp():
    return ExecutionConfig(gcp_project_id="")


@pytest.fixture
def live_config():
    return ExecutionConfig(gcp_project_id="test-project", mode="live")


class TestIsKilled:
    def test_true_from_sm(self, config):
        with patch("scanner.execution.kill_switch.get_secret", return_value="true"):
            assert is_killed(config) is True

    def test_false_from_sm(self, config):
        with patch("scanner.execution.kill_switch.get_secret", return_value="false"):
            assert is_killed(config) is False

    def test_env_fallback(self, config):
        with patch("scanner.execution.kill_switch.get_secret", return_value=""):
            with patch.dict("os.environ", {"KILL_SWITCH": "true"}):
                assert is_killed(config) is True

    def test_error_returns_false(self, config):
        """Fail-open: SM error means NOT killed."""
        with patch("scanner.execution.kill_switch.get_secret", side_effect=Exception("SM down")):
            with patch.dict("os.environ", {"KILL_SWITCH": "false"}):
                assert is_killed(config) is False

    @pytest.mark.parametrize("val,expected", [
        ("true", True), ("TRUE", True), ("True", True),
        ("1", True), ("yes", True), ("YES", True),
        ("false", False), ("0", False), ("no", False),
        ("", False),
    ])
    def test_truthy_variants(self, config_no_gcp, val, expected):
        """Test parsing of various truthy/falsy values via env fallback."""
        with patch.dict("os.environ", {"KILL_SWITCH": val}):
            assert is_killed(config_no_gcp) is expected

    def test_no_gcp_uses_env(self, config_no_gcp):
        with patch.dict("os.environ", {"KILL_SWITCH": "true"}):
            assert is_killed(config_no_gcp) is True


class TestFailClosed:
    """Live mode fails closed — SM error or missing value → killed."""

    def test_live_mode_sm_error_fails_closed(self, live_config):
        """SM raises in live mode → return True (killed)."""
        with patch("scanner.execution.kill_switch.get_secret", side_effect=Exception("SM down")):
            with patch.dict("os.environ", {}, clear=True):
                assert is_killed(live_config) is True

    def test_live_mode_no_value_fails_closed(self, live_config):
        """SM returns empty, no env var, live → True (killed)."""
        with patch("scanner.execution.kill_switch.get_secret", return_value=""):
            with patch.dict("os.environ", {}, clear=True):
                assert is_killed(live_config) is True

    def test_live_mode_env_false_overrides(self, live_config):
        """Explicit KILL_SWITCH=false in live → False (operator override)."""
        with patch("scanner.execution.kill_switch.get_secret", return_value=""):
            with patch.dict("os.environ", {"KILL_SWITCH": "false"}):
                assert is_killed(live_config) is False

    def test_testnet_sm_error_still_fails_open(self, config):
        """SM raises in testnet → False (unchanged behavior)."""
        with patch("scanner.execution.kill_switch.get_secret", side_effect=Exception("SM down")):
            with patch.dict("os.environ", {"KILL_SWITCH": "false"}):
                assert is_killed(config) is False


class TestSetKillSwitch:
    def test_set_active_calls_sm(self, config):
        """Verify set_kill_switch calls SM to add a secret version."""
        mock_client = MagicMock()

        with patch("scanner.execution.kill_switch._get_sm_client", return_value=mock_client):
            set_kill_switch(True, config)
            mock_client.add_secret_version.assert_called_once()
            call_kwargs = mock_client.add_secret_version.call_args
            assert call_kwargs[1]["request"]["payload"]["data"] == b"true"

    def test_set_deactivate(self, config):
        """Verify deactivation sends false payload."""
        mock_client = MagicMock()

        with patch("scanner.execution.kill_switch._get_sm_client", return_value=mock_client):
            set_kill_switch(False, config)
            call_kwargs = mock_client.add_secret_version.call_args
            assert call_kwargs[1]["request"]["payload"]["data"] == b"false"

    def test_no_gcp_project_skips(self, config_no_gcp):
        """No project ID -> logs error, doesn't crash."""
        with patch("scanner.execution.kill_switch.log") as mock_log:
            set_kill_switch(True, config_no_gcp)
            mock_log.error.assert_called()
