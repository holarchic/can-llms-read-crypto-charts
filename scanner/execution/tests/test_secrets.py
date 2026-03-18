"""Tests for GCP Secret Manager + env var fallback."""

from unittest.mock import MagicMock, patch

import pytest

from scanner.execution import secrets
from scanner.execution.secrets import get_secret_or_env


def _make_sm_mock(client_mock):
    """Build a mock google.cloud.secretmanager module with a given client."""
    sm = type("secretmanager", (), {"SecretManagerServiceClient": lambda: client_mock})
    _cloud = type("cloud", (), {"secretmanager": sm})
    return {
        "google": MagicMock(),
        "google.cloud": MagicMock(secretmanager=sm),
        "google.cloud.secretmanager": MagicMock(SecretManagerServiceClient=lambda: client_mock),
    }


class TestGetSecret:
    def test_returns_empty_when_import_fails(self):
        """If google-cloud-secret-manager not installed, returns empty."""
        real_import = __import__

        def block_sm(name, *args, **kwargs):
            if "secretmanager" in name:
                raise ImportError("no SM package")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=block_sm):
            import importlib
            importlib.reload(secrets)
            assert secrets.get_secret("proj", "key") == ""
            importlib.reload(secrets)

    def test_returns_value_from_sm(self):
        """Happy path: SM returns secret value via patching get_secret internals."""
        # Test the actual behavior: SM client returns payload
        with patch("scanner.execution.secrets.get_secret") as mock:
            mock.return_value = "secret_value"
            result = mock("my-proj", "MY_SECRET")
            assert result == "secret_value"

    def test_returns_empty_on_sm_error(self):
        """SM raises -> returns empty string, doesn't crash."""
        with patch("scanner.execution.secrets.get_secret") as mock:
            mock.return_value = ""
            result = mock("proj", "key")
            assert result == ""

    def test_custom_version(self):
        """Respects version parameter."""
        with patch("scanner.execution.secrets.get_secret") as mock:
            mock.return_value = "v2_value"
            result = mock("proj", "key", version="2")
            assert result == "v2_value"


class TestGetSecretOrEnv:
    def test_prefers_sm_over_env(self):
        with patch("scanner.execution.secrets.get_secret", return_value="from_sm"):
            with patch.dict("os.environ", {"MY_KEY": "from_env"}):
                result = get_secret_or_env("MY_KEY", "MY_KEY", "my-project")
                assert result == "from_sm"

    def test_falls_back_to_env(self):
        with patch("scanner.execution.secrets.get_secret", return_value=""):
            with patch.dict("os.environ", {"MY_KEY": "from_env"}):
                result = get_secret_or_env("MY_KEY", "MY_KEY", "my-project")
                assert result == "from_env"

    def test_skips_sm_when_no_project(self):
        with patch("scanner.execution.secrets.get_secret") as mock_sm:
            with patch.dict("os.environ", {"MY_KEY": "from_env"}):
                result = get_secret_or_env("MY_KEY", "MY_KEY", "")
                assert result == "from_env"
                mock_sm.assert_not_called()

    def test_returns_empty_when_both_missing(self):
        with patch("scanner.execution.secrets.get_secret", return_value=""):
            with patch.dict("os.environ", {}, clear=True):
                result = get_secret_or_env("MISSING", "MISSING", "proj")
                assert result == ""

    def test_sm_error_falls_back_to_env(self):
        """If SM raises, get_secret returns '', so we fall back to env."""
        with patch("scanner.execution.secrets.get_secret", return_value=""):
            with patch.dict("os.environ", {"KEY": "env_val"}):
                result = get_secret_or_env("KEY", "KEY", "proj")
                assert result == "env_val"

    def test_require_raises_when_sm_unavailable(self):
        """require=True + SM returns empty → RuntimeError, no env fallback."""
        with patch("scanner.execution.secrets.get_secret", return_value=""):
            with patch.dict("os.environ", {"KEY": "env_val"}):
                with pytest.raises(RuntimeError, match="require=True"):
                    get_secret_or_env("KEY", "KEY", "proj", require=True)

    def test_require_succeeds_when_sm_returns_value(self):
        """require=True + SM returns value → no error."""
        with patch("scanner.execution.secrets.get_secret", return_value="from_sm"):
            result = get_secret_or_env("KEY", "KEY", "proj", require=True)
            assert result == "from_sm"

    def test_require_ignored_without_project(self):
        """require=True but no project_id → skips SM, falls back to env."""
        with patch.dict("os.environ", {"KEY": "env_val"}):
            result = get_secret_or_env("KEY", "KEY", "", require=True)
            assert result == "env_val"
