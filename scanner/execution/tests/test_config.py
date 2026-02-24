"""Tests for ExecutionConfig dataclass + loader."""

from unittest.mock import patch
import pytest

from scanner.execution.config import ExecutionConfig, load_execution_config


class TestExecutionConfig:
    def test_defaults_are_safe(self):
        cfg = ExecutionConfig()
        assert cfg.enabled is False
        assert cfg.mode == "testnet"
        assert cfg.risk_pct == 1.0
        assert cfg.private_key == ""
        assert cfg.max_concurrent_positions == 100

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid execution mode"):
            ExecutionConfig(mode="paper")

    def test_invalid_risk_pct_zero(self):
        with pytest.raises(ValueError, match="risk_pct"):
            ExecutionConfig(risk_pct=0)

    def test_invalid_risk_pct_negative(self):
        with pytest.raises(ValueError, match="risk_pct"):
            ExecutionConfig(risk_pct=-1)

    def test_invalid_risk_pct_over_100(self):
        with pytest.raises(ValueError, match="risk_pct"):
            ExecutionConfig(risk_pct=101)

    def test_valid_live_mode(self):
        cfg = ExecutionConfig(mode="live")
        assert cfg.mode == "live"


class TestLoadExecutionConfig:
    def test_loads_from_env(self):
        env = {
            "EXECUTION_ENABLED": "true",
            "EXECUTION_MODE": "live",
            "HL_PRIVATE_KEY": "0xkey",
            "HL_WALLET_ADDRESS": "0xwallet",
            "EXECUTION_RISK_PCT": "2.0",
            "EXECUTION_MAX_POSITIONS": "5",
            "EXECUTION_MAX_LONGS": "3",
            "EXECUTION_MAX_EXPOSURE_PCT": "50.0",
            "GCP_PROJECT_ID": "",
            "DISCORD_EXECUTION_WEBHOOK_URL": "https://discord.com/hook",
        }
        with patch.dict("os.environ", env, clear=False):
            with patch("scanner.execution.config.get_secret_or_env", side_effect=lambda s, e, p: env.get(e, "")):
                cfg = load_execution_config()

        assert cfg.enabled is True
        assert cfg.mode == "live"
        assert cfg.private_key == "0xkey"
        assert cfg.risk_pct == 2.0
        assert cfg.max_concurrent_positions == 5
        assert cfg.max_open_longs == 3

    def test_defaults_when_env_empty(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("scanner.execution.config.get_secret_or_env", return_value=""):
                cfg = load_execution_config()

        assert cfg.enabled is False
        assert cfg.mode == "testnet"
        assert cfg.private_key == ""

    def test_enabled_accepts_yes_and_1(self):
        for val in ("yes", "1", "YES", "True"):
            with patch.dict("os.environ", {"EXECUTION_ENABLED": val}, clear=False):
                with patch("scanner.execution.config.get_secret_or_env", return_value=""):
                    cfg = load_execution_config()
            assert cfg.enabled is True, f"Failed for EXECUTION_ENABLED={val}"
