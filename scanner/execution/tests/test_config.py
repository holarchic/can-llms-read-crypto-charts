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

    def test_enabled_without_private_key_raises(self):
        with pytest.raises(ValueError, match="private_key"):
            ExecutionConfig(enabled=True, private_key="", wallet_address="0xwallet")

    def test_enabled_without_wallet_address_raises(self):
        with pytest.raises(ValueError, match="wallet_address"):
            ExecutionConfig(enabled=True, private_key="0xkey", wallet_address="")

    def test_enabled_with_credentials_ok(self):
        cfg = ExecutionConfig(enabled=True, private_key="0xkey", wallet_address="0xwallet")
        assert cfg.enabled is True

    def test_max_open_orders_default(self):
        cfg = ExecutionConfig()
        assert cfg.max_open_orders == 200

    def test_max_open_orders_disabled(self):
        cfg = ExecutionConfig(max_open_orders=0)
        assert cfg.max_open_orders == 0


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
            "DISCORD_OPS_WEBHOOK_URL": "https://discord.com/ops",
            "DISCORD_URGENT_WEBHOOK_URL": "https://discord.com/urgent",
            "DISCORD_ERRORS_WEBHOOK_URL": "https://discord.com/errors",
        }
        with patch.dict("os.environ", env, clear=False):
            with patch("scanner.execution.config.get_secret_or_env", side_effect=lambda s, e, p, **kw: env.get(e, "")):
                cfg = load_execution_config()

        assert cfg.enabled is True
        assert cfg.mode == "live"
        assert cfg.private_key == "0xkey"
        assert cfg.risk_pct == 2.0
        assert cfg.max_concurrent_positions == 5
        assert cfg.max_open_longs == 3
        assert cfg.discord_ops_webhook_url == "https://discord.com/ops"
        assert cfg.discord_urgent_webhook_url == "https://discord.com/urgent"
        assert cfg.discord_errors_webhook_url == "https://discord.com/errors"

    def test_defaults_when_env_empty(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("scanner.execution.config.get_secret_or_env", return_value=""):
                cfg = load_execution_config()

        assert cfg.enabled is False
        assert cfg.mode == "testnet"
        assert cfg.private_key == ""
        assert cfg.immediate_freshness_bars == 2  # the dataclass default flows through the loader

    def test_immediate_freshness_bars_env_override(self):
        # the 0 = off kill-switch (T-020.10 / #49) must be reachable from the deployed loader,
        # not just the dataclass — env override wired like the other tuning knobs.
        with patch.dict("os.environ", {"EXECUTION_IMMEDIATE_FRESHNESS_BARS": "0"}, clear=True):
            with patch("scanner.execution.config.get_secret_or_env", return_value=""):
                cfg = load_execution_config()
        assert cfg.immediate_freshness_bars == 0

    def test_enabled_accepts_yes_and_1(self):
        for val in ("yes", "1", "YES", "True"):
            with patch.dict("os.environ", {"EXECUTION_ENABLED": val}, clear=False):
                with patch("scanner.execution.config.get_secret_or_env", return_value="0xtest"):
                    cfg = load_execution_config()
            assert cfg.enabled is True, f"Failed for EXECUTION_ENABLED={val}"

    def test_live_mode_requires_sm_credentials(self):
        """Live mode + GCP project → SM failure raises RuntimeError."""
        env = {
            "EXECUTION_MODE": "live",
            "GCP_PROJECT_ID": "perpstrader",
            "HL_PRIVATE_KEY": "env_fallback",
        }
        with patch.dict("os.environ", env, clear=False):
            with patch("scanner.execution.secrets.get_secret", return_value=""):
                with pytest.raises(RuntimeError, match="require=True"):
                    load_execution_config()

    def test_testnet_mode_falls_back_to_env(self):
        """Testnet mode → SM failure gracefully falls back to env vars."""
        env = {
            "EXECUTION_MODE": "testnet",
            "GCP_PROJECT_ID": "perpstrader",
            "HL_PRIVATE_KEY": "0xfallback",
            "HL_WALLET_ADDRESS": "0xwallet",
        }
        with patch.dict("os.environ", env, clear=False):
            with patch("scanner.execution.secrets.get_secret", return_value=""):
                cfg = load_execution_config()
        assert cfg.private_key == "0xfallback"
        assert cfg.wallet_address == "0xwallet"
