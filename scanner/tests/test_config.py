"""Tests for ScannerConfig validation."""

import pytest

from scanner.config import EXCHANGE_META, ScannerConfig


class TestScannerConfigValidation:
    """__post_init__ validates exchange identifiers."""

    def test_default_config_valid(self):
        cfg = ScannerConfig()
        assert cfg.exchange == "hyperliquid"

    def test_valid_exchange(self):
        for ex in EXCHANGE_META:
            cfg = ScannerConfig(exchange=ex)
            assert cfg.exchange == ex

    def test_invalid_exchange_raises(self):
        with pytest.raises(ValueError, match="Unknown exchange: 'kraken'"):
            ScannerConfig(exchange="kraken")

    def test_default_setup_version_is_v1(self, monkeypatch):
        monkeypatch.delenv("SETUP_VERSION", raising=False)
        assert ScannerConfig().setup_version == "v1"

    def test_setup_version_v2_valid(self):
        assert ScannerConfig(setup_version="v2").setup_version == "v2"

    def test_invalid_setup_version_raises(self):
        with pytest.raises(ValueError, match="Unknown setup_version: 'v3'"):
            ScannerConfig(setup_version="v3")

    def test_setup_version_from_env(self, monkeypatch):
        monkeypatch.setenv("SETUP_VERSION", "v2")
        assert ScannerConfig().setup_version == "v2"
