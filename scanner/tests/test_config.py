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
