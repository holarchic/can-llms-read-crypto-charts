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

    def test_invalid_candle_exchanges_raises(self):
        with pytest.raises(ValueError, match="Unknown candle exchange: 'ftx'"):
            ScannerConfig(candle_exchanges=["hyperliquid", "ftx"])

    def test_valid_candle_exchanges(self):
        cfg = ScannerConfig(candle_exchanges=["hyperliquid", "bybit"])
        assert cfg.candle_exchanges == ["hyperliquid", "bybit"]

    def test_empty_candle_exchanges_valid(self):
        cfg = ScannerConfig(candle_exchanges=[])
        assert cfg.candle_exchanges == []
