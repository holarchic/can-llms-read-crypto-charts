"""Tests for multi-exchange support — config, metadata, data layer, storage."""

from unittest.mock import MagicMock, patch

import pytest

from scanner.config import EXCHANGE_META, ScannerConfig, get_exchange_meta
from scanner.db import get_connection
from scanner.scan_queries import is_duplicate

# --- Config / metadata ---


class TestExchangeMeta:
    def test_all_three_exchanges_defined(self):
        assert "hyperliquid" in EXCHANGE_META
        assert "bybit" in EXCHANGE_META
        assert "binance" in EXCHANGE_META

    def test_meta_has_required_keys(self):
        for eid, meta in EXCHANGE_META.items():
            assert "ccxt_class" in meta, f"{eid} missing ccxt_class"
            assert "symbol_template" in meta, f"{eid} missing symbol_template"
            assert "fee_rate" in meta, f"{eid} missing fee_rate"
            assert "label" in meta, f"{eid} missing label"

    def test_symbol_formatting(self):
        assert EXCHANGE_META["hyperliquid"]["symbol_template"].format(coin="BTC") == "BTC/USDC:USDC"
        assert EXCHANGE_META["bybit"]["symbol_template"].format(coin="ETH") == "ETH/USDT:USDT"
        assert EXCHANGE_META["binance"]["symbol_template"].format(coin="SOL") == "SOL/USDT:USDT"

    def test_get_exchange_meta_valid(self):
        meta = get_exchange_meta("bybit")
        assert meta["ccxt_class"] == "bybit"
        assert meta["label"] == "Bybit"

    def test_get_exchange_meta_invalid(self):
        with pytest.raises(ValueError, match="Unknown exchange"):
            get_exchange_meta("kraken")

    def test_config_default_exchange(self):
        cfg = ScannerConfig()
        assert cfg.exchange == "hyperliquid"

    def test_config_custom_exchange(self):
        cfg = ScannerConfig(exchange="binance")
        assert cfg.exchange == "binance"


# --- Data layer ---


class TestExchangeSingleton:
    @patch("scanner.data.ccxt")
    def test_get_exchange_caches_instance(self, mock_ccxt):
        import scanner.data as data_mod

        # Clear singleton cache
        data_mod._exchanges = {}

        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_ccxt.hyperliquid = mock_cls

        ex1 = data_mod.get_exchange("hyperliquid")
        ex2 = data_mod.get_exchange("hyperliquid")

        assert ex1 is ex2
        mock_cls.assert_called_once()

        # Cleanup
        data_mod._exchanges = {}

    @patch("scanner.data.ccxt")
    def test_get_exchange_different_exchanges(self, mock_ccxt):
        import scanner.data as data_mod

        data_mod._exchanges = {}

        mock_hl = MagicMock()
        mock_bybit = MagicMock()
        mock_ccxt.hyperliquid = MagicMock(return_value=mock_hl)
        mock_ccxt.bybit = MagicMock(return_value=mock_bybit)

        ex_hl = data_mod.get_exchange("hyperliquid")
        ex_bb = data_mod.get_exchange("bybit")

        assert ex_hl is not ex_bb

        # Cleanup
        data_mod._exchanges = {}


# --- Storage ---


@pytest.fixture
def db_conn(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    yield conn
    conn.close()


class TestExchangeScopedDedup:
    def test_dedup_scoped_to_exchange(self, db_conn):
        """Duplicate check should be exchange-scoped."""
        # Insert a scan for hyperliquid
        db_conn.execute(
            "INSERT INTO scans (scan_id, coin, timestamp, setup_type, screener_direction, exchange, created_at) "
            "VALUES ('s1', 'BTC', datetime('now'), 'C', 'LONG', 'hyperliquid', datetime('now'))"
        )
        db_conn.commit()

        # Should be duplicate on hyperliquid
        assert is_duplicate("BTC", "C", "LONG", 12, db_conn, exchange="hyperliquid")

        # Should NOT be duplicate on bybit
        assert not is_duplicate("BTC", "C", "LONG", 12, db_conn, exchange="bybit")


class TestExchangeColumnOnTables:
    def test_signals_has_exchange(self, db_conn):
        cols = {r[1] for r in db_conn.execute("PRAGMA table_info(signals)").fetchall()}
        assert "exchange" in cols

    def test_scans_has_exchange(self, db_conn):
        cols = {r[1] for r in db_conn.execute("PRAGMA table_info(scans)").fetchall()}
        assert "exchange" in cols

    def test_scan_universe_has_exchange(self, db_conn):
        cols = {r[1] for r in db_conn.execute("PRAGMA table_info(scan_universe)").fetchall()}
        assert "exchange" in cols
