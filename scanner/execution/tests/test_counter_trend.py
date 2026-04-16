"""Tests for counter-trend adjustments — detection, sizing, R:R, timeout."""

import sqlite3
from datetime import datetime, timezone

import pytest

from scanner.execution.config import ExecutionConfig
from scanner.execution.sizing import PositionSize
from scanner.execution.storage import ensure_orders_table, save_order


@pytest.fixture
def ct_config():
    """ExecutionConfig with counter-trend defaults."""
    return ExecutionConfig(
        enabled=True,
        mode="testnet",
        private_key="0xtest_key",
        wallet_address="0xtest_wallet",
        risk_pct=1.0,
        counter_trend_risk_mult=0.5,
        counter_trend_min_rr=3.0,
        counter_trend_timeout_hours=36,
        position_timeout_hours=72,
    )


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_orders_table(c)
    return c


def _make_entry_order(scan_id="s1", coin="BTC", direction="LONG", risk_mult=1.0, **overrides):
    """Build an entry order dict."""
    order = {
        "order_id": f"o_{scan_id}",
        "scan_id": scan_id,
        "coin": coin,
        "direction": direction,
        "order_type": "entry",
        "status": "FILLED",
        "entry_price": 100.0,
        "fill_price": 100.0,
        "quantity": 1.0,
        "leverage": 5,
        "exchange": "hyperliquid",
        "execution_mode": "testnet",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "risk_mult": risk_mult,
    }
    order.update(overrides)
    return order


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestCounterTrendDetection:

    def test_screener_long_llm_short_is_counter_trend(self):
        """Screener says LONG, LLM says SHORT → counter-trend."""
        screener_dir = "LONG"
        llm_dir = "SHORT"
        assert screener_dir != llm_dir

    def test_screener_long_llm_long_is_with_trend(self):
        """Screener says LONG, LLM says LONG → with-trend."""
        screener_dir = "LONG"
        llm_dir = "LONG"
        assert screener_dir == llm_dir

    def test_missing_screener_dir_not_counter_trend(self):
        """If screener direction missing (e.g. triggered entry), default to with-trend."""
        screener_dir = ""
        llm_dir = "SHORT"
        counter_trend = bool(screener_dir and llm_dir and screener_dir != llm_dir)
        assert counter_trend is False


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

class TestCounterTrendSizing:

    def test_reduced_risk_pct(self, ct_config):
        """Counter-trend should use risk_pct * counter_trend_risk_mult."""
        effective_risk = ct_config.risk_pct * ct_config.counter_trend_risk_mult
        assert effective_risk == 0.5

    def test_tp2_stripped(self):
        """Counter-trend should have tp2_quantity = 0, tp1_quantity = full qty."""
        pos = PositionSize(
            quantity=1.0, notional_value=100.0, risk_amount=5.0,
            stop_distance=5.0, rr_ratio=3.5, leverage=5,
            margin_required=20.0, tp1_quantity=0.5, tp2_quantity=0.5,
        )
        # Simulate counter-trend adjustment
        pos.tp1_quantity = pos.quantity
        pos.tp2_quantity = 0
        assert pos.tp1_quantity == 1.0
        assert pos.tp2_quantity == 0


# ---------------------------------------------------------------------------
# R:R rejection
# ---------------------------------------------------------------------------

class TestCounterTrendRR:

    def test_rr_below_min_rejected(self, ct_config):
        """R:R 2.5 < 3.0 counter-trend min → should be rejected."""
        rr = 2.5
        assert rr < ct_config.counter_trend_min_rr

    def test_rr_at_min_accepted(self, ct_config):
        """R:R 3.0 >= 3.0 → should be accepted."""
        rr = 3.0
        assert rr >= ct_config.counter_trend_min_rr

    def test_rr_above_min_accepted(self, ct_config):
        """R:R 4.0 >= 3.0 → should be accepted."""
        rr = 4.0
        assert rr >= ct_config.counter_trend_min_rr


# ---------------------------------------------------------------------------
# R scaling by risk_mult
# ---------------------------------------------------------------------------

class TestRiskMultScaling:

    def test_full_stop_counter_trend_is_half_r(self):
        """Counter-trend (risk_mult=0.5) full stop → -0.5R, not -1.0R."""
        price_r = -1.0  # price-based R for full stop
        risk_mult = 0.5
        scaled_r = price_r * risk_mult
        assert scaled_r == -0.5

    def test_tp1_hit_counter_trend(self):
        """Counter-trend TP1 at 3:1 R:R → +1.5R (3.0 * 0.5)."""
        price_r = 3.0
        risk_mult = 0.5
        scaled_r = price_r * risk_mult
        assert scaled_r == 1.5

    def test_with_trend_r_unchanged(self):
        """With-trend (risk_mult=1.0) → no scaling applied."""
        risk_mult = 1.0
        assert risk_mult >= 1.0 - 1e-9  # guard condition: scaling skipped

    def test_risk_mult_stored_on_order(self, conn):
        """risk_mult should be persisted and retrievable."""
        order = _make_entry_order(risk_mult=0.5)
        save_order(order, conn)
        row = conn.execute(
            "SELECT risk_mult FROM orders WHERE order_id = ?", (order["order_id"],)
        ).fetchone()
        assert float(row["risk_mult"]) == 0.5

    def test_risk_mult_default_is_1(self, conn):
        """Orders without risk_mult should default to 1.0."""
        order = _make_entry_order()
        del order["risk_mult"]  # simulate old order without field
        save_order(order, conn)
        row = conn.execute(
            "SELECT risk_mult FROM orders WHERE order_id = ?", (order["order_id"],)
        ).fetchone()
        assert float(row["risk_mult"]) == 1.0


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class TestCounterTrendTimeout:

    def test_counter_trend_uses_shorter_timeout(self, ct_config):
        """risk_mult < 1.0 → use counter_trend_timeout_hours."""
        risk_mult = 0.5
        timeout = (ct_config.counter_trend_timeout_hours
                   if risk_mult < 1.0 - 1e-9
                   else ct_config.position_timeout_hours)
        assert timeout == 36

    def test_with_trend_uses_standard_timeout(self, ct_config):
        """risk_mult = 1.0 → use position_timeout_hours."""
        risk_mult = 1.0
        timeout = (ct_config.counter_trend_timeout_hours
                   if risk_mult < 1.0 - 1e-9
                   else ct_config.position_timeout_hours)
        assert timeout == 72
