"""Tests for chart generation."""


import pytest

from scanner.charts import cleanup_charts, generate_chart
from scanner.indicators import build_context
from scanner.models import Direction, SetupSignal, SetupType


@pytest.fixture
def tmp_charts(tmp_path):
    return tmp_path / "charts"


class TestGenerateChart:
    def test_creates_png(self, ranging_df, tmp_charts):
        ctx = build_context(ranging_df)
        signal = SetupSignal(
            coin="TEST",
            setup_type=SetupType.A,
            direction=Direction.LONG,
            confidence=0.6,
            key_level=100.0,
            context=ctx,
            description="Test signal",
        )
        path = generate_chart(ranging_df, "TEST", "4H", signal, output_dir=tmp_charts)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 1000  # Non-trivial file

    def test_filename_contains_coin_and_tf(self, ranging_df, tmp_charts):
        ctx = build_context(ranging_df)
        signal = SetupSignal(
            coin="ETH",
            setup_type=SetupType.C,
            direction=Direction.SHORT,
            confidence=0.5,
            key_level=2000.0,
            context=ctx,
            description="Test",
        )
        path = generate_chart(ranging_df, "ETH", "1H", signal, output_dir=tmp_charts)
        assert "ETH" in path.name
        assert "1H" in path.name


class TestCleanupCharts:
    def test_removes_pngs(self, tmp_charts):
        tmp_charts.mkdir(parents=True)
        (tmp_charts / "test1.png").write_bytes(b"fake")
        (tmp_charts / "test2.png").write_bytes(b"fake")
        count = cleanup_charts(tmp_charts)
        assert count == 2
        assert len(list(tmp_charts.glob("*.png"))) == 0

    def test_returns_zero_for_empty(self, tmp_path):
        count = cleanup_charts(tmp_path / "nonexistent")
        assert count == 0
