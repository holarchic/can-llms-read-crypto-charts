"""Tests for scanner/data_plane_v2.py (T-019 Chunk 7a, v2 multi-TF data plane).

Synthetic, deterministic, fixed-date fixtures (never ``Timestamp.now()``); an
injectable recording fetcher keeps ``fetch_multi_tf`` off the network. Substrates
are built through the REAL Chunk-1/2/4 functions, so the assembly is exercised
end-to-end, not stubbed.
"""

import time

import pandas as pd

from scanner.config import ScannerConfig
from scanner.data_plane_v2 import (
    CANONICAL_TFS,
    MIN_SUBSTRATE_BARS,
    SETUP_TFS,
    V2_CANDLE_COUNTS,
    MarketContextV2,
    build_market_context_v2,
    build_tf_substrate,
    fetch_multi_tf,
    htf_ema_for_chart,
    resample_ohlcv,
)
from scanner.models import SRLevelV2
from scanner.regime import Regime
from scanner.trend_state import TrendDirection

# --- builders ------------------------------------------------------------------


def _frame(closes, vols=None, wick=0.3, start="2026-01-01", freq="D") -> pd.DataFrame:
    """OHLCV frame with ``Open == Close`` (clean body points) and symmetric wicks."""
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq=freq)
    close = pd.Series([float(c) for c in closes], index=idx)
    vol = pd.Series([float(v) for v in vols], index=idx) if vols is not None else pd.Series(1000.0, index=idx)
    return pd.DataFrame(
        {"Open": close, "High": close + wick, "Low": close - wick, "Close": close, "Volume": vol},
        index=idx,
    )


def _ohlc_frame(rows, start="2026-01-01", freq="D") -> pd.DataFrame:
    """Frame from explicit (open, high, low, close, volume) rows — distinct O/H/L/C/V."""
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(
        {
            "Open": [r[0] for r in rows],
            "High": [r[1] for r in rows],
            "Low": [r[2] for r in rows],
            "Close": [r[3] for r in rows],
            "Volume": [r[4] for r in rows],
        },
        index=idx,
    )


def _zig(n, lo, hi, half):
    """Triangle wave of ``n`` closes between ``lo`` and ``hi``, ``half``-bar half-period."""
    vals, pos, step, d = [], float(lo), (hi - lo) / half, 1
    for _ in range(n):
        vals.append(pos)
        pos += d * step
        if pos >= hi:
            pos, d = float(hi), -1
        elif pos <= lo:
            pos, d = float(lo), 1
    return vals


def _deep_uptrend(n=400) -> pd.DataFrame:
    """A steady uptrend deep enough for a non-degenerate 100/200/300 longer stack."""
    return _frame([100 + i * 0.5 for i in range(n)])


def _realistic_candles() -> dict[str, pd.DataFrame]:
    """A 7-TF candle dict: deep trending setup TFs + zig macro TFs with clear pivots."""
    deep = _deep_uptrend(400)
    return {
        "1mo": _frame(_zig(36, 80, 120, 5)),
        "1w": _frame(_zig(104, 90, 110, 6)),
        "3d": _frame(_zig(130, 95, 115, 7)),
        "1d": deep,
        "4h": deep,
        "1h": deep,
        "15m": deep,
    }


# --- 1. resample_ohlcv ---------------------------------------------------------


def test_resample_ohlcv_1d_to_3d():
    rows = [
        (10, 15, 8, 11, 100), (11, 16, 7, 12, 200), (12, 14, 9, 13, 300),       # bucket0 [01,02,03]
        (20, 25, 18, 21, 400), (21, 26, 17, 22, 500), (22, 24, 19, 23, 600),    # bucket1 [04,05,06]
        (30, 35, 28, 31, 700), (31, 36, 27, 32, 800), (32, 34, 29, 33, 900),    # bucket2 [07,08,09]
        (40, 45, 38, 41, 1000),                                                 # bucket3 [10] partial
    ]
    df = _ohlc_frame(rows)

    full = resample_ohlcv(df, "3D", count=99, period_bars=3)
    # incomplete right-edge bucket3 (1 constituent < 3) dropped → 3 full buckets
    assert len(full) == 3
    assert pd.Timestamp("2026-01-10") not in full.index
    # open-time bucketing: labels are the left (open-time) edges
    assert list(full.index) == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-04"), pd.Timestamp("2026-01-07")]
    # bucket0 hand-check: O=first, H=max, L=min, C=last, V=sum
    b0 = full.iloc[0]
    assert (b0.Open, b0.High, b0.Low, b0.Close, b0.Volume) == (10, 16, 7, 13, 600)

    # exact output length == the trim count (MEDIUM-4), keeping the last two full buckets
    out = resample_ohlcv(df, "3D", count=2, period_bars=3)
    assert len(out) == 2
    b1, b2 = out.iloc[0], out.iloc[1]
    assert (b1.Open, b1.High, b1.Low, b1.Close, b1.Volume) == (20, 26, 17, 23, 1500)
    assert (b2.Open, b2.High, b2.Low, b2.Close, b2.Volume) == (30, 36, 27, 33, 2400)


# --- 2. fetch_multi_tf ---------------------------------------------------------

_FREQ = {"1M": "30D", "1w": "7D", "1d": "D", "4h": "4h", "1h": "h", "15m": "15min"}


def test_fetch_multi_tf_records_normalizes_and_isolates_v1():
    calls = []

    def mock_fetcher(coin, interval, count, config=None):
        calls.append((coin, interval, count))
        idx = pd.date_range("2020-01-01", periods=count, freq=_FREQ[interval])
        c = pd.Series([100.0 + i for i in range(count)], index=idx)
        return pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": 1000.0}, index=idx)

    cfg = ScannerConfig()
    v1_counts_before = dict(cfg.candle_counts)

    out = fetch_multi_tf("BTC", config=cfg, fetcher=mock_fetcher)

    # all 7 canonical keys returned, in canonical HTF→LTF order
    assert list(out) == list(CANONICAL_TFS)
    # native keys requested at the V2 depth with the right ccxt interval (1mo→1M)
    recorded = {interval: count for (_, interval, count) in calls}
    assert recorded == {
        "1M": V2_CANDLE_COUNTS["1mo"],
        "1w": V2_CANDLE_COUNTS["1w"],
        "1d": V2_CANDLE_COUNTS["1d"],
        "4h": V2_CANDLE_COUNTS["4h"],
        "1h": V2_CANDLE_COUNTS["1h"],
        "15m": V2_CANDLE_COUNTS["15m"],
    }
    # 3d is NOT fetched — it is resampled from the already-closed 1d
    assert "3D" not in recorded and "3d" not in recorded
    # each native frame has its forming last bar dropped (HIGH-2b)
    assert len(out["1d"]) == V2_CANDLE_COUNTS["1d"] - 1
    assert len(out["15m"]) == V2_CANDLE_COUNTS["15m"] - 1
    # v1 isolation (HIGH-1): config.candle_counts is neither the source of depths nor mutated
    assert cfg.candle_counts == v1_counts_before
    assert V2_CANDLE_COUNTS != cfg.candle_counts


def test_fetch_multi_tf_threads_as_of_onto_fetcher():
    """`as_of` (T-020.15) is bound onto the fetcher so every native fetch is clamped, while the
    per-TF call signature stays unchanged (the as_of=None test double above is untouched)."""
    seen = []

    def mock_fetcher(coin, interval, count, config=None, as_of=None):
        seen.append(as_of)
        idx = pd.date_range("2020-01-01", periods=count, freq=_FREQ[interval])
        c = pd.Series([100.0 + i for i in range(count)], index=idx)
        return pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": 1000.0}, index=idx)

    T = pd.Timestamp("2026-05-20 12:00", tz="UTC")
    fetch_multi_tf("BTC", config=ScannerConfig(), fetcher=mock_fetcher, as_of=T)
    assert seen and all(s == T for s in seen)  # every native fetch received the bound as_of


# --- 3. build_tf_substrate -----------------------------------------------------


def test_build_tf_substrate_deep_frame_non_flat_longer():
    sub = build_tf_substrate(_deep_uptrend(400))
    assert sub.bands is not None and sub.longer is not None
    assert sub.trend_state is not None and sub.regime is not None
    # the depth fix: the longer stack is real, not degenerate-FLAT
    assert sub.trend_state.longer_trend_direction is not TrendDirection.FLAT


def test_build_tf_substrate_vrvp_wired_rotational_reachable():
    # ROTATIONAL requires vrvp != None inside detect_regime, so reaching it out of
    # build_tf_substrate proves the volume profile is wired (anti-vacuity).
    sub = build_tf_substrate(_frame(_zig(120, 95, 105, 8)))
    assert sub.regime.regime is Regime.ROTATIONAL
    assert sub.regime.range_valid


# --- 4. build_market_context_v2 ------------------------------------------------


def test_build_market_context_v2_keys_and_levels():
    ctx = build_market_context_v2("BTC", _realistic_candles())
    assert isinstance(ctx, MarketContextV2)
    # substrates / setup_contexts keyed by exactly the SETUP_TFS present at ≥ 300
    assert set(ctx.substrates) == set(SETUP_TFS)
    assert set(ctx.setup_contexts) == set(SETUP_TFS)
    # macro TFs are candle-only context — never substrates
    for macro in ("1w", "1mo", "3d"):
        assert macro not in ctx.substrates
    # all 7 TFs present in candles
    assert set(ctx.candles) == set(CANONICAL_TFS)
    # top-down S/R from the W/M/D/3D frames
    assert isinstance(ctx.sr_levels, list)
    assert ctx.sr_levels and all(isinstance(lvl, SRLevelV2) for lvl in ctx.sr_levels)


def test_build_market_context_v2_short_setup_tf_absent_not_raise():
    candles = _realistic_candles()
    candles["15m"] = _deep_uptrend(MIN_SUBSTRATE_BARS - 1)  # below floor → omitted, not a raise
    ctx = build_market_context_v2("BTC", candles)
    assert "15m" not in ctx.substrates
    assert "15m" not in ctx.setup_contexts
    assert set(ctx.substrates) == {"1d", "4h", "1h"}


# --- 5. htf_ema_for_chart ------------------------------------------------------


def test_htf_ema_for_chart_alignment_step_and_anti_lookahead():
    # HTF: 4h bars; EMA(period=3) seeds at idx2 → [NaN,NaN,12,14,16,18] keyed by close-time.
    htf = _frame([10, 12, 14, 16, 18, 20], start="2026-01-01 00:00", freq="4h")
    # LTF: 1h bars, open 00:00..21:00 → last close 22:00, BEFORE the last HTF bar closes
    # (2026-01-02 00:00), so the unclosed last HTF bar's EMA (18.0) must never appear.
    ltf_index = pd.date_range("2026-01-01 00:00", periods=22, freq="h")

    out = htf_ema_for_chart(htf, ltf_index, period=3)

    # returned exactly on ltf_index (Chunk-6 renderer gates on index.equals)
    assert out.index.equals(ltf_index)
    # anti-lookahead (HIGH-2): the unclosed last HTF bar (EMA 18.0) leaks onto no LTF bar
    assert 18.0 not in set(out.dropna().to_numpy())
    # warmup region before the first valid HTF close (12:00) is NaN
    assert pd.isna(out.loc["2026-01-01 00:00"])
    # step placement, no 1-bar lag: HTF idx2 closes at 12:00 → the LTF bar [11:00,12:00]
    # (open 11:00) is the first to carry it; the prior bar [10:00,11:00] is still NaN
    assert pd.isna(out.loc["2026-01-01 10:00"])
    assert out.loc["2026-01-01 11:00"] == 12.0
    # constant within the HTF bucket, then steps to 14.0 when HTF idx3 closes at 16:00
    assert out.loc["2026-01-01 12:00"] == 12.0
    assert out.loc["2026-01-01 14:00"] == 12.0
    assert out.loc["2026-01-01 15:00"] == 14.0  # LTF bar [15:00,16:00] closes at 16:00
    # tail: latest closed HTF bar is idx4 (close 20:00, EMA 16.0)
    assert out.iloc[-1] == 16.0


# --- 6. cost benchmark ---------------------------------------------------------


def test_build_market_context_v2_cost_benchmark(capsys):
    candles = _realistic_candles()
    t0 = time.perf_counter()
    ctx = build_market_context_v2("BTC", candles)
    elapsed = time.perf_counter() - t0
    # informational measure with a very loose tripwire — catches a pathological
    # regression, not a tight perf gate (timing is CI-noisy).
    with capsys.disabled():
        print(f"\nbuild_market_context_v2 per-coin cost: {elapsed * 1000:.1f} ms")
    assert elapsed < 10.0
    assert isinstance(ctx, MarketContextV2)
