"""Tests for scanner/screener_v2.py + the v2 dispatch seam (T-019 Chunk 7b.2b).

The detectors' firing behaviour is owned by test_setups_v2*; here we test the NEW
producer/selection/lift code: deterministic scoring, the lower-TF-reclaim ladder
(15m floor fail-closed), the v2→BoundarySignal lift, cross-TF dedup, one-per-coin
historical dedup, 15m-origin suppression, disabled-type filtering, the score-ordered
bundle, and the v2 enriched-chart inputs. Synthetic, fixed-date fixtures; the network
fetch is monkeypatched, but `build_market_context_v2` runs for real.
"""

import pandas as pd
import pytest

from scanner import screener_v2
from scanner.config import ScannerConfig
from scanner.data_plane_v2 import build_market_context_v2
from scanner.db import get_connection
from scanner.models import (
    BoundarySignal,
    ConfluenceFactor,
    Direction,
    SetupSignalV2,
    SetupTypeV2,
    SoftStop,
    SoftStopMode,
    SpineContext,
    SpineLevel,
    Structure,
)
from scanner.pipeline import _v2_chart_inputs
from scanner.screener_v2 import (
    _dedup_cross_tf,
    _lower_tf_reclaim,
    _round_sig,
    _select_one_per_coin,
    _stage1_score,
    _to_boundary,
    screen_v2,
)

# --- builders ------------------------------------------------------------------


def _frame(closes, wick=0.3, start="2026-01-01", freq="D") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq=freq)
    c = pd.Series([float(x) for x in closes], index=idx)
    return pd.DataFrame(
        {"Open": c, "High": c + wick, "Low": c - wick, "Close": c, "Volume": 1000.0}, index=idx
    )


def _zig(n, lo, hi, half):
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
    return _frame([100 + i * 0.5 for i in range(n)])


def _realistic_candles() -> dict[str, pd.DataFrame]:
    deep = _deep_uptrend(400)
    return {
        "1mo": _frame(_zig(36, 80, 120, 5)),
        "1w": _frame(_zig(104, 90, 110, 6)),
        "3d": _frame(_zig(130, 95, 115, 7)),
        "1d": deep, "4h": deep, "1h": deep, "15m": deep,
    }


def _v2sig(setup_type=SetupTypeV2.TREND_CONTINUATION, direction=Direction.LONG,
           basis=100.0, n_confluence=2, weakness=False) -> SetupSignalV2:
    factors = list(ConfluenceFactor)[:n_confluence]
    return SetupSignalV2(
        setup_type=setup_type, direction=direction, basis_price=basis,
        entry=basis, hard_stop=basis * 0.97,
        soft_stop=SoftStop(SoftStopMode.TREND_RECLAIM_AGAINST, basis),
        confluence=frozenset(factors), weakness_flag=weakness,
    )


def _bsig(coin="BTC", setup_type=SetupTypeV2.TREND_CONTINUATION, direction=Direction.LONG,
          tf="4h", confidence=0.5, key_level=100.0) -> BoundarySignal:
    spine = SpineContext(
        structure=Structure.BULLISH, sr_levels=[SpineLevel(price=key_level)],
        range_bounds=None, atr=2.0, current_price=key_level, fib_levels=None, regime="trending",
    )
    return BoundarySignal(
        coin=coin, setup_type=setup_type, direction=direction, confidence=confidence,
        key_level=key_level, context=spine, description="x", setup_tf=tf, basis_price=key_level,
    )


@pytest.fixture
def conn(tmp_path):
    c = get_connection(tmp_path / "t.db")
    yield c
    c.close()


# --- 1. scoring + rounding -----------------------------------------------------


def test_stage1_score_monotonic_in_confluence():
    s1 = _stage1_score(_v2sig(n_confluence=1))
    s3 = _stage1_score(_v2sig(n_confluence=3))
    s5 = _stage1_score(_v2sig(n_confluence=5))
    assert 0.0 <= s1 < s3 < s5 <= 1.0
    # saturates at SATURATION factors — 6 confluence is not scored above 5
    assert _stage1_score(_v2sig(n_confluence=6)) == s5


def test_stage1_score_weakness_demotes():
    clean = _stage1_score(_v2sig(n_confluence=3, weakness=False))
    weak = _stage1_score(_v2sig(n_confluence=3, weakness=True))
    assert weak < clean


def test_round_sig_scale_free():
    # collapses near-identical levels across very different scales
    assert _round_sig(64123.4) == _round_sig(64123.0)        # BTC scale
    assert _round_sig(0.0012341) == _round_sig(0.0012339)    # sub-cent scale
    assert _round_sig(0.0) == 0.0 and _round_sig(float("nan")) == 0.0


# --- 2. lower-TF reclaim ladder ------------------------------------------------


def test_lower_tf_reclaim_15m_floor_fail_closed():
    ctx2 = build_market_context_v2("BTC", _realistic_candles())
    # 15m is the ladder floor — no lower TF exists, so the reclaim is always False
    # (a trend_reclaim can never emit off accidental truthiness at the floor).
    assert _lower_tf_reclaim("15m", ctx2) is False


def test_lower_tf_reclaim_long_uptrend_true():
    ctx2 = build_market_context_v2("BTC", _realistic_candles())
    # deep uptrend → the one-TF-lower (4h) price sits above its 200EMA → LONG reclaim
    assert _lower_tf_reclaim("1d", ctx2) is True


# --- 3. the v2 → BoundarySignal lift -------------------------------------------


def test_to_boundary_maps_v2_fields():
    ctx2 = build_market_context_v2("BTC", _realistic_candles())
    sig = _v2sig(basis=250.0, n_confluence=3)
    b = _to_boundary("BTC", "4h", sig, ctx2, leverage=20)

    assert b.coin == "BTC" and b.setup_tf == "4h" and b.max_leverage == 20
    assert b.setup_type is sig.setup_type and b.direction is sig.direction
    assert b.key_level == sig.basis_price == 250.0       # key_level carries the basis
    assert b.confidence == _stage1_score(sig)            # confidence carries the score
    # v2-only execution/confluence fields carry through
    assert b.basis_price == 250.0 and b.entry == sig.entry and b.hard_stop == sig.hard_stop
    assert b.soft_stop is sig.soft_stop and b.confluence == sig.confluence
    # context is the neutral SpineContext (nested S/R neutralized to SpineLevel)
    assert isinstance(b.context, SpineContext)
    assert all(isinstance(lvl, SpineLevel) for lvl in b.context.sr_levels)
    assert b.context.fib_levels is None                  # v2 carries no fib
    assert b.context.vol_percentile is None              # unknown → NULL, not a fake 0.0
    assert b.context.regime == "trending"


# --- 4. cross-TF dedup ---------------------------------------------------------


def test_dedup_cross_tf_keeps_highest_score_then_higher_tf():
    # same (coin, type, dir, level) on three TFs; highest score wins
    cands = [
        _bsig(tf="1h", confidence=0.3),
        _bsig(tf="4h", confidence=0.9),
        _bsig(tf="1d", confidence=0.3),
    ]
    out = _dedup_cross_tf(cands)
    assert len(out) == 1 and out[0].setup_tf == "4h"

    # score tie → the higher TF (1d > 4h > 1h) survives
    tie = [_bsig(tf="1h", confidence=0.5), _bsig(tf="1d", confidence=0.5)]
    assert _dedup_cross_tf(tie)[0].setup_tf == "1d"

    # different level → NOT deduped (two distinct levels survive)
    distinct = [_bsig(key_level=100.0), _bsig(key_level=200.0)]
    assert len(_dedup_cross_tf(distinct)) == 2


# --- 5. one-per-coin historical dedup ------------------------------------------


def test_select_one_per_coin_one_signal_each(conn):
    cands = [
        _bsig(coin="BTC", confidence=0.9, key_level=100.0),
        _bsig(coin="BTC", confidence=0.4, key_level=200.0, setup_type=SetupTypeV2.GAP_FILL),
        _bsig(coin="ETH", confidence=0.7, key_level=50.0),
    ]
    out = _select_one_per_coin(cands, ScannerConfig(setup_version="v2"), conn, verbose=False)
    by_coin = {b.coin: b for b in out}
    assert set(by_coin) == {"BTC", "ETH"}
    assert by_coin["BTC"].confidence == 0.9    # highest-score candidate per coin


def test_select_one_per_coin_skips_historical_dup_keeps_next(conn, monkeypatch):
    """Top candidate is a historical duplicate → the next eligible one for that coin
    still emits (preserves v1's per-coin dedup behaviour)."""
    top = _bsig(coin="BTC", confidence=0.9, key_level=100.0, setup_type=SetupTypeV2.TREND_CONTINUATION)
    nxt = _bsig(coin="BTC", confidence=0.4, key_level=200.0, setup_type=SetupTypeV2.GAP_FILL)

    def fake_is_dup(coin, setup_type, direction, hours, c, exchange="hyperliquid"):
        return setup_type == "trend_continuation"  # only the top candidate is a dup

    monkeypatch.setattr(screener_v2, "is_duplicate", fake_is_dup)
    out = _select_one_per_coin([top, nxt], ScannerConfig(setup_version="v2"), conn, verbose=False)
    assert len(out) == 1 and out[0].setup_type is SetupTypeV2.GAP_FILL


# --- 6. screen_v2 end-to-end (real fetch/build, patched leaf detectors) --------


def _patch_pipeline(monkeypatch, run_map):
    """Patch fetch (→ realistic candles) and _run_detectors (→ run_map[coin])."""
    monkeypatch.setattr(screener_v2, "fetch_multi_tf", lambda coin, config: _realistic_candles())
    monkeypatch.setattr(screener_v2, "_run_detectors", lambda ctx2: run_map.get(ctx2.coin, []))


def test_screen_v2_drops_15m_origin(conn, monkeypatch):
    _patch_pipeline(monkeypatch, {"BTC": [("15m", _v2sig(n_confluence=4)), ("4h", _v2sig(n_confluence=2))]})
    survivors, bundle = screen_v2(["BTC"], {"BTC": 10}, None, ScannerConfig(setup_version="v2"), conn, verbose=False)
    assert all(b.setup_tf != "15m" for b in survivors)
    assert len(survivors) == 1 and survivors[0].setup_tf == "4h"


def test_screen_v2_order_and_bundle(conn, monkeypatch):
    run_map = {
        "BTC": [("4h", _v2sig(n_confluence=4))],          # higher score
        "ETH": [("1h", _v2sig(n_confluence=1))],          # lower score
    }
    _patch_pipeline(monkeypatch, run_map)
    survivors, bundle = screen_v2(
        ["BTC", "ETH"], {"BTC": 10, "ETH": 25}, None, ScannerConfig(setup_version="v2"), conn, verbose=False,
    )
    # one per coin, ordered by score desc → BTC (4 confluence) before ETH (1)
    assert [b.coin for b in survivors] == ["BTC", "ETH"]
    assert survivors[0].confidence > survivors[1].confidence
    assert survivors[1].max_leverage == 25
    # bundle pruned to survivor coins, carries the closed-bar candle dicts for charting
    assert set(bundle) == {"BTC", "ETH"}
    assert set(bundle["BTC"]) >= {"1d", "4h", "1h"}


def test_screen_v2_disabled_type_never_lifts(conn, monkeypatch):
    run_map = {"BTC": [
        ("4h", _v2sig(setup_type=SetupTypeV2.TREND_CONTINUATION, n_confluence=2)),
        ("4h", _v2sig(setup_type=SetupTypeV2.GAP_FILL, n_confluence=4)),
    ]}
    _patch_pipeline(monkeypatch, run_map)
    survivors, _ = screen_v2(
        ["BTC"], {"BTC": 10}, {"gap_fill"}, ScannerConfig(setup_version="v2"), conn, verbose=False,
    )
    assert len(survivors) == 1
    assert survivors[0].setup_type is SetupTypeV2.TREND_CONTINUATION  # gap_fill dropped pre-rank


def test_screen_v2_bad_coin_isolated(conn, monkeypatch):
    """A detector/lift failure on one coin is logged + skipped, never aborts the scan or
    half-emits — the per-coin production is one guarded, atomically-committed unit."""
    monkeypatch.setattr(screener_v2, "fetch_multi_tf", lambda coin, config: _realistic_candles())

    def boom_or_sig(ctx2):
        if ctx2.coin == "BAD":
            raise RuntimeError("detector blew up")
        return [("4h", _v2sig(n_confluence=2))]

    monkeypatch.setattr(screener_v2, "_run_detectors", boom_or_sig)
    survivors, bundle = screen_v2(
        ["BAD", "BTC"], {"BAD": 5, "BTC": 10}, None, ScannerConfig(setup_version="v2"), conn, verbose=False,
    )
    assert [b.coin for b in survivors] == ["BTC"]      # BTC still emits — scan not aborted
    assert "BAD" not in bundle and "BTC" in bundle     # the bad coin is not even bundled


def test_select_one_per_coin_dedup_error_isolated(conn, monkeypatch):
    """A per-coin is_duplicate failure skips that coin, never the whole selection."""
    def maybe_boom(coin, *a, **k):
        if coin == "BAD":
            raise RuntimeError("db down for this coin")
        return False

    monkeypatch.setattr(screener_v2, "is_duplicate", maybe_boom)
    cands = [_bsig(coin="BAD", confidence=0.9), _bsig(coin="ETH", confidence=0.7)]
    out = _select_one_per_coin(cands, ScannerConfig(setup_version="v2"), conn, verbose=False)
    assert [b.coin for b in out] == ["ETH"]            # BAD skipped, ETH still selected


def test_screen_v2_real_detectors_no_crash(conn, monkeypatch):
    """The REAL detector path (fetch → build_market_context_v2 → 6 detectors → select)
    runs end-to-end on a realistic fixture without raising; emits 0+ well-formed signals."""
    monkeypatch.setattr(screener_v2, "fetch_multi_tf", lambda coin, config: _realistic_candles())
    survivors, bundle = screen_v2(["BTC"], {"BTC": 10}, None, ScannerConfig(setup_version="v2"), conn, verbose=False)
    assert isinstance(survivors, list)
    for b in survivors:
        assert b.setup_tf != "15m" and isinstance(b.setup_type, SetupTypeV2)
        assert b.coin in bundle


# --- 7. v2 enriched-chart inputs -----------------------------------------------


def test_v2_chart_inputs_align_next_htf_ema():
    candles = _realistic_candles()
    sig = _bsig(coin="BTC")
    candle_sets, htf_emas = _v2_chart_inputs(sig, {"BTC": candles}, ScannerConfig(setup_version="v2"))

    # the 1d/4h/1h detector frames feed the charts
    assert set(candle_sets) == {"1d", "4h", "1h"}
    # each HTF anchor EMA is aligned exactly onto its chart's index (renderer gates on equals)
    for tf in ("4h", "1h"):
        assert htf_emas[tf].index.equals(candle_sets[tf].index)
    # the 4h-200 EMA on the 1h chart has real (non-NaN) values
    assert htf_emas["1h"].notna().any()
