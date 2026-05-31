"""Tests for scanner/screener_v2.py + the v2 dispatch seam (T-019 Chunk 7b.2b).

The detectors' firing behaviour is owned by test_setups_v2*; here we test the NEW
producer/selection/lift code: deterministic scoring, the lower-TF-reclaim ladder
(15m floor fail-closed), the v2→BoundarySignal lift, cross-TF dedup, one-per-coin
historical dedup, 15m-origin setups flowing through (Chunk 8/8.2), disabled-type filtering, the score-ordered
bundle, and the v2 enriched-chart inputs. Synthetic, fixed-date fixtures; the network
fetch is monkeypatched, but `build_market_context_v2` runs for real.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from scanner import screener_v2
from scanner.config import ScannerConfig
from scanner.data_plane_v2 import MarketContextV2, build_market_context_v2, build_tf_substrate
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
    Stage1ArtifactsV2,
    _dedup_cross_tf,
    _htf_bias,
    _htf_eligible,
    _lower_tf_reclaim,
    _round_sig,
    _run_detectors,
    _select_one_per_coin,
    _stage1_score,
    _to_boundary,
    screen_v2,
)
from scanner.setups_v2 import TFSubstrate
from scanner.trend_state import BandPosition, TrendDirection, TrendState

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
    sig.continuation_count = 2    # non-None so the threading assertion below is non-vacuous
    b = _to_boundary("BTC", "4h", sig, ctx2, leverage=20)

    assert b.coin == "BTC" and b.setup_tf == "4h" and b.max_leverage == 20
    assert b.setup_type is sig.setup_type and b.direction is sig.direction
    assert b.key_level == sig.basis_price == 250.0       # key_level carries the basis
    assert b.confidence == _stage1_score(sig)            # confidence carries the score
    # v2-only execution/confluence fields carry through
    assert b.basis_price == 250.0 and b.entry == sig.entry and b.hard_stop == sig.hard_stop
    assert b.soft_stop is sig.soft_stop and b.confluence == sig.confluence
    assert b.continuation_count == sig.continuation_count    # the .9b cap fact threads through
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


def test_select_one_per_coin_skips_dedup_under_as_of(conn, monkeypatch):
    """Under `as_of` (replay, T-020.15 / SPEC #53) the wall-clock historical dedup is SKIPPED:
    a candidate that would be a historical duplicate is still kept (anchors are independent), and
    `is_duplicate` is never consulted. Control: the same candidate is suppressed when `as_of=None`."""
    top = _bsig(coin="BTC", confidence=0.9, key_level=100.0, setup_type=SetupTypeV2.TREND_CONTINUATION)
    calls = []

    def always_dup(*a, **k):
        calls.append(a)
        return True

    monkeypatch.setattr(screener_v2, "is_duplicate", always_dup)
    cfg = ScannerConfig(setup_version="v2")
    T = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)

    out = _select_one_per_coin([top], cfg, conn, verbose=False, as_of=T)
    assert len(out) == 1 and out[0].setup_type is SetupTypeV2.TREND_CONTINUATION
    assert calls == []  # is_duplicate not consulted under as_of

    # control: as_of=None ⇒ the gate fires ⇒ the dup is suppressed
    assert _select_one_per_coin([top], cfg, conn, verbose=False) == []


# --- 6. screen_v2 end-to-end (real fetch/build, patched leaf detectors) --------


def _patch_pipeline(monkeypatch, run_map):
    """Patch fetch (→ realistic candles) and _run_detectors (→ run_map[coin])."""
    monkeypatch.setattr(screener_v2, "fetch_multi_tf", lambda coin, config, as_of=None: _realistic_candles())
    monkeypatch.setattr(screener_v2, "_run_detectors", lambda ctx2: run_map.get(ctx2.coin, []))


def test_screen_v2_keeps_15m_origin(conn, monkeypatch):
    # 15m-origin setups are NO LONGER dropped (Chunk 8/8.2 — the 15m chart is in the LLM
    # bundle): the higher-confluence 15m setup flows through selection and wins per-coin.
    _patch_pipeline(monkeypatch, {"BTC": [("15m", _v2sig(n_confluence=4)), ("4h", _v2sig(n_confluence=2))]})
    survivors, bundle = screen_v2(["BTC"], {"BTC": 10}, None, ScannerConfig(setup_version="v2"), conn, verbose=False)
    assert [b.setup_tf for b in survivors] == ["15m"]


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
    assert set(bundle.candles_by_coin) == {"BTC", "ETH"}
    assert set(bundle.candles_by_coin["BTC"]) >= {"1d", "4h", "1h"}


def test_screen_v2_returns_typed_stage1_artifacts(conn, monkeypatch):
    """The stage-1→2 seam is a named record (`Stage1ArtifactsV2`), not a bare dict
    (Chunk-10 typed-seam cleanup) — consumers read `.candles_by_coin`, never index it
    as a dict. The v1 path retires at the cutover, so v1's untyped 4H map stays."""
    _patch_pipeline(monkeypatch, {"BTC": [("4h", _v2sig(n_confluence=2))]})
    _, bundle = screen_v2(
        ["BTC"], {"BTC": 10}, None, ScannerConfig(setup_version="v2"), conn, verbose=False,
    )
    assert isinstance(bundle, Stage1ArtifactsV2)
    assert set(bundle.candles_by_coin) == {"BTC"}
    assert isinstance(bundle.candles_by_coin["BTC"], dict)  # per-TF candle dict


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
    monkeypatch.setattr(screener_v2, "fetch_multi_tf", lambda coin, config, as_of=None: _realistic_candles())

    def boom_or_sig(ctx2):
        if ctx2.coin == "BAD":
            raise RuntimeError("detector blew up")
        return [("4h", _v2sig(n_confluence=2))]

    monkeypatch.setattr(screener_v2, "_run_detectors", boom_or_sig)
    survivors, bundle = screen_v2(
        ["BAD", "BTC"], {"BAD": 5, "BTC": 10}, None, ScannerConfig(setup_version="v2"), conn, verbose=False,
    )
    assert [b.coin for b in survivors] == ["BTC"]      # BTC still emits — scan not aborted
    assert "BAD" not in bundle.candles_by_coin and "BTC" in bundle.candles_by_coin  # the bad coin is not even bundled


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


def test_screen_v2_total_wipeout_alerts(conn, monkeypatch):
    """Every coin failing → one distinct error alert (systematic producer breakage is
    surfaced loudly, never masked as a quiet 'no setups today')."""
    import scanner.alerts as alerts
    sent = []
    monkeypatch.setattr(alerts, "send_errors_alert", lambda msg, cfg: sent.append(msg))

    def boom(coin, config, as_of=None):
        raise RuntimeError("everything is broken")

    monkeypatch.setattr(screener_v2, "fetch_multi_tf", boom)
    survivors, bundle = screen_v2(
        ["BTC", "ETH"], {"BTC": 5, "ETH": 5}, None, ScannerConfig(setup_version="v2"), conn, verbose=False,
    )
    assert survivors == [] and bundle.candles_by_coin == {}
    assert len(sent) == 1 and "all 2 coins failed" in sent[0]


def test_screen_v2_partial_failure_no_alert(conn, monkeypatch):
    """A subset of coins failing is ordinary scan noise — no systemic alert."""
    import scanner.alerts as alerts
    sent = []
    monkeypatch.setattr(alerts, "send_errors_alert", lambda msg, cfg: sent.append(msg))
    monkeypatch.setattr(screener_v2, "fetch_multi_tf", lambda coin, config, as_of=None: _realistic_candles())

    def boom_or_sig(ctx2):
        if ctx2.coin == "BAD":
            raise RuntimeError("one bad coin")
        return [("4h", _v2sig(n_confluence=2))]

    monkeypatch.setattr(screener_v2, "_run_detectors", boom_or_sig)
    survivors, _ = screen_v2(
        ["BAD", "BTC"], {"BAD": 5, "BTC": 5}, None, ScannerConfig(setup_version="v2"), conn, verbose=False,
    )
    assert [b.coin for b in survivors] == ["BTC"]
    assert sent == []  # partial failure is not systemic — no alert


def test_screen_v2_real_detectors_no_crash(conn, monkeypatch):
    """The REAL detector path (fetch → build_market_context_v2 → 6 detectors → select)
    runs end-to-end on a realistic fixture without raising; emits 0+ well-formed signals."""
    monkeypatch.setattr(screener_v2, "fetch_multi_tf", lambda coin, config, as_of=None: _realistic_candles())
    survivors, bundle = screen_v2(["BTC"], {"BTC": 10}, None, ScannerConfig(setup_version="v2"), conn, verbose=False)
    assert isinstance(survivors, list)
    for b in survivors:
        assert isinstance(b.setup_type, SetupTypeV2)
        assert b.coin in bundle.candles_by_coin


# --- 7. v2 enriched-chart inputs -----------------------------------------------


def test_v2_chart_inputs_align_next_htf_ema():
    candles = _realistic_candles()
    sig = _bsig(coin="BTC")
    candle_sets, htf_emas, _ = _v2_chart_inputs(
        sig, Stage1ArtifactsV2(candles_by_coin={"BTC": candles}, macro_levels_by_coin={"BTC": []}),
        ScannerConfig(setup_version="v2"),
    )

    # the 1d/4h/1h/15m detector frames feed the charts (15m added Chunk 8/8.2)
    assert set(candle_sets) == {"1d", "4h", "1h", "15m"}
    # each HTF anchor EMA is aligned exactly onto its chart's index (renderer gates on equals)
    for tf in ("4h", "1h", "15m"):
        assert htf_emas[tf].index.equals(candle_sets[tf].index)
    # the next-HTF 200 EMA has real (non-NaN) values: 4h-200 on 1h, 1h-200 on 15m
    assert htf_emas["1h"].notna().any()
    assert htf_emas["15m"].notna().any()


# --- 8. Sub-chunk H: per-chart-TF levels + macro plumbing ----------------------


def test_chart_inputs_returns_per_chart_spinelevels():
    """chart_inputs returns a 3-tuple; the 3rd entry maps each chart-TF to a SpineLevel
    list (projected at the producer — never raw SRLevelV2)."""
    candles = _realistic_candles()
    macro = build_market_context_v2("BTC", candles).sr_levels
    result = screener_v2.chart_inputs(candles, macro_levels=macro)
    assert len(result) == 3
    # named record (ChartInputs), not a positional tuple — named access AND unpack both work
    assert result.candle_sets is result[0] and result.per_chart_levels is result[2]
    candle_sets, _htf_emas, per_chart = result
    assert set(per_chart) == set(candle_sets)
    for levels in per_chart.values():
        assert isinstance(levels, list)
        assert all(isinstance(lvl, SpineLevel) for lvl in levels)


def test_chart_inputs_does_not_call_derive_sr_levels(monkeypatch):
    """chart_inputs is a pure consumer of the pre-computed macro — it never recomputes
    `derive_sr_levels`. Pins the architectural contract: stage 1 owns macro, stage 2 reads it."""
    import scanner.sr_levels as srmod
    candles = _realistic_candles()
    macro = build_market_context_v2("BTC", candles).sr_levels   # the precompute (real path)
    called = []
    monkeypatch.setattr(srmod, "derive_sr_levels", lambda *a, **k: called.append(1))
    screener_v2.chart_inputs(candles, macro_levels=macro)
    assert called == []


def test_chart_inputs_per_chart_levels_have_quality_set():
    """Every projected SpineLevel carries a post-`_assign_quality` quality (not None)."""
    candles = _realistic_candles()
    macro = build_market_context_v2("BTC", candles).sr_levels
    _, _, per_chart = screener_v2.chart_inputs(candles, macro_levels=macro)
    assert any(per_chart[tf] for tf in per_chart)        # anti-vacuity: some TF has levels
    for levels in per_chart.values():
        for lvl in levels:
            assert lvl.quality is not None


def test_chart_inputs_per_chart_levels_keyed_only_by_chart_htf():
    """Keys are exactly the chartable-TF set (`_CHART_HTF`) — no leak, single owner."""
    candles = _realistic_candles()
    macro = build_market_context_v2("BTC", candles).sr_levels
    _, _, per_chart = screener_v2.chart_inputs(candles, macro_levels=macro)
    assert set(per_chart) <= set(screener_v2._CHART_HTF)
    assert set(per_chart) == {"1d", "4h", "1h", "15m"}


def test_v2_chart_inputs_bundle_hit_uses_prefetched_macro(monkeypatch):
    """Happy path: macro is read from the prefetched bundle (identity) — NOT recomputed."""
    import scanner.pipeline as pipemod
    candles = _realistic_candles()
    macro = build_market_context_v2("BTC", candles).sr_levels
    prefetched = Stage1ArtifactsV2(candles_by_coin={"BTC": candles}, macro_levels_by_coin={"BTC": macro})
    monkeypatch.setattr(pipemod, "build_market_context_v2",
                        lambda *a, **k: pytest.fail("recomputed macro on a bundle hit"))
    captured = {}
    real_ci = pipemod.chart_inputs

    def spy_ci(c, macro_levels):
        captured["m"] = macro_levels
        return real_ci(c, macro_levels=macro_levels)

    monkeypatch.setattr(pipemod, "chart_inputs", spy_ci)
    _v2_chart_inputs(_bsig(coin="BTC"), prefetched, ScannerConfig(setup_version="v2"))
    assert captured["m"] is macro       # the prefetched list itself, no recompute


def test_v2_chart_inputs_bundle_miss_recomputes_macro(monkeypatch):
    """Bundle miss (prefetched=None): refetch candles AND recompute the macro substrate
    exactly once via build_market_context_v2 — recovery-only path."""
    import scanner.pipeline as pipemod
    candles = _realistic_candles()
    monkeypatch.setattr(pipemod, "fetch_multi_tf", lambda coin, config, as_of=None: candles)
    real_bmc = pipemod.build_market_context_v2
    calls = []

    def spy_bmc(coin, c):
        calls.append(coin)
        return real_bmc(coin, c)

    monkeypatch.setattr(pipemod, "build_market_context_v2", spy_bmc)
    captured = {}
    real_ci = pipemod.chart_inputs

    def spy_ci(c, macro_levels):
        captured["m"] = macro_levels
        return real_ci(c, macro_levels=macro_levels)

    monkeypatch.setattr(pipemod, "chart_inputs", spy_ci)
    _v2_chart_inputs(_bsig(coin="BTC"), None, ScannerConfig(setup_version="v2"))
    assert calls == ["BTC"]                                       # exactly one recompute
    assert captured["m"] == real_bmc("BTC", candles).sr_levels    # the recomputed macro flows in


def test_detection_substrate_unaffected_by_chart_levels():
    """The detection substrate is macro-only regardless of LTF richness in `candles` —
    build_market_context_v2 returns only W/M/D/3D tf_source entries; derive_chart_levels is
    a separate seam never reached from detection. Pins the AT_SR calibration invariance
    structurally (survives the `_TF_RANK` +3 rescale, which is tf_source-agnostic)."""
    candles = _realistic_candles()                # 4h/1h/15m frames are full of LTF pivots
    ctx2 = build_market_context_v2("BTC", candles)
    assert ctx2.sr_levels                          # anti-vacuity: a real macro substrate exists
    assert all(lvl.tf_source in {"1mo", "1w", "3d", "1d"} for lvl in ctx2.sr_levels)


# --- 9. T-020.8: top-down HTF eligibility gate ---------------------------------


def _trend_state(longer: TrendDirection) -> TrendState:
    """A minimal TrendState carrying a chosen 100/200/300 (`longer`) bias — the only field
    the HTF gate reads. Local-stack fields are neutral (FLAT/INSIDE) so `_lower_tf_reclaim`
    short-circuits to False and never depends on this stub."""
    return TrendState(
        local_trend_direction=TrendDirection.FLAT,
        price_vs_ribbon=BandPosition.INSIDE,
        ribbon_compressed=False,
        longer_trend_direction=longer,
        price_vs_longer=BandPosition.INSIDE,
        gap_to_next_ma=0.0,
        compression_tightness=0.0,
    )


def _stub_sub(longer: TrendDirection) -> TFSubstrate:
    """A TFSubstrate whose only meaningful field is `trend_state.longer_trend_direction` —
    `_htf_bias`/`_htf_eligible` touch nothing else, so bands/longer/regime stay None."""
    return TFSubstrate(bands=None, longer=None, trend_state=_trend_state(longer), regime=None)


def _htf_ctx(biases: dict[str, TrendDirection], setup_tfs: set[str] | None = None) -> MarketContextV2:
    """A MarketContextV2 with one substrate per TF in `biases` (the HTF bias to read). Any
    TF in `setup_tfs` not already in `biases` gets a FLAT substrate + a SetupContext so
    `_run_detectors` runs its detector loop (the others are bias-only anchors the loop skips
    via `sc is None`). The detector value is a sentinel — patched detectors ignore it."""
    setup_tfs = setup_tfs or set()
    subs = {tf: _stub_sub(d) for tf, d in biases.items()}
    for tf in setup_tfs:
        subs.setdefault(tf, _stub_sub(TrendDirection.FLAT))
    scs = {tf: object() for tf in setup_tfs}
    return MarketContextV2(coin="BTC", candles={}, substrates=subs, setup_contexts=scs, sr_levels=[])


_DETECTOR_OF = {
    SetupTypeV2.TREND_CONTINUATION: "detect_trend_continuation",
    SetupTypeV2.GAP_FILL: "detect_gap_fill",
    SetupTypeV2.SFP_RANGE: "detect_sfp_range",
    SetupTypeV2.RSI_DIVERGENCE_RANGE: "detect_rsi_divergence_range",
}


def _patch_detectors(monkeypatch, emitter: str, signals):
    """Patch the six leaf detectors so only `emitter` returns `signals`, the rest []. They are
    looked up in screener_v2's namespace (imported names); `*a, **k` absorbs the keyword
    `lower_tf_reclaim` that `detect_trend_reclaim` alone takes."""
    names = ["detect_trend_continuation", "detect_trend_reclaim", "detect_deviation_reclaim",
             "detect_gap_fill", "detect_sfp_range", "detect_rsi_divergence_range"]
    for n in names:
        ret = signals if n == emitter else []
        monkeypatch.setattr(screener_v2, n, lambda *a, _r=ret, **k: _r)


# -- _htf_bias: nearest-higher-available walk --

def test_htf_bias_nearest_higher_available():
    # each setup TF reads its NEAREST-higher present substrate, not a further-up one.
    ctx = _htf_ctx({"1d": TrendDirection.DOWN, "4h": TrendDirection.DOWN, "1h": TrendDirection.UP})
    assert _htf_bias("15m", ctx) is TrendDirection.UP    # nearest-higher of 15m = 1h
    assert _htf_bias("1h", ctx) is TrendDirection.DOWN   # nearest-higher of 1h = 4h
    assert _htf_bias("4h", ctx) is TrendDirection.DOWN   # nearest-higher of 4h = 1d


def test_htf_bias_skips_over_absent_substrate():
    # 1h's substrate is absent → 15m's nearest-AVAILABLE higher is 4h (DOWN), skipping 1h.
    ctx = _htf_ctx({"4h": TrendDirection.DOWN, "1d": TrendDirection.UP})
    assert _htf_bias("15m", ctx) is TrendDirection.DOWN


def test_htf_bias_none_when_no_higher_available():
    # 1d is the highest setup TF — nothing above is a substrate → None (the #46 1d-ungated limit).
    assert _htf_bias("1d", _htf_ctx({"1d": TrendDirection.DOWN})) is None
    # a short-history coin with ONLY 15m present → every higher TF absent → None.
    assert _htf_bias("15m", _htf_ctx({"15m": TrendDirection.UP})) is None


def test_htf_bias_reads_real_substrate():
    # not a stub artifact: a REAL DOWN-trend substrate's longer stack reads DOWN through the gate.
    dn = build_tf_substrate(_frame([300 - i * 0.5 for i in range(400)]))
    ctx = MarketContextV2(coin="BTC", candles={}, substrates={"1d": dn},
                          setup_contexts={}, sr_levels=[])
    assert _htf_bias("4h", ctx) is TrendDirection.DOWN


# -- _htf_eligible: with-trend kept, counter-trend dropped --

def test_htf_eligible_drops_counter_keeps_with_trend():
    down = _htf_ctx({"1d": TrendDirection.DOWN})         # nearest-higher of 4h = 1d = DOWN
    assert _htf_eligible("4h", Direction.LONG, down) is False   # LONG into a DOWN HTF → dropped
    assert _htf_eligible("4h", Direction.SHORT, down) is True   # SHORT with the DOWN HTF → kept
    up = _htf_ctx({"1d": TrendDirection.UP})
    assert _htf_eligible("4h", Direction.LONG, up) is True
    assert _htf_eligible("4h", Direction.SHORT, up) is False


def test_htf_eligible_flat_and_none_are_eligible():
    flat = _htf_ctx({"1d": TrendDirection.FLAT})
    assert _htf_eligible("4h", Direction.LONG, flat) is True    # FLAT HTF = no opposition
    assert _htf_eligible("4h", Direction.SHORT, flat) is True
    no_anchor = _htf_ctx({"4h": TrendDirection.DOWN})           # nothing above 1d is a substrate
    assert _htf_eligible("1d", Direction.LONG, no_anchor) is True


# -- _run_detectors: the gate wired in (selection-filter, anti-vacuity) --

def test_run_detectors_drops_gated_setup_into_opposing_htf(monkeypatch):
    """A with-trend directional setup emitted against a DOWN nearest-higher HTF is dropped;
    the SAME emission under an UP HTF survives — anti-vacuity (the drop is the gate, not a
    non-emission)."""
    long_cont = _v2sig(setup_type=SetupTypeV2.TREND_CONTINUATION, direction=Direction.LONG)
    _patch_detectors(monkeypatch, "detect_trend_continuation", [long_cont])

    down = _htf_ctx({"1h": TrendDirection.DOWN}, setup_tfs={"15m"})   # 15m LONG vs DOWN 1h
    assert [s.setup_type for _, s in _run_detectors(down)] == []

    up = _htf_ctx({"1h": TrendDirection.UP}, setup_tfs={"15m"})       # same emission, UP 1h
    assert [(tf, s.setup_type) for tf, s in _run_detectors(up)] == \
        [("15m", SetupTypeV2.TREND_CONTINUATION)]


def test_run_detectors_exempts_gapfill_and_rotational(monkeypatch):
    """gap_fill (counter-trend by definition) + the rotational range setups are NOT gated —
    each survives emitting against an opposing HTF bias."""
    down = _htf_ctx({"1h": TrendDirection.DOWN}, setup_tfs={"15m"})
    for st in (SetupTypeV2.GAP_FILL, SetupTypeV2.SFP_RANGE, SetupTypeV2.RSI_DIVERGENCE_RANGE):
        sig = _v2sig(setup_type=st, direction=Direction.LONG)        # LONG into a DOWN HTF
        _patch_detectors(monkeypatch, _DETECTOR_OF[st], [sig])
        assert [s.setup_type for _, s in _run_detectors(down)] == [st], f"{st} should be exempt"


def test_run_detectors_gate_drops_at_selection_not_detection(monkeypatch):
    """Detection invariance: a gated counter-HTF setup is still DETECTED (its detector runs),
    then dropped from the output — the gate is a selection filter, not detection suppression."""
    calls = []
    long_cont = _v2sig(setup_type=SetupTypeV2.TREND_CONTINUATION, direction=Direction.LONG)
    for n in ("detect_trend_reclaim", "detect_deviation_reclaim", "detect_gap_fill",
              "detect_sfp_range", "detect_rsi_divergence_range"):
        monkeypatch.setattr(screener_v2, n, lambda *a, **k: [])
    monkeypatch.setattr(screener_v2, "detect_trend_continuation",
                        lambda *a, **k: (calls.append(1), [long_cont])[1])

    out = _run_detectors(_htf_ctx({"1h": TrendDirection.DOWN}, setup_tfs={"15m"}))
    assert out == []        # the counter-HTF continuation is dropped from output
    assert calls == [1]     # but the detector WAS called — detection ran, gate filtered
