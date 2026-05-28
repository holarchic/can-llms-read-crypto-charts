"""Tests for scanner/sr_levels.py (T-019 Chunk 3, top-down S/R substrate).

Tier 1 — synthetic multi-TF (the primary gate). Fixed-date fixtures (never
``Timestamp.now()``); hand-crafted, not RNG. The "fewer-but-higher-quality than
clustering" claim drives the REAL ``cluster_sr_levels`` (not a mimic), and includes
an adversarial fixture where local clustering finds an attractive level the
top-down contract must reject. Tier 2 (real-chart fixture band) is a separate file
built when the W/M/D/3D candle fixtures are committed.

The derivation stages are private (``_find_htf_pivots`` etc.); these unit tests
exercise them directly while production code uses the ``derive_sr_levels`` seam.
A single-tent n=5/window=2 frame with ``Low == High`` yields exactly one high pivot
at the apex (the apex low is the band max, never the min) — the controlled
single-pivot building block used throughout.
"""

import math

import pandas as pd

from scanner.indicators import cluster_sr_levels, find_swing_points
from scanner.ma_indicators import VolumeProfile, compute_volume_profile
from scanner.models import SRLevelV2
from scanner.sr_levels import (
    _assign_quality,
    _find_htf_pivots,
    _merge_sr_levels,
    _refine_pivots_on_lower_tf,
    _score_with_vrvp,
    anchor_vrvp_for_range,
    derive_sr_levels,
)


def _frame(highs, lows=None, start="2024-01-01", freq="D"):
    highs = [float(h) for h in highs]
    lows = list(highs) if lows is None else [float(x) for x in lows]
    n = len(highs)
    idx = pd.date_range(start, periods=n, freq=freq)
    closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    return pd.DataFrame(
        {"Open": closes, "High": highs, "Low": lows, "Close": closes, "Volume": [1.0] * n},
        index=idx,
    )


def _empty():
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def _lvl(price, tf, ptype="high", confirmations=1, vrvp=False, formed="2024-01-01", last=None):
    """Test fixture: a line-like level (zone collapsed to `price`). Real production levels
    carry wick+body zones from `_pivot`; this helper covers fixtures that don't care about
    zone semantics. Asymmetric inward pad preserves the `price == zone-edge` invariant.
    """
    f = pd.Timestamp(formed)
    p = float(price)
    pad = abs(p) * 0.001
    if ptype == "high":
        zone_low, zone_high = p - pad, p
    else:
        zone_low, zone_high = p, p + pad
    return SRLevelV2(
        price=p, tf_source=tf, pivot_type=ptype, confirmations=confirmations,
        vrvp_confluence=vrvp, formed_at=f, last_test=pd.Timestamp(last) if last else f,
        zone_low=zone_low, zone_high=zone_high,
    )


# ── pivots ────────────────────────────────────────────────────────────────────

def test_single_tent_yields_one_high_pivot():
    pivots = _find_htf_pivots(_frame([96, 98, 100, 98, 96], freq="7D"), _empty(), window=2, tol=0.01)
    assert len(pivots) == 1
    p = pivots[0]
    assert p.pivot_type == "high" and p.price == 100.0 and p.tf_source == "1w"


def test_tie_tolerant_plateau_keeps_most_recent_timestamp():
    # window=1, two equal maxima at idx1,idx2 → ONE pivot at the LATER bar (idx2).
    df = _frame([1, 5, 5, 1, 1], lows=[0.5, 0.4, 0.3, 0.4, 0.5], start="2024-03-01", freq="7D")
    highs = [p for p in _find_htf_pivots(df, _empty(), window=1, tol=0.01) if p.pivot_type == "high"]
    assert len(highs) == 1
    assert highs[0].formed_at == df.index[2]   # later of the tied bars, not idx1
    assert highs[0].formed_at != df.index[1]


def test_short_frame_yields_no_pivots_not_a_raise():
    assert _find_htf_pivots(_frame([1, 2, 3]), _empty(), window=2, tol=0.01) == []


def test_w_and_m_at_same_price_merge_to_one_higher_rank_level():
    w = _frame([96, 98, 100, 98, 96], start="2024-01-01", freq="7D")
    m = _frame([97, 99, 100, 99, 97], start="2024-01-01", freq="30D")
    levels = _find_htf_pivots(w, m, window=2, tol=0.01)
    at_100 = [l for l in levels if abs(l.price - 100) < 1]
    assert len(at_100) == 1
    assert at_100[0].tf_source == "1mo"     # higher rank wins
    assert at_100[0].confirmations == 2     # two distinct lineages (1w + 1mo)


# ── refinement ──────────────────────────────────────────────────────────────────

def _refine(macro, daily, three_day=None, **kw):
    return _refine_pivots_on_lower_tf(
        macro, daily, three_day if three_day is not None else _empty(),
        window=kw.get("window", 2), n_sub=kw.get("n_sub", 3), tol=kw.get("tol", 0.01),
    )


def test_refine_includes_pivot_strictly_inside_bracket():
    macro = [_lvl(50, "1w", "low"), _lvl(150, "1w", "high")]
    daily = _frame([98, 99, 100, 99, 98], start="2024-06-01")  # forms after the macro edges
    refined = _refine(macro, daily)
    assert len(refined) == 1 and refined[0].price == 100.0 and refined[0].tf_source == "1d"


def test_refine_excludes_pivot_within_tol_of_a_macro_edge():
    macro = [_lvl(50, "1w", "low"), _lvl(150, "1w", "high")]
    daily = _frame([50.2, 50.3, 50.4, 50.3, 50.2], start="2024-06-01")  # 50.4 within 1% of edge 50
    assert _refine(macro, daily) == []


def test_refine_excludes_pivot_formed_before_later_macro_edge():
    macro = [_lvl(50, "1w", "low", formed="2024-09-01"), _lvl(150, "1w", "high", formed="2024-09-01")]
    daily = _frame([98, 99, 100, 99, 98], start="2024-01-01")  # forms BEFORE the macro edges
    assert _refine(macro, daily) == []


def test_1d_3d_lineage_dedup_counts_once():
    macro = [_lvl(50, "1w", "low"), _lvl(150, "1w", "high")]
    d = _frame([98, 99, 100, 99, 98], start="2024-06-01", freq="D")
    td = _frame([98.3, 99.3, 100.3, 99.3, 98.3], start="2024-06-01", freq="3D")  # ~same price
    refined = _refine(macro, d, td)
    assert len(refined) == 1
    assert refined[0].confirmations == 1     # lineage dedup: not 2
    assert refined[0].tf_source == "3d"      # higher rank of the lower pair


def test_sub_level_cap_per_bracket():
    macro = [_lvl(0, "1w", "low"), _lvl(1000, "1w", "high")]
    highs = [98, 99, 100, 99, 98, 198, 199, 200, 199, 198,
             298, 299, 300, 299, 298, 398, 399, 400, 399, 398]
    daily = _frame(highs, start="2024-06-01")  # four distinct inside pivots
    assert len(_refine(macro, daily, n_sub=3)) == 3   # capped from 4


# ── VRVP confluence ──────────────────────────────────────────────────────────────

def test_score_with_vrvp_flags_value_area_membership_only():
    levels = [_lvl(100, "1w"), _lvl(200, "1w")]
    scored = _score_with_vrvp(levels, VolumeProfile(vah=150.0, val=90.0, poc=110.0))
    assert len(scored) == 2                   # never creates or drops a level
    by_price = {l.price: l.vrvp_confluence for l in scored}
    assert by_price[100.0] is True            # inside [90,150]
    assert by_price[200.0] is False           # outside


def test_score_with_vrvp_nan_profile_clears_flag():
    scored = _score_with_vrvp([_lvl(100, "1w", vrvp=True)], VolumeProfile(float("nan"), float("nan"), float("nan")))
    assert scored[0].vrvp_confluence is False


def test_vrvp_confluence_uses_merged_final_price_not_a_contributor():
    # 1w@100 + 1mo@101 merge to one level at 101 (1mo anchor). The flag reflects the
    # FINAL price 101, set post-merge — NOT OR'd from the 1w@100 contributor that
    # straddled the value area. (Regression for the dead merge-time OR-combine.)
    w = _frame([96, 98, 100, 98, 96], freq="7D")
    m = _frame([97, 99, 101, 99, 97], freq="30D")
    out_false = derive_sr_levels(weekly=w, monthly=m, vrvp=VolumeProfile(vah=100.5, val=99.0, poc=100.0))
    lvl = next(l for l in out_false if abs(l.price - 101) < 1)
    assert lvl.price == 101.0 and lvl.vrvp_confluence is False   # 101 ∉ [99,100.5]
    out_true = derive_sr_levels(weekly=w, monthly=m, vrvp=VolumeProfile(vah=102.0, val=99.0, poc=100.5))
    assert next(l for l in out_true if abs(l.price - 101) < 1).vrvp_confluence is True  # 101 ∈ [99,102]


def test_anchor_vrvp_profiles_full_suffix_not_trailing_60():
    # Heavy volume at price 100 lives in the EARLY suffix; light price-200 bars after.
    prices = [100.0] * 16 + [200.0] * 64
    vols = [1000.0] * 16 + [1.0] * 64
    df = _frame(prices, start="2024-01-01")
    df["Volume"] = vols
    full = anchor_vrvp_for_range(df, 0)
    trailing60 = compute_volume_profile(df)        # default bars=60 → misses the early heavy zone
    assert full.poc < 150.0                          # full suffix → POC near the early 100 cluster
    assert trailing60.poc > 150.0                    # trailing 60 → POC near 200 (proves the difference)


def test_anchor_vrvp_empty_suffix_is_nan_not_a_raise():
    vp = anchor_vrvp_for_range(_frame([1, 2, 3]), 99)
    assert math.isnan(vp.poc)


# ── quality ordering ─────────────────────────────────────────────────────────────

def test_quality_rank_dominates_confirmations():
    a, b = _assign_quality([_lvl(10, "1mo", confirmations=1), _lvl(20, "1d", confirmations=3)])
    assert a.quality > b.quality                    # rank 3 beats rank 0 despite higher confirmations


def test_quality_increases_with_confirmations_then_vrvp():
    weak, strong = _assign_quality([_lvl(10, "1w", confirmations=1), _lvl(20, "1w", confirmations=2)])
    assert strong.quality > weak.quality
    no_vp, vp = _assign_quality([_lvl(10, "1w", confirmations=1), _lvl(20, "1w", confirmations=1, vrvp=True)])
    assert vp.quality > no_vp.quality


# ── the core architectural claim + adversarial anti-vacuity ──────────────────────

def _zig(center, jitter):
    """Half-period-8 zigzag (±7 clearance for find_swing_points lookback=5), 2 cycles.

    Small per-bar/per-cycle offsets keep extrema unique so the v1 tie-suppression
    (Known Limitation #5) doesn't blank the baseline.
    """
    lo, hi = center - 10.0, center
    seg = ([lo + (hi - lo) * i / 8 for i in range(8)] + [hi]
           + [hi - (hi - lo) * i / 8 for i in range(1, 8)])
    out = []
    for k in range(2):
        out += [round(v + jitter + k * 0.25, 3) for v in seg]
    return out


def _local_cluster_setup():
    # Daily has two price bands (~100/~90, ~200) → v1 clustering finds 3 local
    # levels. Macro (single tent in W @50 + single tent in M @60, no spurious
    # valleys) is two levels far from the daily action.
    daily = _frame(_zig(100, 0) + _zig(200, 0), start="2024-06-01")
    kw = {
        "weekly": _frame([46, 48, 50, 48, 46], freq="7D"),
        "monthly": _frame([56, 58, 60, 58, 56], freq="30D"),
        "daily": daily,
    }
    return daily, kw


def test_top_down_emits_fewer_levels_than_local_clustering():
    daily, kw = _local_cluster_setup()
    top_down = derive_sr_levels(**kw)
    clustered = cluster_sr_levels(find_swing_points(daily))
    assert len(clustered) >= 2                       # the real v1 baseline finds levels
    assert len(top_down) < len(clustered)            # top-down is structurally fewer


def test_adversarial_attractive_local_level_is_rejected_top_down():
    # v1 clustering finds a strong ~100 level; macro structure is 50/60, so ~100
    # falls outside every bracket → top-down must NOT emit it.
    daily, kw = _local_cluster_setup()
    clustered = cluster_sr_levels(find_swing_points(daily))
    top_down = derive_sr_levels(**kw)
    assert any(abs(l.price - 100) <= 5 for l in clustered)        # v1 is attracted to ~100
    assert not any(95 <= l.price <= 205 for l in top_down)        # top-down rejects it (no bracket)


# ── orchestration + degradation ──────────────────────────────────────────────────

def test_derive_end_to_end_sorted_and_scored():
    levels = derive_sr_levels(
        weekly=_frame([40, 45, 50, 45, 40, 140, 145, 150, 145, 140], freq="7D"),
        daily=_frame([98, 99, 100, 99, 98], start="2024-06-01"),
        vrvp=VolumeProfile(vah=160.0, val=40.0, poc=100.0),
    )
    assert [l.price for l in levels] == sorted(l.price for l in levels)  # price-sorted
    assert all(l.quality > 0 for l in levels)
    assert any(abs(l.price - 100) < 1 and l.vrvp_confluence for l in levels)  # refined + in value area


def test_degradation_no_frames_and_all_empty():
    assert derive_sr_levels() == []
    assert derive_sr_levels(weekly=_empty(), monthly=_empty(), daily=_empty(), three_day=_empty()) == []


def test_merge_keeps_disjoint_macro_and_refined():
    macro = [_lvl(50, "1w", "low"), _lvl(150, "1w", "high")]
    refined = [_lvl(100, "1d", formed="2024-06-01")]
    merged = _merge_sr_levels(macro, refined, tol=0.01)
    assert sorted(l.price for l in merged) == [50.0, 100.0, 150.0]
