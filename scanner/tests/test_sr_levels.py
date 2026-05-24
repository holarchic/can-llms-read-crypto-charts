"""Tests for scanner/sr_levels.py (T-019 Chunk 3, top-down S/R substrate).

Tier 1 — synthetic multi-TF (the primary gate). Fixed-date fixtures (never
``Timestamp.now()``); hand-crafted, not RNG. The "fewer-but-higher-quality than
clustering" claim drives the REAL ``cluster_sr_levels`` (not a mimic), and includes
an adversarial fixture where local clustering finds an attractive level the
top-down contract must reject. Tier 2 (real-chart fixture band) is a separate file
built when the W/M/D/3D candle fixtures are committed.

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
    anchor_vrvp_for_range,
    derive_sr_levels,
    find_htf_pivots,
    merge_sr_levels,
    refine_pivots_on_lower_tf,
    score_with_vrvp,
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


def _lvl(price, tf, ptype="high", strength=1, vrvp=False, formed="2024-01-01", last=None):
    f = pd.Timestamp(formed)
    return SRLevelV2(
        price=float(price), tf_source=tf, pivot_type=ptype, strength=strength,
        vrvp_confluence=vrvp, formed_at=f, last_test=pd.Timestamp(last) if last else f,
    )


# ── pivots ────────────────────────────────────────────────────────────────────

def test_single_tent_yields_one_high_pivot():
    pivots = find_htf_pivots(_frame([96, 98, 100, 98, 96], freq="7D"), _empty())
    assert len(pivots) == 1
    p = pivots[0]
    assert p.pivot_type == "high" and p.price == 100.0 and p.tf_source == "1w"


def test_tie_tolerant_plateau_keeps_most_recent_timestamp():
    # window=1, two equal maxima at idx1,idx2 → ONE pivot at the LATER bar (idx2).
    df = _frame([1, 5, 5, 1, 1], lows=[0.5, 0.4, 0.3, 0.4, 0.5], start="2024-03-01", freq="7D")
    highs = [p for p in find_htf_pivots(df, _empty(), window=1) if p.pivot_type == "high"]
    assert len(highs) == 1
    assert highs[0].formed_at == df.index[2]   # later of the tied bars, not idx1
    assert highs[0].formed_at != df.index[1]


def test_short_frame_yields_no_pivots_not_a_raise():
    assert find_htf_pivots(_frame([1, 2, 3]), _empty()) == []


def test_w_and_m_at_same_price_merge_to_one_higher_rank_level():
    w = _frame([96, 98, 100, 98, 96], start="2024-01-01", freq="7D")
    m = _frame([97, 99, 100, 99, 97], start="2024-01-01", freq="30D")
    levels = find_htf_pivots(w, m)
    at_100 = [l for l in levels if abs(l.price - 100) < 1]
    assert len(at_100) == 1
    assert at_100[0].tf_source == "1mo"   # higher rank wins
    assert at_100[0].strength == 2        # two distinct lineages (1w + 1mo)


# ── refinement ──────────────────────────────────────────────────────────────────

def test_refine_includes_pivot_strictly_inside_bracket():
    macro = [_lvl(50, "1w", "low", formed="2024-01-01"), _lvl(150, "1w", "high", formed="2024-01-01")]
    daily = _frame([98, 99, 100, 99, 98], start="2024-06-01")  # forms after the macro edges
    refined = refine_pivots_on_lower_tf(macro, daily, _empty())
    assert len(refined) == 1 and refined[0].price == 100.0 and refined[0].tf_source == "1d"


def test_refine_excludes_pivot_within_tol_of_a_macro_edge():
    macro = [_lvl(50, "1w", "low", formed="2024-01-01"), _lvl(150, "1w", "high", formed="2024-01-01")]
    daily = _frame([50.2, 50.3, 50.4, 50.3, 50.2], start="2024-06-01")  # 50.4 within 1% of edge 50
    assert refine_pivots_on_lower_tf(macro, daily, _empty()) == []


def test_refine_excludes_pivot_formed_before_later_macro_edge():
    macro = [_lvl(50, "1w", "low", formed="2024-09-01"), _lvl(150, "1w", "high", formed="2024-09-01")]
    daily = _frame([98, 99, 100, 99, 98], start="2024-01-01")  # forms BEFORE the macro edges
    assert refine_pivots_on_lower_tf(macro, daily, _empty()) == []


def test_1d_3d_lineage_dedup_counts_once():
    macro = [_lvl(50, "1w", "low", formed="2024-01-01"), _lvl(150, "1w", "high", formed="2024-01-01")]
    d = _frame([98, 99, 100, 99, 98], start="2024-06-01", freq="D")
    td = _frame([98.3, 99.3, 100.3, 99.3, 98.3], start="2024-06-01", freq="3D")  # ~same price
    refined = refine_pivots_on_lower_tf(macro, d, td)
    assert len(refined) == 1
    assert refined[0].strength == 1          # lineage dedup: not 2
    assert refined[0].tf_source == "3d"      # higher rank of the lower pair


def test_sub_level_cap_per_bracket():
    macro = [_lvl(0, "1w", "low", formed="2024-01-01"), _lvl(1000, "1w", "high", formed="2024-01-01")]
    highs = [98, 99, 100, 99, 98, 198, 199, 200, 199, 198,
             298, 299, 300, 299, 298, 398, 399, 400, 399, 398]
    daily = _frame(highs, start="2024-06-01")  # four distinct inside pivots
    refined = refine_pivots_on_lower_tf(macro, daily, _empty(), n_sub=3)
    assert len(refined) == 3                  # capped from 4


# ── VRVP confluence ──────────────────────────────────────────────────────────────

def test_score_with_vrvp_flags_value_area_membership_only():
    levels = [_lvl(100, "1w"), _lvl(200, "1w")]
    scored = score_with_vrvp(levels, VolumeProfile(vah=150.0, val=90.0, poc=110.0))
    assert len(scored) == 2                   # never creates or drops a level
    by_price = {l.price: l.vrvp_confluence for l in scored}
    assert by_price[100.0] is True            # inside [90,150]
    assert by_price[200.0] is False           # outside


def test_score_with_vrvp_nan_profile_clears_flag():
    scored = score_with_vrvp([_lvl(100, "1w", vrvp=True)], VolumeProfile(float("nan"), float("nan"), float("nan")))
    assert scored[0].vrvp_confluence is False


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

def test_quality_rank_dominates_strength():
    a, b = _assign_quality([_lvl(10, "1mo", strength=1), _lvl(20, "1d", strength=3)])
    assert a.quality > b.quality                    # rank 3 beats rank 0 despite higher strength


def test_quality_increases_with_strength_then_vrvp():
    weak, strong = _assign_quality([_lvl(10, "1w", strength=1), _lvl(20, "1w", strength=2)])
    assert strong.quality > weak.quality
    no_vp, vp = _assign_quality([_lvl(10, "1w", strength=1), _lvl(20, "1w", strength=1, vrvp=True)])
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


def _local_cluster_frames():
    # Daily has two price bands (~100/~90, ~200) → v1 clustering finds 3 local
    # levels. Macro (single tent in W @50 + single tent in M @60, no spurious
    # valleys) is two levels far from the daily action.
    daily = _frame(_zig(100, 0) + _zig(200, 0), start="2024-06-01")
    frames = {
        "1w": _frame([46, 48, 50, 48, 46], freq="7D"),
        "1mo": _frame([56, 58, 60, 58, 56], freq="30D"),
        "1d": daily, "3d": _empty(),
    }
    return daily, frames


def test_top_down_emits_fewer_levels_than_local_clustering():
    daily, frames = _local_cluster_frames()
    top_down = derive_sr_levels(frames)
    clustered = cluster_sr_levels(find_swing_points(daily))
    assert len(clustered) >= 2                       # the real v1 baseline finds levels
    assert len(top_down) < len(clustered)            # top-down is structurally fewer


def test_adversarial_attractive_local_level_is_rejected_top_down():
    # v1 clustering finds a strong ~100 level; macro structure is 50/60, so ~100
    # falls outside every bracket → top-down must NOT emit it.
    daily, frames = _local_cluster_frames()
    clustered = cluster_sr_levels(find_swing_points(daily))
    top_down = derive_sr_levels(frames)
    assert any(abs(l.price - 100) <= 5 for l in clustered)        # v1 is attracted to ~100
    assert not any(95 <= l.price <= 205 for l in top_down)        # top-down rejects it (no bracket)


# ── orchestration + degradation ──────────────────────────────────────────────────

def test_derive_end_to_end_sorted_and_scored():
    frames = {
        "1w": _frame([40, 45, 50, 45, 40, 140, 145, 150, 145, 140], freq="7D"),
        "1mo": _empty(),
        "1d": _frame([98, 99, 100, 99, 98], start="2024-06-01"),
        "3d": _empty(),
    }
    levels = derive_sr_levels(frames, vrvp=VolumeProfile(vah=160.0, val=40.0, poc=100.0))
    assert [l.price for l in levels] == sorted(l.price for l in levels)  # price-sorted
    assert all(l.quality > 0 for l in levels)
    assert any(abs(l.price - 100) < 1 and l.vrvp_confluence for l in levels)  # refined + in value area


def test_degradation_empty_and_missing_frames():
    assert derive_sr_levels({}) == []
    assert derive_sr_levels({"1w": _empty(), "1mo": _empty(), "1d": _empty(), "3d": _empty()}) == []


def test_merge_keeps_disjoint_macro_and_refined():
    macro = [_lvl(50, "1w", "low"), _lvl(150, "1w", "high")]
    refined = [_lvl(100, "1d", formed="2024-06-01")]
    merged = merge_sr_levels(macro, refined)
    assert sorted(l.price for l in merged) == [50.0, 100.0, 150.0]
