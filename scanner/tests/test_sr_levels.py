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
import pytest

from scanner.indicators import cluster_sr_levels, find_swing_points
from scanner.ma_indicators import VolumeProfile, compute_volume_profile
from scanner.models import SRLevelV2
from scanner.sr_levels import (
    _MAX_RETESTS,
    _RETEST_DECAY,
    _assign_quality,
    _combine,
    _count_retests,
    _find_htf_pivots,
    _merge_sr_levels,
    _pivot,
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


# ── Sub-chunk F: zones, retest counting, decay ────────────────────────────────


def test_srlevel_v2_zone_fields_required():
    """No zone defaults — missing zone_low/zone_high at construction → TypeError."""
    with pytest.raises(TypeError):
        SRLevelV2(                                          # type: ignore[call-arg]
            price=100.0, tf_source="1d", pivot_type="high", confirmations=1,
            vrvp_confluence=False, formed_at=pd.Timestamp("2024-01-01"),
            last_test=pd.Timestamp("2024-01-01"),
        )


@pytest.mark.parametrize(
    "open_,high,low,close,ptype",
    [
        (95.0, 100.0, 94.0, 96.0, "high"),                  # bullish-bodied resistance
        (96.0, 100.0, 94.0, 95.0, "high"),                  # bearish-bodied resistance
        (104.0, 106.0, 100.0, 105.0, "low"),                # bullish-bodied support
        (105.0, 106.0, 100.0, 104.0, "low"),                # bearish-bodied support
    ],
)
def test_pivot_zone_derivation_invariant(open_, high, low, close, ptype):
    """Resistance: zone_high == high (= price); zone_low == max(open, close).
    Support: zone_low == low (= price); zone_high == min(open, close)."""
    lvl = _pivot(open_, high, low, close, "1d", ptype, pd.Timestamp("2024-01-01"))
    if ptype == "high":
        assert lvl.price == high
        assert lvl.zone_high == high
        assert lvl.zone_low == max(open_, close)
    else:
        assert lvl.price == low
        assert lvl.zone_low == low
        assert lvl.zone_high == min(open_, close)


@pytest.mark.parametrize("ptype", ["high", "low"])
def test_pivot_four_price_doji_gets_inward_min_width_pad(ptype):
    """O=H=L=C → zone padded ASYMMETRICALLY inward from the wick extreme so
    `price == zone-edge` invariant still holds. The pad extends INTO the candle-body
    side (the side the level "respects" from), never outward past the wick."""
    p = 100.0
    lvl = _pivot(p, p, p, p, "1d", ptype, pd.Timestamp("2024-01-01"))
    if ptype == "high":
        assert lvl.zone_high == p            # wick extreme preserved
        assert lvl.zone_low < p              # inward (body-side) pad
        assert lvl.price == lvl.zone_high
    else:
        assert lvl.zone_low == p
        assert lvl.zone_high > p             # inward pad above the wick low
        assert lvl.price == lvl.zone_low


def test_pivot_doji_zone_supports_at_sr_membership_for_adjacent_price():
    """A four-price doji's min-width zone is wide enough that the touch line + a
    tiny adjacent price both fall inside (AT_SR membership not collapsed)."""
    p = 100.0
    lvl = _pivot(p, p, p, p, "1d", "high", pd.Timestamp("2024-01-01"))
    # price right at the wick line is inside (touch line == zone_high boundary)
    assert lvl.zone_low <= p <= lvl.zone_high
    # a price just inward (the body side) is inside the pad
    assert lvl.zone_low <= (p - 1e-6) <= lvl.zone_high


def _zlvl(price, tf, ptype, zone_low, zone_high, *, confirmations=1, formed="2024-01-01"):
    """Construct a level with explicit zone bounds — used to drive `_combine` directly."""
    f = pd.Timestamp(formed)
    return SRLevelV2(
        price=float(price), tf_source=tf, pivot_type=ptype, confirmations=confirmations,
        vrvp_confluence=False, formed_at=f, last_test=f,
        zone_low=float(zone_low), zone_high=float(zone_high),
    )


def test_combine_outer_envelope_widens_zone_across_contributors():
    """Three clustered resistance pivots → merged zone is the outer envelope (min low,
    max high), wider than any single contributor."""
    a = _zlvl(100.0, "1d", "high", 99.5, 100.0)
    b = _zlvl(100.1, "1d", "high", 99.7, 100.1, confirmations=2)
    c = _zlvl(100.05, "3d", "high", 99.6, 100.05)
    merged = _combine([a, b, c])
    assert merged.zone_low == 99.5
    assert merged.zone_high == 100.1
    assert (merged.zone_high - merged.zone_low) > max(
        x.zone_high - x.zone_low for x in (a, b, c)
    )


def test_combine_max_width_guard_falls_back_to_best_contributor_zone():
    """1w wide-wick + 1d tight pivot at the same cluster price — the outer envelope
    is comically wide (>3% of mid-price), so `_combine` returns the best
    contributor's zone, not the union."""
    wide = _zlvl(100.0, "1w", "high", 95.0, 100.0, confirmations=1)   # 5% wide
    tight = _zlvl(100.05, "1d", "high", 99.8, 100.05, confirmations=2)
    merged = _combine([wide, tight])
    # best by (rank, confirmations, last_test) = 1w wins on rank
    assert merged.tf_source == "1w"
    assert merged.zone_low == 95.0
    assert merged.zone_high == 100.0


def test_combine_merged_price_pinned_to_zone_edge_resistance():
    """Resistance merge: `price == zone_high` (the wick line that SFP semantics read)."""
    a = _zlvl(100.0, "1d", "high", 99.5, 100.0)
    b = _zlvl(99.95, "1d", "high", 99.4, 99.95, confirmations=2)
    merged = _combine([a, b])
    assert merged.pivot_type == "high"
    assert merged.price == merged.zone_high      # SFP-pierce invariant


def test_combine_merged_price_pinned_to_zone_edge_support():
    """Support merge: `price == zone_low`."""
    a = _zlvl(100.0, "1d", "low", 100.0, 100.5)
    b = _zlvl(100.05, "1d", "low", 100.05, 100.6, confirmations=2)
    merged = _combine([a, b])
    assert merged.pivot_type == "low"
    assert merged.price == merged.zone_low


def test_merge_by_price_does_not_collapse_opposite_sides():
    """Codex correctness regression: a same-price `high` + `low` pair must NOT collapse
    into one merged record — the zone shape and `pivot_type` would describe a level that
    isn't a real wick-to-body cluster. `_merge_by_price` partitions by `pivot_type` first.
    """
    from scanner.sr_levels import _merge_by_price
    res = _zlvl(100.0, "1d", "high", 99.5, 100.0)
    sup = _zlvl(100.1, "1d", "low", 100.1, 100.6)
    merged = _merge_by_price([res, sup], tol=0.01)
    assert len(merged) == 2                                  # two side-segregated records
    by_type = {m.pivot_type for m in merged}
    assert by_type == {"high", "low"}
    high = next(m for m in merged if m.pivot_type == "high")
    low = next(m for m in merged if m.pivot_type == "low")
    assert (high.zone_low, high.zone_high) == (99.5, 100.0)  # resistance zone untouched
    assert (low.zone_low, low.zone_high) == (100.1, 100.6)   # support zone untouched


def test_combine_formed_at_is_latest_contributor_not_dominant():
    """Codex correctness regression: when a later, wider contributor merges with an
    earlier one, the merged `formed_at` is the LATER timestamp — so retest counting
    cannot back-date through bars that pre-date the widened zone shape."""
    early = _zlvl(100.0, "1w", "high", 99.5, 100.0, formed="2026-01-01")
    later = _zlvl(100.05, "1d", "high", 98.0, 100.05, formed="2026-03-01")
    merged = _combine([early, later])
    assert merged.formed_at == pd.Timestamp("2026-03-01")    # latest, not best's


def test_count_retests_excludes_bars_before_widened_zone_formed():
    """End-to-end of the formed_at fix: a February bar that sits inside the
    March-widened zone but NOT inside the January contributor's zone must not count as
    a retest of the merged record."""
    early = _zlvl(100.0, "1w", "high", 99.5, 100.0, formed="2026-01-01")
    later = _zlvl(100.05, "1d", "high", 98.0, 100.05, formed="2026-03-01")
    merged = _combine([early, later])
    # Build the frame with explicit (date, O, H, L, C) rows.
    dated = [
        ("2026-01-10", 95.0, 96.0, 94.0, 95.0),               # respect side, pre-widen (filtered by formed_at)
        ("2026-02-01", 98.2, 98.6, 97.8, 98.1),               # inside WIDENED zone only — false-positive retest pre-fix
        ("2026-02-10", 95.0, 96.0, 94.0, 95.0),               # respect side, still pre-formed_at
        ("2026-03-10", 95.0, 96.0, 94.0, 95.0),               # respect side, post-formation (counts as side state)
        ("2026-03-15", 98.3, 98.7, 97.9, 98.2),               # re-enter from respect side — retest 1
        ("2026-03-20", 95.0, 96.0, 94.0, 95.0),               # back to respect side
        ("2026-04-01", 98.3, 98.7, 97.9, 98.2),               # second post-formation re-entry — retest 2
    ]
    df = pd.DataFrame(
        {
            "Open":   [r[1] for r in dated],
            "High":   [r[2] for r in dated],
            "Low":    [r[3] for r in dated],
            "Close":  [r[4] for r in dated],
            "Volume": [1.0] * len(dated),
        },
        index=pd.to_datetime([r[0] for r in dated]),
    )
    # Post-fix: only the two re-entries after 2026-03-01 should count (Feb bar is
    # filtered out by `after = df[df.index > level.formed_at]`).
    assert _count_retests(merged, df) == 2


# ── _count_retests ────────────────────────────────────────────────────────────


def _retest_bars(rows, start="2024-01-10"):
    """Build a per-TF frame from a list of (open, high, low, close) tuples."""
    n = len(rows)
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open":   [r[0] for r in rows],
            "High":   [r[1] for r in rows],
            "Low":    [r[2] for r in rows],
            "Close":  [r[3] for r in rows],
            "Volume": [1.0] * n,
        },
        index=idx,
    )


def _resistance(zone_low=99.5, zone_high=100.0, formed="2024-01-01"):
    return _zlvl(zone_high, "1d", "high", zone_low, zone_high, formed=formed)


def test_count_retests_zero_when_price_never_returns():
    """Price drops away and never re-enters the zone → 0 retests."""
    lvl = _resistance()
    df = _retest_bars([(95.0, 96.0, 94.0, 95.0)] * 6)
    assert _count_retests(lvl, df) == 0


def test_count_retests_three_same_side_retests_at_resistance():
    """Price oscillates: respect-side (below) → into zone → respect-side → into zone …
    Three completed re-entry cycles count as 3 retests."""
    lvl = _resistance()
    # bar layout per cycle: drop below zone (respect_side), re-enter zone
    cycle = [
        (95.0, 96.0, 94.0, 95.0),               # respect-side (high 96 < zone_low 99.5)
        (98.0, 99.8, 97.5, 99.7),               # back inside zone (low 97.5 <= 100, high 99.8 >= 99.5)
    ]
    df = _retest_bars(cycle * 3)
    assert _count_retests(lvl, df) == 3


def test_count_retests_stops_at_flip_close_above_resistance():
    """A close ABOVE the zone (acceptance, not a wick poke) breaks the level → counting
    halts there. Same-side retests after the flip are not credited."""
    lvl = _resistance()
    rows = [
        (95.0, 96.0, 94.0, 95.0),               # respect-side
        (98.0, 99.8, 97.5, 99.7),               # retest 1
        (95.0, 96.0, 94.0, 95.0),               # respect-side
        (99.0, 101.0, 98.0, 100.5),             # FLIP: close 100.5 > zone_high 100
        (95.0, 96.0, 94.0, 95.0),               # would-be retest, ignored
        (99.0, 99.9, 97.0, 99.8),
    ]
    assert _count_retests(lvl, _retest_bars(rows)) == 1


def test_count_retests_wick_above_resistance_is_not_a_flip():
    """A single-bar WICK above the zone (close still inside / below) is NOT a flip —
    the framework's acceptance is close-based, not wick-based. Counting continues."""
    lvl = _resistance()
    rows = [
        (95.0, 96.0, 94.0, 95.0),               # respect-side
        (98.0, 100.5, 97.5, 99.7),              # wick to 100.5 above zone_high but close 99.7 inside
        (95.0, 96.0, 94.0, 95.0),               # respect-side again
        (98.0, 99.8, 97.5, 99.7),               # retest cycle 2
    ]
    assert _count_retests(lvl, _retest_bars(rows)) == 2


def test_count_retests_caps_at_max_plus_one():
    """The counter stops as soon as it exceeds ``_MAX_RETESTS`` — caller drops the level."""
    lvl = _resistance()
    cycle = [
        (95.0, 96.0, 94.0, 95.0),
        (98.0, 99.8, 97.5, 99.7),
    ]
    df = _retest_bars(cycle * (_MAX_RETESTS + 3))
    rc = _count_retests(lvl, df)
    assert rc > _MAX_RETESTS                       # marks the level for drop
    assert rc <= _MAX_RETESTS + 1                  # early-exits at the cap, never further


def test_count_retests_respect_side_for_support():
    """For a support (pivot_type='low'), respect_side is ABOVE — price oscillates
    above the zone and re-enters from above."""
    lvl = _zlvl(95.0, "1d", "low", 95.0, 95.5)
    rows = [
        (97.0, 98.0, 96.0, 97.5),               # respect-side (low 96 > zone_high 95.5)
        (95.4, 95.4, 95.0, 95.2),               # re-enter zone from above
        (97.0, 98.0, 96.0, 97.5),               # respect-side
        (95.4, 95.4, 95.0, 95.2),               # retest cycle 2
    ]
    assert _count_retests(lvl, _retest_bars(rows)) == 2


def test_count_retests_empty_or_missing_frame_yields_zero():
    """Degraded inputs never raise — a missing or empty per-TF frame returns 0."""
    lvl = _resistance()
    assert _count_retests(lvl, None) == 0
    assert _count_retests(lvl, _empty()) == 0


# ── pipeline integration: drop > _MAX_RETESTS + quality decay ──────────────────


def _high_retest_daily_frame(zone_low=99.5, zone_high=100.0, formed="2024-01-01"):
    """A daily frame whose price oscillates around a 99.5–100 resistance enough to
    exceed `_MAX_RETESTS` — used to verify drop + decay end-to-end through
    `derive_sr_levels`. The frame starts with a 5-bar tent at the resistance so
    `_fractal_pivots` finds the pivot, then oscillates well past the cap."""
    tent = [(99.0, 99.5, 98.5, 99.0),
            (99.5, 99.8, 99.2, 99.7),
            (99.8, 100.0, 99.4, 99.6),          # pivot apex
            (99.5, 99.8, 99.2, 99.7),
            (99.0, 99.5, 98.5, 99.0)]
    cycle = [(95.0, 96.0, 94.0, 95.0), (98.0, 99.8, 97.5, 99.7)]
    return _retest_bars(tent + cycle * (_MAX_RETESTS + 4), start=formed)


def test_apply_retest_decay_quality_scales_by_decay_power():
    """A level with retest_count=3 has quality = base * 0.7^3 (rest of the score
    structure unchanged)."""
    fresh = _lvl(100, "1d")                              # retest_count default 0
    aged = SRLevelV2(
        price=fresh.price, tf_source="1d", pivot_type="high", confirmations=1,
        vrvp_confluence=False, formed_at=fresh.formed_at, last_test=fresh.last_test,
        zone_low=fresh.zone_low, zone_high=fresh.zone_high, retest_count=3,
    )
    scored = _assign_quality([fresh, aged])
    f, a = scored[0], scored[1]
    assert a.quality == pytest.approx(f.quality * (_RETEST_DECAY ** 3), rel=1e-6)


def test_derive_drops_level_when_retest_count_exceeds_max():
    """End-to-end through `derive_sr_levels`: a resistance with >5 same-side retests
    on its daily frame is absent from the output (and only that level)."""
    daily = _high_retest_daily_frame()
    # Provide a weekly to anchor brackets so the daily pivot is allowed to refine inside.
    weekly = _frame([80, 85, 90, 85, 80, 110, 115, 120, 115, 110], freq="7D")
    levels = derive_sr_levels(weekly=weekly, daily=daily)
    assert not any(99.0 <= l.price <= 100.5 for l in levels)
