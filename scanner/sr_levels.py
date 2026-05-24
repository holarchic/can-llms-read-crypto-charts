"""Top-down support/resistance for the v10 v2 framework (T-019 Chunk 3).

the framework works S/R top-down (his words, synthesis 2026-03-10): weekly/monthly macro
pivots define the structure, then daily/3D refine *within* it; VRVP only adds
confluence to a level, never creates one. This module is that derivation, opposite
to v1's local-swing ``cluster_sr_levels`` (which flattens every 2+-touch cluster
with no provenance). The output is structurally *fewer, higher-quality* levels.

Pure functions, no I/O. Substrate-only: nothing wires this into the pipeline yet
(Chunk 7), and the range-detection that supplies an anchor index to
``anchor_vrvp_for_range`` lives in Chunk 4 (``regime``) — here that function is the
pure ``(df, anchor_idx) -> VolumeProfile`` primitive.

Algorithm contract (provenance precedence ``1mo > 1w > 3d > 1d``):
1. macro = tie-tolerant W & M fractal pivots, merged by price.
2. brackets = gaps between adjacent macro levels (sorted by price).
3. a D/3D pivot refines a bracket iff its price is *strictly* inside (not within
   ``tol`` of either edge — edge touches reinforce the macro level instead) AND it
   formed at/after the later-forming of the two bracket edges. Because refined
   sub-levels are strictly inside, they never coincide with macro levels — the
   three stages produce price-disjoint sets, so the final merge cannot double-count.
4. ``1d`` and locally-resampled ``3d`` are lineage siblings — a same-price pair
   counts once (``strength`` is a distinct-lineage count, ``1d``/``3d`` → ``ltf``).
5. at most ``n_sub`` refined sub-levels per bracket (top by strength then recency).
6. merge unions macro + refined and dedups by price (higher rank wins, strength
   never decreases — ``max(best, distinct-lineages)`` — ``vrvp_confluence``
   OR-combines, ``last_test`` = latest, ``formed_at`` = the dominant contributor's).
7. ``quality`` is monotonic in rank → strength → vrvp → recency, with bounded bands
   so the scalar is lexicographic-consistent with that priority.

Degradation: missing/short ``1w``/``1mo`` history yields fewer/zero macro pivots →
fewer/zero brackets → possibly an empty result, never a raise. Thresholds
(``window`` / ``tol`` / ``n_sub``) are v1 defaults, calibrated in Chunk 8b. Per-TF
multi-touch beyond distinct lineage is not separately counted in v1 strength.
"""

from dataclasses import replace

import pandas as pd

from .ma_indicators import VolumeProfile, compute_volume_profile
from .models import SRLevelV2

_TF_RANK = {"1mo": 3, "1w": 2, "3d": 1, "1d": 0}
_DEFAULT_TOL = 0.01      # price-cluster width, relative (1%) — v1 default
_DEFAULT_WINDOW = 2      # fractal half-width on sparse HTF bars — v1 default
_DEFAULT_N_SUB = 3       # max refined sub-levels per macro bracket — v1 default


def _lineage_key(tf_source: str) -> str:
    """``1d`` and ``3d`` share a lineage (3d is resampled 1d); W and M are independent."""
    return "ltf" if tf_source in ("1d", "3d") else tf_source


def _within(a: float, b: float, tol: float) -> bool:
    """True when prices ``a`` and ``b`` are within ``tol`` (relative) of each other."""
    anchor = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / anchor <= tol


def _pivot(price: float, tf_source: str, pivot_type: str, ts) -> SRLevelV2:
    t = pd.Timestamp(ts)
    return SRLevelV2(
        price=float(price), tf_source=tf_source, pivot_type=pivot_type,
        strength=1, vrvp_confluence=False, formed_at=t, last_test=t,
    )


def _fractal_pivots(df: pd.DataFrame, tf_source: str, window: int) -> list[SRLevelV2]:
    """Tie-tolerant fractal pivots from one TF frame (wick extremes).

    A bar is a pivot-high when its ``High`` equals the max over the centered
    ±``window`` band; a flat-top plateau collapses to a single pivot carrying the
    plateau's *most-recent* timestamp (left-edge would backdate the time that
    bracket-eligibility depends on). Symmetric for pivot-lows on ``Low``. A frame
    too short for a centered window yields no pivots (never a raise).
    """
    n = len(df)
    if n < 2 * window + 1:
        return []
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    index = df.index

    pivots: list[SRLevelV2] = []
    for kind, series, is_max in (("high", highs, True), ("low", lows, False)):
        run: list[int] = []  # consecutive plateau candidates at the same extreme
        for i in range(window, n - window):
            band = series[i - window: i + window + 1]
            extreme = band.max() if is_max else band.min()
            if series[i] != extreme:
                continue
            if run and series[i] == series[run[-1]] and i - run[-1] <= window:
                run.append(i)  # same flat-top → extend, most-recent wins
            else:
                if run:
                    pivots.append(_pivot(series[run[-1]], tf_source, kind, index[run[-1]]))
                run = [i]
        if run:
            pivots.append(_pivot(series[run[-1]], tf_source, kind, index[run[-1]]))
    return pivots


def _combine(cluster: list[SRLevelV2]) -> SRLevelV2:
    """Collapse one price cluster per contract clause 6 (strength never decreases)."""
    best = max(cluster, key=lambda l: (_TF_RANK[l.tf_source], l.strength, l.last_test))
    lineages = {_lineage_key(l.tf_source) for l in cluster}
    return SRLevelV2(
        price=best.price,
        tf_source=best.tf_source,
        pivot_type=best.pivot_type,
        strength=max(best.strength, len(lineages)),
        vrvp_confluence=any(l.vrvp_confluence for l in cluster),
        formed_at=best.formed_at,
        last_test=max(l.last_test for l in cluster),
    )


def _merge_by_price(levels: list[SRLevelV2], tol: float) -> list[SRLevelV2]:
    """Group levels within ``tol`` (anchored on each cluster's lowest price) and combine."""
    if not levels:
        return []
    ordered = sorted(levels, key=lambda l: l.price)
    clusters: list[list[SRLevelV2]] = [[ordered[0]]]
    for lvl in ordered[1:]:
        if _within(lvl.price, clusters[-1][0].price, tol):
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])
    return [_combine(c) for c in clusters]


def find_htf_pivots(
    df_weekly: pd.DataFrame, df_monthly: pd.DataFrame,
    *, window: int = _DEFAULT_WINDOW, tol: float = _DEFAULT_TOL,
) -> list[SRLevelV2]:
    """Macro pivots: tie-tolerant W & M fractal pivots merged by price (clause 1)."""
    pivots = _fractal_pivots(df_weekly, "1w", window) + _fractal_pivots(df_monthly, "1mo", window)
    return _merge_by_price(pivots, tol)


def refine_pivots_on_lower_tf(
    htf_pivots: list[SRLevelV2], df_daily: pd.DataFrame, df_3d: pd.DataFrame,
    *, window: int = _DEFAULT_WINDOW, n_sub: int = _DEFAULT_N_SUB, tol: float = _DEFAULT_TOL,
) -> list[SRLevelV2]:
    """Daily/3D sub-levels strictly inside macro brackets (clauses 2–5).

    Lower-TF pivots are merged first so ``1d``/``3d`` lineage siblings collapse
    (clause 4); each bracket keeps at most ``n_sub`` by strength then recency.
    """
    if len(htf_pivots) < 2:
        return []
    macro = sorted(htf_pivots, key=lambda l: l.price)
    lower = _merge_by_price(
        _fractal_pivots(df_daily, "1d", window) + _fractal_pivots(df_3d, "3d", window),
        tol,
    )
    refined: list[SRLevelV2] = []
    for low, high in zip(macro, macro[1:]):
        edge_t = max(low.formed_at, high.formed_at)
        inside = [
            lvl for lvl in lower
            if low.price < lvl.price < high.price
            and not _within(lvl.price, low.price, tol)
            and not _within(lvl.price, high.price, tol)
            and lvl.formed_at >= edge_t
        ]
        inside.sort(key=lambda l: (l.strength, l.last_test), reverse=True)
        refined.extend(inside[:n_sub])
    return refined


def score_with_vrvp(levels: list[SRLevelV2], vrvp: VolumeProfile) -> list[SRLevelV2]:
    """Flag ``vrvp_confluence`` where a level sits in the value area ``[VAL, VAH]``.

    Confluence only — never creates or moves a level. POC lies inside the value
    area, so value-area membership is the whole rule (no separate POC tolerance).
    """
    if vrvp is None or pd.isna(vrvp.val) or pd.isna(vrvp.vah):
        return [replace(l, vrvp_confluence=False) for l in levels]
    lo, hi = min(vrvp.val, vrvp.vah), max(vrvp.val, vrvp.vah)
    return [replace(l, vrvp_confluence=bool(lo <= l.price <= hi)) for l in levels]


def anchor_vrvp_for_range(df: pd.DataFrame, anchor_idx: int) -> VolumeProfile:
    """Volume profile over the *entire* suffix ``df.iloc[anchor_idx:]``.

    The Chunk-1 ``compute_volume_profile`` defaults to the trailing 60 bars; this
    passes ``bars=len(suffix)`` so the whole post-anchor range is profiled, not the
    last 60. The *source* of ``anchor_idx`` (range/breakdown detection) is Chunk 4.
    """
    idx = max(0, min(int(anchor_idx), len(df)))
    suffix = df.iloc[idx:]
    if suffix.empty:
        return VolumeProfile(float("nan"), float("nan"), float("nan"))
    return compute_volume_profile(suffix, bars=len(suffix))


def merge_sr_levels(
    htf: list[SRLevelV2], refined: list[SRLevelV2], *, tol: float = _DEFAULT_TOL,
) -> list[SRLevelV2]:
    """Union macro + refined into one price-deduped set (clause 6)."""
    return _merge_by_price(list(htf) + list(refined), tol)


def _assign_quality(levels: list[SRLevelV2]) -> list[SRLevelV2]:
    """Set ``quality`` monotonic in rank → strength → vrvp → recency (clause 7).

    Bounded bands keep the scalar lexicographic-consistent: rank (×1000) dominates
    strength (×100, ≤ a few hundred), which dominates the vrvp flag (5) and the
    normalized recency (≤1). Numeric weights are v1 defaults, calibrated Chunk 8b.
    """
    if not levels:
        return []
    times = [pd.Timestamp(l.last_test).value for l in levels]
    lo, hi = min(times), max(times)
    span = hi - lo
    out = []
    for l in levels:
        recency = 1.0 if span == 0 else (pd.Timestamp(l.last_test).value - lo) / span
        q = (
            _TF_RANK[l.tf_source] * 1000.0
            + l.strength * 100.0
            + (5.0 if l.vrvp_confluence else 0.0)
            + recency
        )
        out.append(replace(l, quality=round(q, 4)))
    return out


def derive_sr_levels(
    frames: dict[str, pd.DataFrame], vrvp: VolumeProfile | None = None,
    *, window: int = _DEFAULT_WINDOW, tol: float = _DEFAULT_TOL, n_sub: int = _DEFAULT_N_SUB,
) -> list[SRLevelV2]:
    """Public seam: find → refine → merge → score → quality, sorted by price.

    ``frames`` is keyed by canonical TF (``1w``/``1mo``/``1d``/``3d``); a missing
    key degrades to no contribution from that TF rather than raising.
    """
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    macro = find_htf_pivots(frames.get("1w", empty), frames.get("1mo", empty), window=window, tol=tol)
    refined = refine_pivots_on_lower_tf(
        macro, frames.get("1d", empty), frames.get("3d", empty), window=window, n_sub=n_sub, tol=tol,
    )
    merged = merge_sr_levels(macro, refined, tol=tol)
    scored = score_with_vrvp(merged, vrvp) if vrvp is not None else merged
    return sorted(_assign_quality(scored), key=lambda l: l.price)
