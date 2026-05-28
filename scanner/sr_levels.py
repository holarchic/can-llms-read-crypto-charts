"""Top-down support/resistance for the v10 v2 framework (T-019 Chunk 3).

the framework works S/R top-down (his words, synthesis 2026-03-10): weekly/monthly macro
pivots define the structure, then daily/3D refine *within* it; VRVP only adds
confluence to a level, never creates one. This module is that derivation, opposite
to v1's local-swing ``cluster_sr_levels`` (which flattens every 2+-touch cluster
with no provenance). The output is structurally *fewer, higher-quality* levels.

Public surface = ``derive_sr_levels`` (the supported seam) + ``anchor_vrvp_for_range``
(the Chunk-4 VRVP-anchor primitive). The derivation stages are private — callers
get the whole pipeline, not partial assemblies against internal sequencing.

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
   counts once (``confirmations`` is a distinct-lineage count, ``1d``/``3d`` → ``ltf``).
5. at most ``n_sub`` refined sub-levels per bracket (top by confirmations then recency).
6. merge unions macro + refined and dedups by price (higher rank wins, confirmations
   never decreases — ``max(best, distinct-lineages)``, ``last_test`` = latest,
   ``formed_at`` = the dominant contributor's). VRVP confluence is **not** carried
   through merge: it is set once, *after* merging, by ``_score_with_vrvp`` on each
   final level's price — a merged level reflects its own (post-merge) price, not a
   contributor that straddled a value-area edge.
7. ``quality`` is monotonic in rank → confirmations → vrvp → recency, with bounded
   bands so the scalar is lexicographic-consistent with that priority.

Pipeline order in ``derive_sr_levels``: find → refine → merge → score → retest-decay
→ quality. The retest-decay pass (Sub-chunk F) counts post-formation same-side
re-entries into each level's zone and drops `retest_count > _MAX_RETESTS`; `quality`
is then decayed by `_RETEST_DECAY ** retest_count` so fresh zones outrank retested
ones (framework-canonical "fresh = highest probability").

Degradation: missing/short ``1w``/``1mo`` history yields fewer/zero macro pivots →
fewer/zero brackets → possibly an empty result, never a raise. Thresholds
(``window`` / ``tol`` / ``n_sub``) are v1 defaults, calibrated in Chunk 8b. Per-TF
multi-touch beyond distinct lineage is not separately counted in ``confirmations``.
"""

import logging
from dataclasses import replace

import pandas as pd

from .ma_indicators import VolumeProfile, compute_volume_profile
from .models import SRLevelV2

logger = logging.getLogger(__name__)

# Public surface (enforced): the supported pipeline seam + the Chunk-4 VRVP primitive.
# The derivation stages are private (`_find_htf_pivots` etc.); unit tests reach them
# deliberately, but no production caller should.
__all__ = ["derive_sr_levels", "anchor_vrvp_for_range"]

_TF_RANK = {"1mo": 3, "1w": 2, "3d": 1, "1d": 0}
_DEFAULT_TOL = 0.01      # price-cluster width, relative (1%) — v1 default
_DEFAULT_WINDOW = 2      # fractal half-width on sparse HTF bars — v1 default
_DEFAULT_N_SUB = 3       # max refined sub-levels per macro bracket — v1 default
_RETEST_DECAY = 0.7      # per-retest multiplicative quality decay (framework-canonical)
_MAX_RETESTS = 5         # > 5 same-side retests → drop (the framework: "8 is too many")
_MIN_ZONE_PCT = 0.001    # 0.1% of pivot price — min zone width pad (avoid zero-width)
_MAX_ZONE_PCT = 0.03     # 3% of mid-price — max merged-cluster width before best-fallback


def _lineage_key(tf_source: str) -> str:
    """``1d`` and ``3d`` share a lineage (3d is resampled 1d); W and M are independent."""
    return "ltf" if tf_source in ("1d", "3d") else tf_source


def _within(a: float, b: float, tol: float) -> bool:
    """True when prices ``a`` and ``b`` are within ``tol`` (relative) of each other."""
    anchor = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / anchor <= tol


def _pivot(
    open_: float, high: float, low: float, close: float,
    tf_source: str, pivot_type: str, ts,
) -> SRLevelV2:
    """One bar → one zoned pivot. Zone is framework-canonical (resistance: body-top → wick;
    support: wick → body-bottom). Min-width pad is *asymmetric*: it extends INWARD from
    the wick extreme so the ``price == zone-edge`` invariant holds even for a
    four-price doji where O=H=L=C.
    """
    t = pd.Timestamp(ts)
    h, lo, op, cl = float(high), float(low), float(open_), float(close)
    if pivot_type == "high":
        price = h
        zone_high = h
        zone_low_raw = max(op, cl)
        # Pad INWARD from the wick (down from price), so zone_high == price holds.
        zone_low = min(zone_low_raw, price - price * _MIN_ZONE_PCT)
    else:
        price = lo
        zone_low = lo
        zone_high_raw = min(op, cl)
        zone_high = max(zone_high_raw, price + price * _MIN_ZONE_PCT)
    return SRLevelV2(
        price=price, tf_source=tf_source, pivot_type=pivot_type,
        confirmations=1, vrvp_confluence=False, formed_at=t, last_test=t,
        zone_low=zone_low, zone_high=zone_high,
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

    opens = df["Open"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)

    def _emit(k: int, kind: str) -> SRLevelV2:
        return _pivot(opens[k], highs[k], lows[k], closes[k], tf_source, kind, index[k])

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
                    pivots.append(_emit(run[-1], kind))
                run = [i]
        if run:
            pivots.append(_emit(run[-1], kind))
    return pivots


def _combine(cluster: list[SRLevelV2]) -> SRLevelV2:
    """Collapse one same-side price cluster per contract clause 6 (confirmations never
    decreases). Callers MUST pass a single-``pivot_type`` cluster — `_merge_by_price`
    enforces this by partitioning before clustering, so a resistance and a support pivot
    near the same price never collapse into one semantically invalid record.

    Zones use the outer envelope (min ``zone_low``, max ``zone_high``) with a
    max-width guard: if the envelope exceeds ``_MAX_ZONE_PCT`` of mid-price, fall back
    to the best contributor's zone (a 1w wide-wick + 1d tight pivot at the same cluster
    shouldn't produce a comically wide merged zone). Merged ``price`` is then pinned to
    the zone edge — resistance: ``zone_high``; support: ``zone_low`` — preserving the
    SFP-pierce invariant ``_significant_level`` reads. ``formed_at`` is the LATEST
    contributor's formation timestamp (not the dominant contributor's) — the merged
    record's zone shape only exists once the widening contributor has formed, so retest
    counting must not back-date through pre-widening bars. VRVP is deliberately reset to
    ``False`` here — confluence is owned by the post-merge ``_score_with_vrvp`` pass on
    the final price, not carried through.
    """
    best = max(cluster, key=lambda l: (_TF_RANK[l.tf_source], l.confirmations, l.last_test))
    lineages = {_lineage_key(l.tf_source) for l in cluster}
    zone_low_raw = min(c.zone_low for c in cluster)
    zone_high_raw = max(c.zone_high for c in cluster)
    mid = (zone_low_raw + zone_high_raw) / 2.0
    if mid > 0 and (zone_high_raw - zone_low_raw) / mid > _MAX_ZONE_PCT:
        zone_low, zone_high = best.zone_low, best.zone_high
    else:
        zone_low, zone_high = zone_low_raw, zone_high_raw
    price = zone_high if best.pivot_type == "high" else zone_low
    return SRLevelV2(
        price=price,
        tf_source=best.tf_source,
        pivot_type=best.pivot_type,
        confirmations=max(best.confirmations, len(lineages)),
        vrvp_confluence=False,
        formed_at=max(c.formed_at for c in cluster),
        last_test=max(l.last_test for l in cluster),
        zone_low=zone_low,
        zone_high=zone_high,
    )


def _merge_by_price(levels: list[SRLevelV2], tol: float) -> list[SRLevelV2]:
    """Group levels within ``tol`` of each other AND of the same ``pivot_type``, then
    combine. Splitting by side first guards against a resistance and a support pivot at
    near-identical price collapsing into one semantically invalid zone (codex correctness
    finding) — the two sides are tracked separately end-to-end."""
    if not levels:
        return []
    out: list[SRLevelV2] = []
    for ptype in ("high", "low"):
        same_side = [l for l in levels if l.pivot_type == ptype]
        if not same_side:
            continue
        ordered = sorted(same_side, key=lambda l: l.price)
        clusters: list[list[SRLevelV2]] = [[ordered[0]]]
        for lvl in ordered[1:]:
            if _within(lvl.price, clusters[-1][0].price, tol):
                clusters[-1].append(lvl)
            else:
                clusters.append([lvl])
        out.extend(_combine(c) for c in clusters)
    return out


def _find_htf_pivots(
    df_weekly: pd.DataFrame, df_monthly: pd.DataFrame, *, window: int, tol: float,
) -> list[SRLevelV2]:
    """Macro pivots: tie-tolerant W & M fractal pivots merged by price (clause 1)."""
    pivots = _fractal_pivots(df_weekly, "1w", window) + _fractal_pivots(df_monthly, "1mo", window)
    return _merge_by_price(pivots, tol)


def _refine_pivots_on_lower_tf(
    htf_pivots: list[SRLevelV2], df_daily: pd.DataFrame, df_3d: pd.DataFrame,
    *, window: int, n_sub: int, tol: float,
) -> list[SRLevelV2]:
    """Daily/3D sub-levels strictly inside macro brackets (clauses 2–5).

    Lower-TF pivots are merged first so ``1d``/``3d`` lineage siblings collapse
    (clause 4); each bracket keeps at most ``n_sub`` by confirmations then recency.
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
        inside.sort(key=lambda l: (l.confirmations, l.last_test), reverse=True)
        refined.extend(inside[:n_sub])
    return refined


def _count_retests(level: SRLevelV2, df: pd.DataFrame) -> int:
    """Post-formation same-side re-entries into ``[zone_low, zone_high]``.

    the framework: "recurring tests from the same direction generally make the level weaker…
    you don't want to trade that the eighth time round; the first/fresh test is the
    highest-probability trade." Only respect-side touches count — a level *flip* (price
    accepted on the wrong side via a close, not a single-bar wick) means the level is
    broken differently and we stop counting. The check fires regardless of state so a
    respect-side bar can flip directly to the wrong side without re-entering the zone.
    A missing/empty per-TF frame yields zero (degraded inputs never raise).
    """
    if df is None or df.empty:
        return 0
    respect_side = "below" if level.pivot_type == "high" else "above"
    after = df[df.index > level.formed_at]
    if after.empty:
        return 0
    retests = 0
    state = "in_zone"  # the pivot bar itself IS in zone
    for _, bar in after.iterrows():
        in_zone = (bar["Low"] <= level.zone_high) and (bar["High"] >= level.zone_low)
        # close-based acceptance: a single-bar wick beyond is not a flip
        flipped = (
            (respect_side == "below" and bar["Close"] > level.zone_high)
            or (respect_side == "above" and bar["Close"] < level.zone_low)
        )
        if flipped:
            break
        on_respect_side = (
            (respect_side == "below" and bar["High"] < level.zone_low)
            or (respect_side == "above" and bar["Low"] > level.zone_high)
        )
        if state == "in_zone":
            if on_respect_side:
                state = "respect_side"
        elif state == "respect_side" and in_zone:
            retests += 1
            state = "in_zone"
            if retests > _MAX_RETESTS:
                break  # one over the cap is enough to mark for drop
    return retests


def _apply_retest_decay(
    levels: list[SRLevelV2], frames_by_tf: dict[str, pd.DataFrame],
) -> list[SRLevelV2]:
    """Stamp ``retest_count`` per level and drop ``> _MAX_RETESTS``.

    Each level counts retests on its own ``tf_source`` frame — a ``1w`` pivot counts
    on weekly bars, a ``1d`` pivot on dailies — so the cadence of "same-direction
    retest" matches the TF the level was found on. Logs the dropped count at INFO
    when non-zero (operator visibility per plan).
    """
    out: list[SRLevelV2] = []
    dropped = 0
    for lvl in levels:
        rc = _count_retests(lvl, frames_by_tf.get(lvl.tf_source))
        if rc > _MAX_RETESTS:
            dropped += 1
            continue
        out.append(replace(lvl, retest_count=rc))
    if dropped > 0:
        logger.info(
            "derive_sr_levels: dropped %d level(s) with retest_count > %d",
            dropped, _MAX_RETESTS,
        )
    return out


def _score_with_vrvp(levels: list[SRLevelV2], vrvp: VolumeProfile) -> list[SRLevelV2]:
    """Flag ``vrvp_confluence`` where a level sits in the value area ``[VAL, VAH]``.

    Confluence only — never creates or moves a level. POC lies inside the value
    area, so value-area membership is the whole rule (no separate POC tolerance).
    Runs *after* merging on each final level's price (contract clause 6).
    """
    if vrvp is None or pd.isna(vrvp.val) or pd.isna(vrvp.vah):
        return [replace(l, vrvp_confluence=False) for l in levels]
    lo, hi = min(vrvp.val, vrvp.vah), max(vrvp.val, vrvp.vah)
    return [replace(l, vrvp_confluence=bool(lo <= l.price <= hi)) for l in levels]


def _merge_sr_levels(
    htf: list[SRLevelV2], refined: list[SRLevelV2], *, tol: float,
) -> list[SRLevelV2]:
    """Union macro + refined into one price-deduped set (clause 6)."""
    return _merge_by_price(list(htf) + list(refined), tol)


def _assign_quality(levels: list[SRLevelV2]) -> list[SRLevelV2]:
    """Set ``quality`` monotonic in rank → confirmations → vrvp → recency (clause 7),
    then decayed by ``_RETEST_DECAY ** retest_count`` (Sub-chunk F).

    Bounded bands keep the scalar lexicographic-consistent: rank (×1000) dominates
    confirmations (×100, ≤ a few hundred), which dominates the vrvp flag (5) and the
    normalized recency (≤1). Numeric weights are v1 defaults, calibrated Chunk 8b.
    The retest-decay multiplier is applied *after* the base — a level retested 3 times
    keeps its rank/confluence/vrvp ordering relative to its un-decayed self but is
    demoted vs a fresh peer (framework-canonical "fresh = highest probability").
    """
    if not levels:
        return []
    times = [pd.Timestamp(l.last_test).value for l in levels]
    lo, hi = min(times), max(times)
    span = hi - lo
    out = []
    for l in levels:
        recency = 1.0 if span == 0 else (pd.Timestamp(l.last_test).value - lo) / span
        base = (
            _TF_RANK[l.tf_source] * 1000.0
            + l.confirmations * 100.0
            + (5.0 if l.vrvp_confluence else 0.0)
            + recency
        )
        q = base * (_RETEST_DECAY ** l.retest_count)
        out.append(replace(l, quality=round(q, 4)))
    return out


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


def derive_sr_levels(
    weekly: pd.DataFrame | None = None,
    monthly: pd.DataFrame | None = None,
    daily: pd.DataFrame | None = None,
    three_day: pd.DataFrame | None = None,
    vrvp: VolumeProfile | None = None,
    *,
    window: int = _DEFAULT_WINDOW,
    tol: float = _DEFAULT_TOL,
    n_sub: int = _DEFAULT_N_SUB,
) -> list[SRLevelV2]:
    """Public seam: find → refine → merge → score → quality, sorted by price.

    Each TF frame is an explicit named parameter (canonical TFs ``weekly``/
    ``monthly``/``daily``/``three_day``) rather than a string-keyed dict — a missing
    frame is an honest ``None`` (no contribution from that TF), not a silently
    dropped misspelled key.
    """
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    w = weekly if weekly is not None else empty
    m = monthly if monthly is not None else empty
    d = daily if daily is not None else empty
    td = three_day if three_day is not None else empty
    macro = _find_htf_pivots(w, m, window=window, tol=tol)
    refined = _refine_pivots_on_lower_tf(macro, d, td, window=window, n_sub=n_sub, tol=tol)
    merged = _merge_sr_levels(macro, refined, tol=tol)
    scored = _score_with_vrvp(merged, vrvp) if vrvp is not None else merged
    # Sub-chunk F: post-formation retest count + drop > _MAX_RETESTS; each level
    # counts on its own tf_source frame.
    frames_by_tf = {"1mo": m, "1w": w, "3d": td, "1d": d}
    retested = _apply_retest_decay(scored, frames_by_tf)
    return sorted(_assign_quality(retested), key=lambda l: l.price)
