"""v10 the framework v2 producer + selection (T-019 Chunk 7b.2b).

The `SETUP_VERSION=v2` half of stage 1: per coin, assemble the multi-TF
`MarketContextV2` (`data_plane_v2`), run the six `setups_v2` detectors over each deep
setup TF, and lift the resulting `SetupSignalV2`s into the version-neutral
`BoundarySignal` the stage-2→4 spine consumes (Option C — no lossy down-conversion to
the v1 type). Selection is deterministic (no shuffle / diversity penalty, unlike v1):
collapse the same setup across adjacent TFs, then pick one historically-fresh signal
per coin and order by the screener score (all four setup TFs incl. 15m — the 15m chart
is in the LLM bundle as of Chunk 8 / 8.2).

`pipeline.stage1_screen` dispatches here on `config.setup_version == "v2"`; v1 stays
byte-identical on its own path. The returned bundle is the per-coin closed-bar candle
dict so stage-2 enriched charts render from the SAME frames the detectors used (no
refetch divergence).

`stage1_score` (carried on `BoundarySignal.confidence` — the unified screener-rank
slot v1 fills with its heuristic confidence) ranks deterministically and monotonically
in confluence (the detectors own their own thresholds; see `setups_v2`).
"""

import math
import sqlite3
from typing import NamedTuple

import pandas as pd

from .config import ScannerConfig
from .data_plane_v2 import (
    MarketContextV2,
    build_market_context_v2,
    fetch_multi_tf,
    htf_ema_for_chart,
)
from .models import (
    SETUP_NAMES,
    BoundarySignal,
    Direction,
    SetupSignalV2,
    SpineContext,
    SpineLevel,
    SRLevelV2,
)
from .scan_queries import is_duplicate
from .setups_v2 import (
    detect_deviation_reclaim,
    detect_gap_fill,
    detect_rsi_divergence_range,
    detect_sfp_range,
    detect_trend_continuation,
    detect_trend_reclaim,
)
from .sr_levels import chartable_tfs, derive_chart_levels
from .trend_state import TrendDirection


class Stage1ArtifactsV2(NamedTuple):
    """Per-coin multi-TF closed-bar candle bundle + the macro `SRLevelV2` substrate built
    once at stage 1, handed from stage 1 to stage 2 on the v2 path. Stage-2 enriched charts
    render from the same frames the v2 detectors used (no refetch divergence); `chart_inputs`
    layers per-chart-TF LTF pivots onto `macro_levels_by_coin` rather than recomputing
    `derive_sr_levels` (so stage 2 never re-emits the public-seam INFO log per coin). v1
    keeps its untyped 4H candle_map dict — that path retires at the cutover."""

    candles_by_coin: dict[str, dict[str, pd.DataFrame]]
    macro_levels_by_coin: dict[str, list[SRLevelV2]]    # ctx2.sr_levels stamped per coin


# The one-TF-lower ladder for the `lower_tf_reclaim` derivation. 15m is the floor —
# no lower TF exists, so its reclaim is fail-closed False (a trend_reclaim can never
# emit off accidental truthiness at the floor).
_LOWER_TF: dict[str, str | None] = {"1d": "4h", "4h": "1h", "1h": "15m", "15m": None}

# TF precedence for dedup / order tiebreaks — a higher TF wins an equal-score tie.
_TF_RANK: dict[str, int] = {"1d": 3, "4h": 2, "1h": 1, "15m": 0}

# Confluence count at which the deterministic screener rank saturates (pre-8b).
_SCORE_SATURATION = 5

# The v2 chartable-TF set, HTF-anchor view: keys = the frames charted (ordered HTF→LTF;
# `chart_inputs` selects exactly these), values = each frame's next-HTF anchor EMA. The
# renderer skips a line whose HTF frame is too short to seed the 200EMA (e.g. 1w on a 1d
# chart), so an absent/all-NaN entry is graceful, not an error.
_CHART_HTF = {"1d": "1w", "4h": "1d", "1h": "4h", "15m": "1h"}

# Chartable-TF MEMBERSHIP has one owner: `sr_levels.chartable_tfs()` (the public seam over
# the per-chart refinement chains). `_CHART_HTF` is this layer's HTF-anchor VIEW of that set
# — a different value concern, so it can't collapse into the chain map, but its membership
# is validated against the owner at import so the two views can't drift (cf. the Chunk 8c/8d
# prompt-map raise-guard).
if set(_CHART_HTF) != chartable_tfs():
    raise RuntimeError(
        f"chartable-TF set drift: _CHART_HTF={sorted(_CHART_HTF)} != "
        f"chartable_tfs()={sorted(chartable_tfs())}"
    )


def _latest(series: pd.Series) -> float:
    return float(series.iloc[-1]) if len(series) else float("nan")


def _direction_of(td: TrendDirection) -> Direction | None:
    """LONG/SHORT for an UP/DOWN local stack; None when FLAT (matches setups_v2)."""
    if td is TrendDirection.UP:
        return Direction.LONG
    if td is TrendDirection.DOWN:
        return Direction.SHORT
    return None


def _round_sig(x: float, sig: int = 4) -> float:
    """Round to ``sig`` significant figures — a scale-free identity key for the
    cross-TF dedup (a fixed decimal place is wrong across BTC vs a sub-cent coin)."""
    if not x or math.isnan(x):
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def _stage1_score(sig: SetupSignalV2) -> float:
    """Deterministic screener rank in [0,1] — monotonic in confluence count, demoted
    on the weakness flag. Regime-fit and S/R proximity are already folded into the
    detector's confluence set (TREND_ALIGNED / AT_SR), so confluence count is the
    single non-redundant ranking signal. Weights are placeholders pending Chunk-8b."""
    base = min(sig.confluence_count, _SCORE_SATURATION) / _SCORE_SATURATION
    return round(base * 0.9, 4) if sig.weakness_flag else round(base, 4)


def _lower_tf_reclaim(setup_tf: str, ctx2: MarketContextV2) -> bool:
    """Did the one-TF-lower substrate reclaim its 200EMA in the setup's direction?

    Direction is read from the setup-TF local stack (the same read `detect_trend_reclaim`
    uses), so the bool matches the direction the detector will derive. 15m floor / a
    missing lower substrate / a NaN 200EMA all fail closed (False)."""
    sub = ctx2.substrates.get(setup_tf)
    if sub is None:
        return False
    direction = _direction_of(sub.trend_state.local_trend_direction)
    if direction is None:  # FLAT — trend_reclaim won't emit regardless
        return False
    lower = _LOWER_TF.get(setup_tf)
    if lower is None:  # 15m floor — fail closed
        return False
    lower_sub = ctx2.substrates.get(lower)
    lower_sc = ctx2.setup_contexts.get(lower)
    if lower_sub is None or lower_sc is None:
        return False
    ema200 = _latest(lower_sub.longer.ema_200)
    price = lower_sc.current_price
    if math.isnan(ema200) or math.isnan(price):
        return False
    return price > ema200 if direction is Direction.LONG else price < ema200


def _run_detectors(ctx2: MarketContextV2) -> list[tuple[str, SetupSignalV2]]:
    """Run the six detectors over every deep setup TF; return (setup_tf, signal) pairs.

    A TF appears in `substrates`/`setup_contexts` only when it cleared the substrate
    floor (Chunk 7a), so absent TFs are skipped. `lower_tf_reclaim` is derived per TF
    and passed only to `detect_trend_reclaim` (the sole detector that takes it)."""
    out: list[tuple[str, SetupSignalV2]] = []
    for tf, sub in ctx2.substrates.items():
        sc = ctx2.setup_contexts.get(tf)
        if sc is None:
            continue
        sr = ctx2.sr_levels
        ltr = _lower_tf_reclaim(tf, ctx2)
        found: list[SetupSignalV2] = []
        found += detect_trend_continuation(sc, sub, sr)
        found += detect_trend_reclaim(sc, sub, sr, lower_tf_reclaim=ltr)
        found += detect_deviation_reclaim(sc, sub, sr)
        found += detect_gap_fill(sc, sub, sr)
        found += detect_sfp_range(sc, sub, sr)
        found += detect_rsi_divergence_range(sc, sub, sr)
        out.extend((tf, s) for s in found)
    return out


def _describe(coin: str, sig: SetupSignalV2) -> str:
    """Human-readable log/persistence label (NEVER LLM-visible — blind contract)."""
    name = SETUP_NAMES.get(sig.setup_type, sig.setup_type.value)
    return f"{name} {sig.direction.value} @ {sig.basis_price:.6g} ({sig.confluence_count} confluence)"


def _to_boundary(
    coin: str, setup_tf: str, sig: SetupSignalV2, ctx2: MarketContextV2, leverage: int | None,
) -> BoundarySignal:
    """Lift a `SetupSignalV2` into the version-neutral `BoundarySignal` (the Option-C v2
    producer). `context` is the `SpineContext` projection of the setup-TF read; the
    v2-only execution/confluence fields carry through; `confidence` holds the screener
    score and `key_level` the basis level (the v1 field the spine ranks/persists on)."""
    sc = ctx2.setup_contexts[setup_tf]
    sub = ctx2.substrates[setup_tf]
    spine = SpineContext(
        structure=sc.structure,
        sr_levels=[SpineLevel.from_sr_level_v2(lvl) for lvl in ctx2.sr_levels],
        range_bounds=sc.range_bounds,
        atr=sc.atr,
        current_price=sc.current_price,
        fib_levels=None,        # v2 has no fib swing — the spine carries None
        vol_percentile=None,    # no v2 volume-percentile source — NULL, not a fake 0.0
        efficiency_ratio=sc.efficiency_ratio,
        regime=sub.regime.regime.value,
    )
    return BoundarySignal(
        coin=coin,
        setup_type=sig.setup_type,
        direction=sig.direction,
        confidence=_stage1_score(sig),
        key_level=sig.basis_price,
        context=spine,
        description=_describe(coin, sig),
        setup_tf=setup_tf,
        max_leverage=leverage,
        basis_price=sig.basis_price,
        entry=sig.entry,
        hard_stop=sig.hard_stop,
        soft_stop=sig.soft_stop,
        confluence=sig.confluence,
        weakness_flag=sig.weakness_flag,
    )


def _dedup_cross_tf(candidates: list[BoundarySignal]) -> list[BoundarySignal]:
    """Collapse the same setup emitted across adjacent TFs to one instance, keeping the
    highest score (tie → higher TF). Identity = (coin, setup_type, direction, rounded
    key_level) so the survivor's `setup_tf` is well-defined for the per-coin step."""
    best: dict[tuple, BoundarySignal] = {}
    for c in candidates:
        key = (c.coin, c.setup_type.value, c.direction.value, _round_sig(c.key_level))
        cur = best.get(key)
        if cur is None or (c.confidence, _TF_RANK[c.setup_tf]) > (cur.confidence, _TF_RANK[cur.setup_tf]):
            best[key] = c
    return list(best.values())


def _select_one_per_coin(
    candidates: list[BoundarySignal], config: ScannerConfig, conn: sqlite3.Connection, verbose: bool,
) -> list[BoundarySignal]:
    """One signal per coin: the highest-score candidate that is not a historical
    duplicate (the existing `is_duplicate` gate), skipping a duplicated top candidate
    and keeping the next eligible one — preserving v1's per-coin dedup behaviour."""
    by_coin: dict[str, list[BoundarySignal]] = {}
    for c in candidates:
        by_coin.setdefault(c.coin, []).append(c)

    survivors: list[BoundarySignal] = []
    for coin, cs in by_coin.items():
        cs.sort(key=lambda b: (-b.confidence, -_TF_RANK[b.setup_tf],
                               b.setup_type.value, b.direction.value, _round_sig(b.key_level)))
        try:
            for c in cs:
                if is_duplicate(coin, c.setup_type.value, c.direction.value,
                                config.dedup_hours, conn, exchange=config.exchange):
                    if verbose:
                        print(f"  {coin} {c.setup_type.value} — skipped (duplicate)")
                    continue
                survivors.append(c)
                break
        except Exception as e:  # a per-coin dedup failure skips that coin, never the scan (matches v1)
            if verbose:
                print(f"  {coin} — selection error: {e}")
    return survivors


def screen_v2(
    coins: list[str],
    leverage_map: dict[str, int | None],
    disabled_types: set[str] | None,
    config: ScannerConfig,
    conn: sqlite3.Connection,
    verbose: bool = True,
) -> tuple[list[BoundarySignal], Stage1ArtifactsV2]:
    """Produce + select the v2 screener signals, deterministically ordered by score.

    Returns the ordered (uncapped) `BoundarySignal` list plus a `Stage1ArtifactsV2`
    bundle mapping each surviving coin to its closed-bar candle dict (for the stage-2
    enriched charts). The caller (`stage1_screen`) caps `sent_to_llm` and persists the
    full list — mirroring v1's `save_signals(all, ..., len(capped))`.
    """
    disabled = {t for t in (disabled_types or set())}

    bundles: dict[str, dict[str, pd.DataFrame]] = {}
    macro_by_coin: dict[str, list[SRLevelV2]] = {}
    candidates: list[BoundarySignal] = []
    errors = 0
    for coin in coins:
        # The whole per-coin production (fetch → build → detectors → lift) is one guarded
        # unit, committed atomically: any failure logs + skips THIS coin and never aborts
        # the scan or half-emits (matches v1 `_screen_v1`'s per-coin isolation). The catch
        # is broad by design — a ~200-coin cron scan over a flaky exchange must survive any
        # one coin's failure, and an exchange/pandas exception allowlist would inevitably
        # miss a type and abort the whole scan (a worse failure than skipping one coin).
        # Silent masking is the real risk, so a TOTAL wipeout is surfaced loudly below.
        try:
            candles = fetch_multi_tf(coin, config)
            ctx2 = build_market_context_v2(coin, candles)
            leverage = leverage_map.get(coin)
            coin_candidates = [
                _to_boundary(coin, setup_tf, sig, ctx2, leverage)
                for setup_tf, sig in _run_detectors(ctx2)
                if sig.setup_type.value not in disabled  # disabled_types vocab is case-normalized
            ]
        except Exception as e:
            errors += 1
            if verbose:
                print(f"  {coin} — error: {e}")
            continue
        bundles[coin] = candles
        macro_by_coin[coin] = ctx2.sr_levels    # the macro substrate, built once here
        candidates.extend(coin_candidates)

    # A few per-coin failures are ordinary scan noise; EVERY coin failing means the v2
    # producer is systematically broken (a bug / bad deploy / wrong data shape), not a
    # quiet "no setups today" — alert distinctly so it can't hide as scan noise.
    if errors and errors == len(coins):
        from .alerts import send_errors_alert
        send_errors_alert(f"v2 screener: all {errors} coins failed in stage 1 — producer likely broken", config)

    # (2) cross-TF dedup → (3) one historically-fresh signal per coin. 15m-origin
    # setups flow through — the 15m chart is in the LLM bundle as of Chunk 8 (8.2).
    survivors = _select_one_per_coin(_dedup_cross_tf(candidates), config, conn, verbose)
    # (4) global deterministic order — score desc, higher TF, then stable name/dir/level
    # keys so the cap is reproducible regardless of upstream insertion order.
    survivors.sort(key=lambda b: (-b.confidence, -_TF_RANK[b.setup_tf],
                                  b.coin, b.setup_type.value, b.direction.value, _round_sig(b.key_level)))

    # Prune the bundle to coins that survived — the only frames stage 2 will chart.
    survivor_coins = {b.coin for b in survivors}
    bundle = {coin: bundles[coin] for coin in survivor_coins}
    macro_bundle = {coin: macro_by_coin[coin] for coin in survivor_coins}
    return survivors, Stage1ArtifactsV2(candles_by_coin=bundle, macro_levels_by_coin=macro_bundle)


class ChartInputs(NamedTuple):
    """The producer-owned bundle the v2 dispatch hands to the chart renderer — a named
    record, not a positional tuple, so widening it can't silently shift a caller's
    unpacking. `candle_sets` = chartable frames (HTF→LTF); `htf_emas` = each chart's
    next-HTF anchor EMA; `per_chart_levels` = each chart-TF's S/R `SpineLevel` list (the
    macro substrate + that TF's local pivots)."""

    candle_sets: dict[str, pd.DataFrame]
    htf_emas: dict[str, pd.Series]
    per_chart_levels: dict[str, list[SpineLevel]]


def chart_inputs(
    candles: dict[str, pd.DataFrame],
    macro_levels: list[SRLevelV2],
) -> ChartInputs:
    """The v2 enriched-chart inputs from a coin's closed-bar candle dict (a `screen_v2`
    bundle entry): the chartable frames (HTF→LTF) the detectors used, the next-HTF 200EMA
    aligned onto each chart's index, AND the per-chart-TF S/R level map — so enriched charts
    render from the SAME frames as detection and each TF carries its own locally-refined
    zones. `_CHART_HTF` is the HTF-anchor view of the chartable-TF set (its keys, ordered
    1d/4h/1h/15m; membership shared with `_CHART_TF_CHAIN`, drift-guarded at import) + each
    TF's HTF anchor (its values). `macro_levels` is the pre-computed stage-1 substrate
    (`ctx2.sr_levels`); `derive_chart_levels` layers chart-TF-local pivots onto it per chart
    (no `derive_sr_levels` recompute), and the `SRLevelV2 → SpineLevel` projection happens
    HERE so the renderer stays SRLevelV2-agnostic. `per_chart_levels` is keyed by EXACTLY
    the rendered TFs (== `candle_sets` keys) — the renderer indexes it per TF, no fallback.
    Producer-owned; pure; renderer skips a line whose HTF frame is too short to seed the
    200EMA."""
    candle_sets = {tf: candles[tf] for tf in _CHART_HTF if tf in candles}
    htf_emas: dict[str, pd.Series] = {}
    per_chart_levels: dict[str, list[SpineLevel]] = {}
    for tf, frame in candle_sets.items():
        htf = _CHART_HTF.get(tf)
        if htf and htf in candles:
            htf_emas[tf] = htf_ema_for_chart(candles[htf], frame.index)
        srlvs = derive_chart_levels(tf, candles, macro_levels=macro_levels)
        per_chart_levels[tf] = [SpineLevel.from_sr_level_v2(lvl) for lvl in srlvs]
    return ChartInputs(candle_sets, htf_emas, per_chart_levels)
