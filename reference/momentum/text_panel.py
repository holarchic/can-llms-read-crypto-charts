"""Stage 1 .2/.3 — Santiment social-volume data layer + the real text-lead readout (T-030).

Fetches raw `social_volume_total` (the PIT-robust raw count, NOT a revised score) per panel coin from the
Santiment free tier, builds a FAIL-CLOSED ticker->slug map (crypto tickers collide), aligns the series to the
cached price panel, caches the result, and runs the `text_lead` engine on the fixed-start covered cohort.

Free-tier reality (Stage-0 probe 2026-06): history is a trailing ~365 days (boundary ~2025-06-06) minus a
~30-day holdback, and the price panel ends 2026-03 — so the overlap is ~2025-06..2026-03, ~40 weekly
rebalances, single-regime. That is powered only for a STRONG lead; a weak effect reads INCONCLUSIVE by design.
Fetch once -> cache -> re-analyse free.
"""
import importlib.util
import os
import re
import time

import numpy as np
import pandas as pd
import san
from dotenv import load_dotenv

HERE = os.path.dirname(__file__)
load_dotenv(os.path.join(HERE, "..", "..", ".env"))
san.ApiConfig.api_key = os.getenv("SANTIMENT_API_KEY")

CACHE = "/tmp/text_panel"
METRIC = "social_volume_total"
FETCH_FROM = "2025-06-06"          # free-tier history boundary (Stage-0 probe)
FETCH_TO = "2026-06-06"            # san clamps the recent end at the ~30-day holdback
THROTTLE_S = 2.0                   # each metric query costs several credits vs the 100/min free cap
RETRY_429_S = 25                   # back-off on a per-minute rate-limit hit


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


panel = _load("panel")
tl = _load("text_lead")


def _slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")


def build_slug_map(tickers, proj):
    """ticker -> Santiment slug, FAIL-CLOSED on ambiguity. A ticker with >1 candidate slug maps only if exactly
    one candidate's slug equals slugify(its project name) (the canonical project); else it is excluded as
    ambiguous and logged — never silently pick one. Returns (mapped, ambiguous, unmapped)."""
    by_ticker = {}
    for _, row in proj.iterrows():
        by_ticker.setdefault(str(row["ticker"]), []).append((row["slug"], row["name"]))
    mapped, ambiguous, unmapped = {}, {}, []
    for t in tickers:
        cands = by_ticker.get(t, [])
        if not cands:
            unmapped.append(t)
        elif len(cands) == 1:
            mapped[t] = cands[0][0]
        else:
            canon = [s for s, n in cands if s == _slugify(n)]
            if len(canon) == 1:
                mapped[t] = canon[0]
            else:
                ambiguous[t] = [s for s, _ in cands]
    return mapped, ambiguous, unmapped


def _get_one(slug, tries=3):
    """One metric pull, retrying a 429 (per-minute rate limit) with a back-off so the fetch self-heals."""
    for attempt in range(tries):
        try:
            return san.get(f"{METRIC}/{slug}", from_date=FETCH_FROM, to_date=FETCH_TO, interval="1d")
        except Exception as e:
            if "429" in str(e) and attempt < tries - 1:
                time.sleep(RETRY_429_S)
                continue
            raise


def fetch_sv(slug_map):
    """One `social_volume_total` pull per slug over the free-tier window. Returns {ticker: value Series}. A
    failed/empty pull is dropped + logged (selective missingness by accident is the risk — so it is visible)."""
    out, failed = {}, []
    items = list(slug_map.items())
    for i, (t, slug) in enumerate(items):
        try:
            df = _get_one(slug)
            if df is not None and len(df):
                out[t] = df["value"]
            else:
                failed.append(t)
        except Exception as e:
            failed.append(t)
            print(f"  fetch {t}/{slug}: ERR {str(e)[:70]}")
        time.sleep(THROTTLE_S)
        if (i + 1) % 40 == 0:
            print(f"  ...fetched {i + 1}/{len(items)}")
    if failed:
        print(f"  {len(failed)} slugs returned no data: {', '.join(failed[:20])}{'...' if len(failed) > 20 else ''}")
    return out


def align(sv_by_ticker, price_close):
    """Reindex each coin's daily SV onto the price panel's date index (dates x coins). A real 0-mention day is a
    valid 0 row; a date with no Santiment row stays NaN (0 != not-covered)."""
    idx = price_close.index
    data = {}
    for t, s in sv_by_ticker.items():
        if t not in price_close.columns:
            continue
        s = s.copy()
        s.index = pd.to_datetime(s.index, utc=True).normalize()
        data[t] = s.reindex(idx)
    return pd.DataFrame(data, index=idx).reindex(columns=price_close.columns)


def fixed_start_cohort(sv_arr):
    """The text-side survivorship control: coins whose Santiment coverage is live by the overlap start (a full
    SV_BASE_WIN window from the first date with >=MIN_ELIG covered coins). Excludes coins that only appear once
    they go viral (endogenous coverage-onset bias)."""
    covered = np.isfinite(sv_arr)
    per_day = covered.sum(axis=1)
    starts = np.where(per_day >= tl.MIN_ELIG)[0]
    if len(starts) == 0:
        return np.zeros(sv_arr.shape[1], bool), None
    s = starts[0]
    cohort = covered[s:s + tl.SV_BASE_WIN].all(axis=0)
    return cohort, s


def main():
    o, c, dvm, v = panel.load()
    proj = san.get("projects/all")
    mapped, ambiguous, unmapped = build_slug_map(list(c.columns), proj)
    print(f"=== slug map: {len(mapped)} mapped · {len(ambiguous)} ambiguous (excluded) · "
          f"{len(unmapped)} unmapped (of {len(c.columns)} panel coins) ===")
    if ambiguous:
        print("  ambiguous (fail-closed):", ", ".join(f"{t}->{v}" for t, v in list(ambiguous.items())[:8]))

    # Incremental fetch: top up only the mapped coins not yet in the cache (so a 429-truncated run resumes
    # without re-spending quota on the coins it already has).
    cache_pq = os.path.join(CACHE, "sv.parquet")
    sv = pd.read_parquet(cache_pq) if os.path.exists(cache_pq) else pd.DataFrame(index=c.index)
    have = {t for t in sv.columns if np.isfinite(sv[t].to_numpy(dtype=float)).any()}
    todo = {t: s for t, s in mapped.items() if t not in have}
    if todo:
        print(f"fetching {len(todo)} missing series (have {len(have)} cached; throttle ~{THROTTLE_S}s/call)...")
        fetched = align(fetch_sv(todo), c)
        sv = sv.combine_first(fetched) if len(sv.columns) else fetched
        os.makedirs(CACHE, exist_ok=True)
        sv.to_parquet(cache_pq)
        print(f"SV panel now {sv.shape}")
    else:
        print(f"SV panel complete from cache {sv.shape}")

    sv_full = sv.reindex(columns=c.columns)
    sv_arr = sv_full.to_numpy(dtype=float)
    cov = np.isfinite(sv_arr)
    rows_covered = np.where(cov.any(axis=1))[0]
    span = f"{c.index[rows_covered[0]].date()}..{c.index[rows_covered[-1]].date()}" if len(rows_covered) else "none"
    print(f"=== coverage: {int(cov.any(axis=0).sum())} coins with any SV; date span {span} ===")

    cohort, start_i = fixed_start_cohort(sv_arr)
    print(f"fixed-start cohort: {int(cohort.sum())} coins (coverage live by "
          f"{c.index[start_i].date() if start_i is not None else 'n/a'})")

    O, C, V, DV = o.to_numpy(), c.to_numpy(), v.to_numpy(), dvm.to_numpy()
    rng = np.random.default_rng(0)
    print("\n=== TEXT-LEAD readout (Stage 1, real Santiment social_volume) ===")
    res_primary = tl.run_gate(O, C, V, DV, sv_arr, cohort, rng, "primary (cohort)")
    res_expand = tl.run_gate(O, C, V, DV, sv_arr, np.ones(C.shape[1], bool), rng, "secondary (expand)")
    print(f"\nPRIMARY VERDICT: {res_primary['verdict']}  (secondary/expanding: {res_expand['verdict']})")
    return res_primary, res_expand


if __name__ == "__main__":
    main()
