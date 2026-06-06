"""Stage 1 — text-attention analysis engine + anti-vacuity fixtures (T-030, plan Codex-GREEN).

Does an abnormal-attention text factor add MARGINAL IC over price-momentum (Test A), and does it genuinely
LEAD momentum rather than confirm the recent past (Test B)? Both are rank-based partial correlations, computed
per weekly rebalance on the same cadence/eligibility as `xs_momentum.py`, aggregated into a per-rebalance IC
series with a weekly circular moving-block bootstrap.

This module is the engine PLUS its three-fixture self-validation — it runs $0/offline on synthetic panels with
a KNOWN text->return relationship, so the inference is trusted before any real Santiment quota is spent:
  (i)  planted-lead   — T tracks the FORWARD return  -> both mIC and lead CIs must clear 0
  (ii) shuffled-null  — T carries no forward info     -> both CIs must cover 0
  (iii) momentum-clone — T := momentum + noise         -> positive STANDALONE IC, but marginal mIC and lead
                                                          both strip to ~0 after partialling momentum out
The real Santiment readout (chunk .3) feeds `run_gate` real (O, C, valid, dvol, SV) arrays in place of these.
"""
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd

# Cadence — mirror xs_momentum.py exactly, so "marginal IC over momentum" is exact not approximate.
LOOKBACK = 28          # 4wk formation (daily bars) — the momentum factor M = C[d]/C[d-28]-1
HOLD = 7               # weekly rebalance == forward horizon == the "recent past" window for the lead test
DVOL_MIN = 5e6         # $-vol liquidity floor (xs_momentum parity)
SV_BASE_WIN = 28       # abnormal-attention baseline: a coin's own trailing SV mean
MIN_ELIG = 10          # min eligible coins to score a rebalance

# Inference — weekly circular moving-block bootstrap. NOT discrete_trade's L_BLOCK=20: that is 20 *trading
# days* in a daily-trade bootstrap; here each observation is one weekly rebalance, so the block is in weeks.
L_BLOCK_WEEKS = 4
B_BOOT = 3000
RHO_REDUNDANT = 0.9    # |rho(T, M)| >= this => text is momentum in disguise, not an independent signal
MIN_DATES = 30         # fewer scored rebalances than this => underpowered cohort => INCONCLUSIVE

# Per-rebalance series schema — a fixed column set so a zero-date (thin) cohort still classifies INCONCLUSIVE
# instead of KeyError-ing on an empty frame.
_STAT_COLS = ["d", "n", "mic", "lead", "rho", "ic_T"]


class CI(NamedTuple):
    """A bootstrap CI on a mean. `lo`/`hi` are the 2.5/97.5 percentiles; the CI clears 0 iff `lo > 0`."""
    lo: float
    hi: float
    mean: float


def _rank(x):
    """Cross-sectional average-rank (Spearman basis). x is the finite eligible slice at one date."""
    return pd.Series(x).rank().to_numpy()


def _resid(y, x):
    """Residual of y on [1, x] by OLS — both 1-D, equal length, finite."""
    a = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    return y - a @ coef


def _partial_corr(a, b, c):
    """Partial correlation of a, b controlling for c — corr of each one's residual after regressing on c."""
    ra, rb = _resid(a, c), _resid(b, c)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _text_factor(sv, d):
    """Abnormal attention at d: log1p(SV[d]) minus log1p(the coin's own trailing SV_BASE_WIN-day mean).
    Size-decoupled — the raw level just proxies market cap; the surprise vs the coin's baseline is the pulse.
    log1p so a real 0-mention day is a valid low value (0 != not-covered, which is NaN, filtered upstream)."""
    with warnings.catch_warnings():
        # An all-NaN trailing window (an uncovered coin) -> NaN baseline; those coins fail eligibility and
        # never reach the gate, so the "Mean of empty slice" warning here is benign.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        base = np.nanmean(sv[d - SV_BASE_WIN:d], axis=0)
    return np.log1p(sv[d]) - np.log1p(base)


def _eligible(o, v, dv, sv, d, cohort):
    """xs_momentum eligibility (full 28d valid formation + liquid + forward-tradable) AND text-covered AND in
    the fixed-start cohort. `panel.eligible()` alone is insufficient — it omits the forward-tradability check."""
    win_valid = v[d - LOOKBACK:d + 1].all(axis=0)
    liquid = np.nan_to_num(dv[d]) >= DVOL_MIN
    tradable = ~np.isnan(o[d + 1]) & ~np.isnan(o[d + 1 + HOLD])
    covered = ~np.isnan(sv[d - SV_BASE_WIN:d]).any(axis=0) & ~np.isnan(sv[d])
    return win_valid & liquid & tradable & covered & cohort


def per_date_stats(o, c, v, dv, sv, cohort, t_mat=None):
    """One row per weekly rebalance: marginal IC, the lead differential, rho(T,M), and the standalone IC.
    All four factors (T, M, F_fwd, R_past7) share ONE per-date cross-sectional rank transform — so Test A and
    Test B stay comparable. `t_mat` (dates x coins) overrides the abnormal-attention factor with a supplied
    one — used only by the momentum-clone fixture to inject a post-transform `T := M + noise` and so isolate
    the partial-correlation control; production and the other fixtures leave it None (full SV->T pipeline)."""
    n = len(c)
    recs = []
    for d in range(LOOKBACK, n - HOLD - 1, HOLD):
        idx = np.where(_eligible(o, v, dv, sv, d, cohort))[0]
        if len(idx) < MIN_ELIG:
            continue
        m = c[d, idx] / c[d - LOOKBACK, idx] - 1.0              # momentum (trailing 28d)
        t = (t_mat[d] if t_mat is not None else _text_factor(sv, d))[idx]   # abnormal attention (or injected)
        f = o[d + 1 + HOLD, idx] / o[d + 1, idx] - 1.0          # next-open forward return
        rp = c[d, idx] / c[d - HOLD, idx] - 1.0                 # recent past (trailing 7d) text would echo
        rm, rt, rf, rrp = _rank(m), _rank(t), _rank(f), _rank(rp)
        pc_fwd = _partial_corr(rt, rf, rm)                      # text vs FUTURE | momentum
        pc_past = _partial_corr(rt, rrp, rm)                    # text vs recent PAST | momentum
        recs.append({
            "d": d, "n": len(idx),
            "mic": pc_fwd,                                      # Test A
            "lead": pc_fwd - pc_past,                           # Test B: leads (>0) vs merely confirms (<0)
            "rho": float(np.corrcoef(rt, rm)[0, 1]),           # redundancy of text with momentum
            "ic_T": float(np.corrcoef(rt, rf)[0, 1]),          # standalone text IC (clone-fixture readout)
        })
    return pd.DataFrame(recs, columns=_STAT_COLS)


def _block_idx(n, rng, length=L_BLOCK_WEEKS, b=B_BOOT):
    """Circular moving-block resample indices (b, n) over the per-rebalance series, blocked in weeks."""
    nb = int(np.ceil(n / length))
    starts = rng.integers(0, n, size=(b, nb))
    off = np.arange(length)
    idx = (starts[:, :, None] + off[None, None, :]) % n
    return idx.reshape(b, -1)[:, :n]


def boot_ci(vals, rng, qlo=2.5, qhi=97.5):
    """Two-sided weekly-block-bootstrap CI on the mean of a per-rebalance series. NaN dates are dropped, not
    averaged through (`v is not None` would let a NaN poison the whole mean). Empty series -> all-NaN CI, which
    a zero-date cohort carries straight to INCONCLUSIVE."""
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return CI(np.nan, np.nan, np.nan)
    means = v[_block_idx(len(v), rng)].mean(axis=1)
    return CI(float(np.percentile(means, qlo)), float(np.percentile(means, qhi)), float(v.mean()))


def classify(mic_ci, lead_ci, rho_med, n_dates):
    """Four powered outcomes + the underpowered escape. mic_ci/lead_ci are (lo, hi, mean); CI clears 0 iff
    lo > 0. A thin cohort is INCONCLUSIVE, never a hard FAIL."""
    if n_dates < MIN_DATES:
        return "INCONCLUSIVE"
    if not mic_ci.lo > 0:
        return "FAIL-REDUNDANT"                                # Test A fails: text adds nothing over price
    if abs(rho_med) >= RHO_REDUNDANT:
        return "FAIL-REDUNDANT"                                # text is momentum in disguise
    if not lead_ci.lo > 0:
        return "FAIL-NONLEADING"                               # adds marginal IC but does not LEAD at weekly
    return "PASS"


def run_gate(o, c, v, dv, sv, cohort, rng, label, t_mat=None):
    """Score one panel through both gate tests and print the readout. Returns the result dict."""
    df = per_date_stats(o, c, v, dv, sv, cohort, t_mat=t_mat)
    mic_ci = boot_ci(df["mic"], rng)
    lead_ci = boot_ci(df["lead"], rng)
    rho_med = float(np.nanmedian(df["rho"])) if len(df) else np.nan
    ic_med = float(np.nanmedian(df["ic_T"])) if len(df) else np.nan
    verdict = classify(mic_ci, lead_ci, rho_med, len(df))
    print(f"  {label:16} dates={len(df):>3}  "
          f"mIC {mic_ci.mean:+.3f} [{mic_ci.lo:+.3f},{mic_ci.hi:+.3f}]  "
          f"lead {lead_ci.mean:+.3f} [{lead_ci.lo:+.3f},{lead_ci.hi:+.3f}]  "
          f"rho {rho_med:+.2f}  IC_T {ic_med:+.3f}  -> {verdict}")
    return {"label": label, "n_dates": len(df), "mic": mic_ci, "lead": lead_ci,
            "rho_med": rho_med, "ic_med": ic_med, "verdict": verdict}


# ----- anti-vacuity fixtures: synthetic panels with a KNOWN text->return relationship -----

def synth_panel(rng, target, n_days=540, n_coins=140, mom=0.0, strength=0.6):
    """Build (O, C, valid, dvol, SV) where the abnormal-attention factor T tracks a known target.
    target 'lead'  -> T tracks the FORWARD return  (text should pass both tests)
    target 'none'  -> T is structureless noise     (the null base; caller shuffles, or the clone supplies its
                                                     own factor via t_mat — see main())
    `mom`>0 ties returns to a persistent latent AR(1) factor so trailing return M carries real forward IC
    (needed for the clone's 'positive standalone IC') — a STATIONARY momentum source, not price-level feedback
    (which compounds into runaway trends and a degenerate fat-tailed M). Open[d] = Close[d-1], so
    F[d] = C[d+HOLD]/C[d]-1 is the forward leg."""
    eps = rng.normal(0.0, 0.03, (n_days, n_coins))
    phi = np.zeros((n_days, n_coins))                         # persistent latent factor (AR1) = momentum source
    for d in range(1, n_days):
        phi[d] = 0.94 * phi[d - 1] + rng.normal(0.0, 1.0, n_coins)
    r = eps + mom * 0.02 * (phi / (phi.std() + 1e-9))         # mom>0 => trailing return predicts forward
    c = np.cumprod(1.0 + r, axis=0)
    o = np.vstack([c[0:1], c[:-1]])                           # open[d] = close[d-1]

    g = rng.normal(0.0, 1.0, (n_days, n_coins))               # the per-(d,coin) attention signal
    for d in range(LOOKBACK, n_days - HOLD - 1):
        if target == "lead":
            base = c[d + HOLD] / c[d] - 1.0                    # forward return
        else:
            continue
        z = (base - base.mean()) / (base.std() + 1e-9)
        g[d] = strength * z + np.sqrt(1.0 - strength ** 2) * rng.normal(0.0, 1.0, n_coins)

    base_size = np.exp(rng.normal(7.0, 1.5, n_coins))         # heterogeneous coin popularity (size)
    sv = np.round(base_size[None, :] * np.exp(0.5 * g))       # raw mention counts; T recovers ~0.5*g
    v = np.ones((n_days, n_coins), bool)
    dv = np.full((n_days, n_coins), 1e9)
    cohort = np.ones(n_coins, bool)
    return o, c, v, dv, sv, cohort


def main():
    rng = np.random.default_rng(0)
    print("=== Stage-1 text-lead engine — anti-vacuity fixtures (synthetic, $0) ===\n")

    res_lead = run_gate(*synth_panel(rng, "lead", mom=0.0), rng, "planted-lead")

    # momentum-clone: returns carry momentum (so M has real forward IC), and T is M + noise injected POST-
    # transform — the only faithful test that `_partial_corr` strips a momentum-collinear factor to ~0.
    o, c, v, dv, sv, cohort = synth_panel(rng, "none", mom=0.5)
    t_clone = np.full_like(c, np.nan)
    for d in range(LOOKBACK, len(c) - HOLD - 1):
        m = c[d] / c[d - LOOKBACK] - 1.0
        z = (m - m.mean()) / (m.std() + 1e-9)
        t_clone[d] = z + 0.5 * rng.normal(0.0, 1.0, c.shape[1])
    res_clone = run_gate(o, c, v, dv, sv, cohort, rng, "momentum-clone", t_mat=t_clone)

    # shuffled-null: a planted-lead panel whose SV is permuted across coins within each date — preserves the
    # per-date SV distribution, destroys the text<->return alignment (the textbook shuffle null).
    o, c, v, dv, sv, cohort = synth_panel(rng, "lead", mom=0.0)
    for d in range(sv.shape[0]):
        sv[d] = rng.permutation(sv[d])
    res_null = run_gate(o, c, v, dv, sv, cohort, rng, "shuffled-null")

    # thin-cohort: an all-ineligible panel scores 0 rebalances -> must classify INCONCLUSIVE, never crash.
    o, c, v, dv, sv, cohort = synth_panel(rng, "none", mom=0.0)
    v[:] = False
    res_thin = run_gate(o, c, v, dv, sv, cohort, rng, "thin-cohort")

    print("\n--- anti-vacuity contract (the engine must measure what it claims) ---")
    checks = [
        ("planted-lead  mIC clears 0",      res_lead["mic"].lo > 0),
        ("planted-lead  lead clears 0",     res_lead["lead"].lo > 0),
        ("planted-lead  verdict PASS",      res_lead["verdict"] == "PASS"),
        ("shuffled-null mIC covers 0",      res_null["mic"].lo <= 0 <= res_null["mic"].hi),
        ("shuffled-null lead covers 0",     res_null["lead"].lo <= 0 <= res_null["lead"].hi),
        ("momentum-clone T tracks M (rho>.5)", res_clone["rho_med"] > 0.5),
        ("momentum-clone standalone IC>0",  res_clone["ic_med"] > 0),
        ("momentum-clone marginal mIC~0",   res_clone["mic"].lo <= 0 <= res_clone["mic"].hi),
        ("momentum-clone lead~0",           res_clone["lead"].lo <= 0 <= res_clone["lead"].hi),
        ("thin-cohort -> INCONCLUSIVE",     res_thin["verdict"] == "INCONCLUSIVE"),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print(f"\n{'ALL FIXTURES PASS — engine validated' if ok else 'FIXTURE FAILURE — engine not trustworthy'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
