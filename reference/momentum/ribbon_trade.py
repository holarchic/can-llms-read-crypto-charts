"""T-029.2 — ribbon-pullback as a CONDITIONED discrete trade (edge-existence test).

Plan: plans/momentum-ribbon-pullback-test.md (Codex r1/r2 YELLOW -> r3 GREEN). Offline, read-only, $0.

The first three momentum tests fired naked single-bar triggers; this one is the operator's actual method —
a *conditioned* setup on the framework's ribbon (EMA 13/25/36): an aligned uptrend with HH/HL structure where the
ribbon has been *respected* >=2 times, BTC above its own ribbon, then a pullback to the band midpoint (EMA25)
with the stop under the ribbon (EMA36) and a structural runner exit (ride until close < EMA36 = trend lost).

The decisive question: does the CONDITIONING beat its own ordinary (unconditioned) ribbon-pullbacks on the
same days? The exit is no longer a confound (the runner was validated in T-029.1) — selectivity is the only
remaining lever. Imports the PRODUCTION ribbon (`scanner.ma_indicators.compute_ema_band`) for exact fidelity.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from scanner.ma_indicators import compute_ema_band  # noqa: E402  (production ribbon, pure fn)

_dt = importlib.util.spec_from_file_location("discrete_trade",
                                             os.path.join(os.path.dirname(__file__), "discrete_trade.py"))
dt = importlib.util.module_from_spec(_dt)
_dt.loader.exec_module(dt)

COSTS, PRIMARY_COST = dt.COSTS, dt.PRIMARY_COST
ybucket, add_net, gate_report, r_dist = dt.ybucket, dt.add_net, dt.gate_report, dt.r_dist

DVOL_MIN = 5e6
ATR_N = 14
PIV_K = 3                    # fractal pivot half-width (swing confirmed at i + PIV_K)
RESPECT_WIN = 60            # trailing window for the respect count
MIN_RESPECTS = 2
W = 10                      # bars to wait for the pullback fill
H = 40                     # time-stop (trading days)
STOP_ATR = 0.5             # stop = EMA36 - 0.5*ATR (under the ribbon)
FORM_WIN = 60              # formation/integrity window (covers EMA36 warmup + respect window)
OUT = "/tmp/mom_ribbon"


def ema_bands(C):
    f, m, s = {}, {}, {}
    for coin in C.columns:
        band = compute_ema_band(C[coin].astype(float), 13, 25, 36)
        f[coin], m[coin], s[coin] = band.fast, band.mid, band.slow
    cols = list(C.columns)
    return pd.DataFrame(f)[cols], pd.DataFrame(m)[cols], pd.DataFrame(s)[cols]


def atr14(O, H_, L_, C):
    prev = C.shift()
    tr = pd.DataFrame(np.maximum.reduce([(H_ - L_).to_numpy(),
                                         (H_ - prev).abs().to_numpy(),
                                         (L_ - prev).abs().to_numpy()]),
                      index=C.index, columns=C.columns)
    return tr.ewm(alpha=1.0 / ATR_N, adjust=False, min_periods=ATR_N).mean()


def confirmed_hhhl(H_, L_):
    """HH and HL both true as-of each bar, using ONLY swings confirmed by i+PIV_K <= d (no right-edge peek)."""
    Hn, Ln = H_.to_numpy(), L_.to_numpy()
    n, ncol = Hn.shape
    out = np.zeros((n, ncol), dtype=bool)
    for j in range(ncol):
        h, l = Hn[:, j], Ln[:, j]
        sh, sl = [], []                                    # (confirm_bar = i+PIV_K, extreme value)
        for i in range(PIV_K, n - PIV_K):
            wh, wl = h[i - PIV_K:i + PIV_K + 1], l[i - PIV_K:i + PIV_K + 1]
            if np.isfinite(h[i]) and h[i] == np.nanmax(wh) and (wh == h[i]).sum() == 1:
                sh.append((i + PIV_K, h[i]))
            if np.isfinite(l[i]) and l[i] == np.nanmin(wl) and (wl == l[i]).sum() == 1:
                sl.append((i + PIV_K, l[i]))
        si = sj = 0
        last_h, last_l = [], []
        for d in range(n):
            while si < len(sh) and sh[si][0] <= d:
                last_h.append(sh[si][1])
                si += 1
            while sj < len(sl) and sl[sj][0] <= d:
                last_l.append(sl[sj][1])
                sj += 1
            hh = len(last_h) >= 2 and last_h[-1] > last_h[-2]
            hl = len(last_l) >= 2 and last_l[-1] > last_l[-2]
            out[d, j] = hh and hl
    return out


def _exit(e, stop, risk, fillbar, Lm, Cm, Sm, j, n):
    """Structural runner: ride until close < EMA36; protective stop checked first (pessimistic); time-stop H."""
    end = fillbar + H
    for u in range(fillbar, end + 1):
        if Lm[u, j] <= stop:
            return (stop - e) / risk, stop, u - fillbar, "stop"
        if Cm[u, j] < Sm[u, j]:
            return (Cm[u, j] - e) / risk, Cm[u, j], u - fillbar, "trail"
    return (Cm[end, j] - e) / risk, Cm[end, j], H, "time"


def resolve_ribbon(arm, Om, Hm, Lm, Cm, Mm, Sm, Am, Vm, dates, cols):
    """Limit-at-EMA25 (gap-aware, pre-bar) entry over W bars, then the structural runner exit."""
    n = len(dates)
    iso = [(t.isocalendar().year, t.isocalendar().week) for t in dates]
    rows = []
    for j, coin in enumerate(cols):
        free = 0
        for d in range(n):
            if not arm[d, j] or d < free:
                continue
            free = d + W                                   # default: occupied through the wait window
            for t in range(d + 1, min(d + 1 + W, n)):
                lim, s_prev, a_prev = Mm[t - 1, j], Sm[t - 1, j], Am[t - 1, j]
                ot, lt = Om[t, j], Lm[t, j]
                if not np.isfinite(lim) or not np.isfinite(s_prev) or not np.isfinite(a_prev) or not np.isfinite(ot):
                    free = t
                    break
                stp = s_prev - STOP_ATR * a_prev
                if ot <= stp:                              # gapped through the ribbon -> expire
                    free = t
                    break
                if ot <= lim:
                    e = ot
                elif lt <= lim:
                    e = lim
                else:
                    continue
                risk = e - stp
                if risk <= 0 or e <= 0:
                    free = t
                    break
                if t + H > n - 1 or not Vm[t:t + H + 1, j].all():   # forward integrity / insufficient history
                    free = t
                    break
                gr, xp, bars, outc = _exit(e, stp, risk, t, Lm, Cm, Sm, j, n)
                rows.append({
                    "coin": coin, "dec_day": d, "entry_di": t, "entry_date": dates[t],
                    "iso": iso[t], "ybucket": ybucket(dates[t]), "gross_r": gr, "risk_frac": risk / e,
                    "exit_over_e": xp / e, "bars": bars, "outcome": outc,
                    "fillstop": outc == "stop" and bars == 0,
                })
                free = t + bars
                break
    return rows


def resolve_longall(elig, Om, Hm, Lm, Cm, Sm, Am, Vm, dates, cols):
    """Baseline: next-open entry on every eligible coin-day, same ribbon stop + structural runner exit."""
    n = len(dates)
    iso = [(t.isocalendar().year, t.isocalendar().week) for t in dates]
    rows = []
    for j, coin in enumerate(cols):
        free = 0
        for d in range(n):
            if not elig[d, j] or d < free:
                continue
            if d + 1 > n - 1:
                continue
            e = Om[d + 1, j]
            s_d, a_d = Sm[d, j], Am[d, j]
            if not np.isfinite(e) or not np.isfinite(s_d) or not np.isfinite(a_d) or e <= 0:
                continue
            stop = s_d - STOP_ATR * a_d
            risk = e - stop
            if risk <= 0:
                continue                                   # price already below the ribbon -> skip
            fb = d + 1
            if fb + H > n - 1 or not Vm[fb:fb + H + 1, j].all():
                continue
            gr, xp, bars, outc = _exit(e, stop, risk, fb, Lm, Cm, Sm, j, n)
            rows.append({
                "coin": coin, "dec_day": d, "entry_di": fb, "entry_date": dates[fb],
                "iso": iso[fb], "ybucket": ybucket(dates[fb]), "gross_r": gr, "risk_frac": risk / e,
                "exit_over_e": xp / e, "bars": bars, "outcome": outc, "fillstop": False,
            })
            free = fb + bars
    return rows


def main():
    O, H_, L_, C, dvol_med, valid, dates, cols = dt.load_ohlc()
    f, m, s = ema_bands(C)
    atr = atr14(O, H_, L_, C)
    aligned = (f > m) & (m > s)
    hhhl = confirmed_hhhl(H_, L_)
    respect_event = aligned & (L_ <= f) & (C > s)
    respects2 = (respect_event.astype(int).shift(1).rolling(RESPECT_WIN).sum() >= MIN_RESPECTS)
    btc_up = (C["BTC"] > s["BTC"]).to_numpy()
    form_valid = valid.astype(int).rolling(FORM_WIN).min().fillna(0).astype(bool)
    liquid = dvol_med >= DVOL_MIN

    armed = (aligned & (C > f) & form_valid & liquid).to_numpy()
    cond = hhhl & respects2.to_numpy() & btc_up[:, None]
    conditioned = armed & cond
    control = armed & ~cond
    elig = (form_valid & liquid).to_numpy()

    Om, Hm, Lm, Cm = O.to_numpy(), H_.to_numpy(), L_.to_numpy(), C.to_numpy()
    Mm, Sm, Am, Vm = m.to_numpy(), s.to_numpy(), atr.to_numpy(), valid.to_numpy()

    arms = {
        "ribbon":  add_net(pd.DataFrame(resolve_ribbon(conditioned, Om, Hm, Lm, Cm, Mm, Sm, Am, Vm, dates, cols))),
        "control": add_net(pd.DataFrame(resolve_ribbon(control, Om, Hm, Lm, Cm, Mm, Sm, Am, Vm, dates, cols))),
        "longall": add_net(pd.DataFrame(resolve_longall(elig, Om, Hm, Lm, Cm, Sm, Am, Vm, dates, cols))),
    }
    rng = np.random.default_rng(20260605)

    print(f"=== T-029.2 ribbon-pullback (conditioned) | {dates.min().date()}..{dates.max().date()} | "
          f"{len(cols)} coins | EMA 13/25/36, stop<EMA36, structural runner ===")
    print(f"  armed coin-days: {int(armed.sum())} | conditioned: {int(conditioned.sum())} "
          f"({conditioned.sum() / max(armed.sum(), 1):.0%} of armed) | control: {int(control.sum())}")
    for nm in ("ribbon", "control", "longall"):
        r_dist(nm, arms[nm])
    for nm in ("ribbon", "control"):
        fs = arms[nm].fillstop.mean() if len(arms[nm]) else float("nan")
        print(f"    {nm} same-day fill+stop share: {fs:.0%}")

    gate_report(arms["ribbon"], arms["control"], valid, C, dates, cols, rng, "ribbon-conditioned",
                market=arms["longall"])

    os.makedirs(OUT, exist_ok=True)
    for nm, df in arms.items():
        df.to_parquet(f"{OUT}/{nm}.parquet")
    print("\nGATE (operator's bar) = item[1] SigmaR>0  AND  item[2] beats matched control  AND  [base] beats long-all")
    print(f"saved -> {OUT}/")


if __name__ == "__main__":
    main()
