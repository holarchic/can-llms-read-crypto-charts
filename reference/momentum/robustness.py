"""Step 4 — the decisive controls: survivorship (fixed-start cohort) + parameter grid.

Per plan, judged on the LONG-SHORT leg (the trustworthy one), net @ repo cost (8.5bps/side):
- **Fixed-start cohort**: re-run the primary config on ONLY the coins live at 2023-04-01
  (no later listings can be added) — removes the new-listing-pump confound that inflates
  expanding-universe long momentum. If the edge holds here it is not a listing artifact.
- **Grid**: lookback {2,4,8wk} × k {10,20,30%} × freq {weekly,biweekly}. PASS gate item 4 =
  long-short net positive in >=75% of cells. The grid is robustness only; the primary cell
  is what's judged. Read-only.
"""
import importlib.util
import os

import numpy as np
import pandas as pd

ps = importlib.util.spec_from_file_location("panel", os.path.join(os.path.dirname(__file__), "panel.py"))
panel = importlib.util.module_from_spec(ps); ps.loader.exec_module(panel)
xs_s = importlib.util.spec_from_file_location("xs", os.path.join(os.path.dirname(__file__), "xs_momentum.py"))
xs = importlib.util.module_from_spec(xs_s); xs_s.loader.exec_module(xs)

REPO = xs.COSTS["repo 8.5"]


def stats(net, hold):
    """Annualize with the cell's ACTUAL hold (xs.stats hardcodes weekly = 52/yr)."""
    py = 365.25 / hold
    net = np.asarray(net, float)
    ann = (1 + np.nanmean(net)) ** py - 1
    sh = np.nanmean(net) / np.nanstd(net) * np.sqrt(py) if np.nanstd(net) > 0 else float("nan")
    eq = np.cumprod(1 + np.nan_to_num(net))
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return ann, sh, dd


def run(O, C, DV, V, cols, lookback, hold, topq, subset=None, dvol_min=5e6):
    n = len(C)
    recs = []
    w_prev = np.zeros(len(cols))
    for d in range(lookback, n - hold - 1, hold):
        win_valid = V[d - lookback:d + 1].all(axis=0)
        liquid = np.nan_to_num(DV[d]) >= dvol_min
        tradable = ~np.isnan(O[d + 1]) & ~np.isnan(O[d + 1 + hold])
        elig = win_valid & liquid & tradable
        if subset is not None:
            elig = elig & subset
        if elig.sum() < 8:
            continue
        idx = np.where(elig)[0]
        form = C[d, idx] / C[d - lookback, idx] - 1.0
        fwd = O[d + 1 + hold, idx] / O[d + 1, idx] - 1.0
        order = np.argsort(form)
        k = max(1, int(round(topq * len(idx))))
        top, bot = idx[order[-k:]], idx[order[:k]]
        w = np.zeros(len(cols)); w[top] = 1.0 / k; w[bot] = -1.0 / k
        ff = np.full(len(cols), np.nan); ff[idx] = fwd
        recs.append({"year": d, "ls_g": np.nansum(w * ff), "to": np.abs(w - w_prev).sum()})
        w_prev = w
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["net"] = df.ls_g - df.to * REPO
    return df


def main():
    o, c, dvm, v = panel.load()
    O, C, DV, V = o.to_numpy(), c.to_numpy(), dvm.to_numpy(), v.to_numpy()
    cols = np.array(c.columns); dates = c.index
    per_year_bars = {}
    for i, ts in enumerate(dates):
        per_year_bars.setdefault(ts.year, []).append(i)

    print("=== SURVIVORSHIP: fixed-start cohort (coins live 2023-04-01) vs full PIT universe ===")
    cohort = V[0]                                          # 64 coins valid at bar 0
    print(f"  cohort size: {int(cohort.sum())} coins")
    for lab, sub in [("full PIT universe", None), ("fixed-start cohort", cohort)]:
        df = run(O, C, DV, V, cols, 28, 7, 0.20, subset=sub)
        a, s, dd = stats(df.net, 7)
        df["yr"] = [dates[i].year for i in df.year]
        py = " ".join(f"{yr}:{stats(g.net, 7)[0]:+.0%}" for yr, g in df.groupby("yr") if len(g) >= 20)
        print(f"  {lab:20} LS net {a:+7.1%}/yr  Sharpe {s:.2f}  maxDD {dd:.0%}  | {py}")

    print("\n=== ROBUSTNESS GRID: long-short net (repo cost), gate item 4 = >=75% cells positive ===")
    print(f"  {'lookback':9}{'k':>5}{'freq':>9}{'net ann':>9}{'Sharpe':>8}")
    results = []
    for lb, lbl in [(14, "2wk"), (28, "4wk"), (56, "8wk")]:
        for tq in (0.10, 0.20, 0.30):
            for hd, hl in [(7, "weekly"), (14, "biweekly")]:
                df = run(O, C, DV, V, cols, lb, hd, tq)
                if df.empty:
                    continue
                a, s, _ = stats(df.net, hd)
                results.append((a, s))
                star = "  *primary" if (lb, tq, hd) == (28, 0.20, 7) else ""
                print(f"  {lbl:9}{int(tq*100):>4}%{hl:>9}{a:+8.1%}{s:>8.2f}{star}")
    anns = [a for a, _ in results]
    pos = sum(a > 0 for a in anns) / len(anns)
    print(f"\n  cells positive: {pos:.0%} ({sum(a>0 for a in anns)}/{len(anns)})  "
          f"[gate item 4 needs >=75%]")
    print(f"  net ann across grid: min {min(anns):+.0%}  median {np.median(anns):+.0%}  max {max(anns):+.0%}")


if __name__ == "__main__":
    main()
