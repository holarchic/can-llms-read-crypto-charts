"""Step 2 — cross-sectional momentum backtest (primary config), PIT + net of costs.

Primary config (predeclared in the plan): 4wk lookback, weekly (7-bar) rebalance, top-20%,
evaluated long-only AND market-neutral long-short, over the PIT expanding universe.

Look-ahead guard (Codex HIGH-3): rank on close[d] (the decision bar), transact entirely in
bars d+1..d+1+HOLD — enter at open[d+1], exit at open[d+1+HOLD], open-to-open. No same-close
fill. Costs charged on turnover (sum|Δw|) under three per-side bands. Benchmarks (hold-all,
BTC) use the SAME PIT-eligible universe / timing. Read-only.
"""
import importlib.util
import os

import numpy as np
import pandas as pd

ps = importlib.util.spec_from_file_location("panel", os.path.join(os.path.dirname(__file__), "panel.py"))
panel = importlib.util.module_from_spec(ps); ps.loader.exec_module(panel)

LOOKBACK = 28          # 4 weeks (daily bars)
HOLD = 7               # weekly rebalance
TOPQ = 0.20            # top/bottom 20%
DVOL_MIN = 5e6
COSTS = {"fee 3.5": 3.5e-4, "repo 8.5": 8.5e-4, "stress 25": 25e-4}   # per side (fraction)
PER_YEAR = 365.25 / HOLD


def backtest():
    o, c, dvm, v = panel.load()
    O, C, DV, V = o.to_numpy(), c.to_numpy(), dvm.to_numpy(), v.to_numpy()
    cols = np.array(c.columns)
    dates = c.index
    n = len(dates)
    rebs = list(range(LOOKBACK, n - HOLD - 1, HOLD))

    recs = []
    w_long_prev = np.zeros(len(cols))
    w_ls_prev = np.zeros(len(cols))
    for d in rebs:
        win_valid = V[d - LOOKBACK:d + 1].all(axis=0)
        liquid = np.nan_to_num(DV[d]) >= DVOL_MIN
        tradable = ~np.isnan(O[d + 1]) & ~np.isnan(O[d + 1 + HOLD])
        elig = win_valid & liquid & tradable
        if elig.sum() < 10:
            continue
        idx = np.where(elig)[0]
        form = C[d, idx] / C[d - LOOKBACK, idx] - 1.0
        fwd = O[d + 1 + HOLD, idx] / O[d + 1, idx] - 1.0
        order = np.argsort(form)                       # ascending
        k = max(1, int(round(TOPQ * len(idx))))
        top = idx[order[-k:]]                           # highest momentum
        bot = idx[order[:k]]                            # lowest

        w_long = np.zeros(len(cols)); w_long[top] = 1.0 / k
        w_ls = np.zeros(len(cols)); w_ls[top] = 1.0 / k; w_ls[bot] = -1.0 / k
        fwd_full = np.full(len(cols), np.nan); fwd_full[idx] = fwd
        long_g = np.nansum(w_long * fwd_full)
        ls_g = np.nansum(w_ls * fwd_full)              # spread return (long - short)
        allret = np.nanmean(fwd)
        btc_i = np.where(cols == "BTC")[0]
        btc = (O[d + 1 + HOLD, btc_i[0]] / O[d + 1, btc_i[0]] - 1.0) if len(btc_i) and not np.isnan(O[d + 1, btc_i[0]]) else np.nan
        recs.append({
            "date": dates[d], "year": dates[d].year, "n_elig": len(idx),
            "long_g": long_g, "ls_g": ls_g, "all": allret, "btc": btc,
            "to_long": np.abs(w_long - w_long_prev).sum(), "to_ls": np.abs(w_ls - w_ls_prev).sum(),
            "contrib_long": w_long * fwd_full, "contrib_ls": w_ls * fwd_full,
        })
        w_long_prev, w_ls_prev = w_long, w_ls
    return pd.DataFrame(recs), cols


def stats(net):
    net = np.asarray(net, float)
    ann = (1 + np.nanmean(net)) ** PER_YEAR - 1
    sharpe = np.nanmean(net) / np.nanstd(net) * np.sqrt(PER_YEAR) if np.nanstd(net) > 0 else float("nan")
    eq = np.cumprod(1 + np.nan_to_num(net))
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return ann, sharpe, dd


def main():
    df, cols = backtest()
    print(f"XS momentum | {len(df)} weekly rebalances | {df.date.min().date()}..{df.date.max().date()} "
          f"| mean eligible {df.n_elig.mean():.0f}\n")

    print(f"{'strategy / cost':22}{'net ann':>9}{'Sharpe':>8}{'maxDD':>8}{'turnover':>10}")
    for tag, gcol, tcol in [("LO", "long_g", "to_long"), ("LS", "ls_g", "to_ls")]:
        for cname, cps in COSTS.items():
            net = df[gcol] - df[tcol] * cps
            ann, sh, dd = stats(net)
            print(f"  {tag+' '+cname:20}{ann:+8.1%}{sh:>8.2f}{dd:>8.1%}{df[tcol].mean():>9.2f}x")
    print()
    for lab, col in [("hold-all (gross)", "all"), ("BTC-hold (gross)", "btc")]:
        ann, sh, dd = stats(df[col]); print(f"  {lab:20}{ann:+8.1%}{sh:>8.2f}{dd:>8.1%}")

    print("\n=== per-year NET @ repo cost (8.5bps) ===")
    for col, tcol, lab in [("long_g", "to_long", "long-only"), ("ls_g", "to_ls", "long-short")]:
        net = df[col] - df[tcol] * COSTS["repo 8.5"]
        line = f"  {lab:11}"
        for yr, g in df.assign(net=net).groupby("year"):
            if len(g) >= 20:
                a, _, _ = stats(g["net"]); line += f"  {yr}:{a:+6.0%}"
        print(line)

    print("\n=== break-even cost (bps/side) ===")
    lo_be = (df.long_g.mean() - df["all"].mean()) / df.to_long.mean() * 1e4
    ls_be = df.ls_g.mean() / df.to_ls.mean() * 1e4
    print(f"  long-only vs hold-all: {lo_be:.1f} bps   |   long-short vs 0: {ls_be:.1f} bps")

    print("\n=== return concentration ===")
    for lab, ccol in [("long-only", "contrib_long"), ("long-short", "contrib_ls")]:
        contrib = np.nansum(np.vstack(df[ccol].values), axis=0)
        s = pd.Series(contrib, index=cols)
        tot = s.sum()
        sg = s.sort_values(ascending=False)
        print(f"  {lab}: top-5 coins = {sg.head(5).sum()/tot:+.0%} of P&L: {', '.join(sg.head(5).index)}")
    best = (df.long_g - df.to_long * COSTS['repo 8.5']).sort_values(ascending=False)
    print(f"  long-only top-5 weeks = {best.head(5).sum()/best.sum():.0%} of net P&L  "
          f"(>100% => most weeks lose)")

    print("\n--- PRIMARY-CELL gate read (items 1-3; grid is step 4) ---")
    net_lo = df.long_g - df.to_long * COSTS["repo 8.5"]; net_ls = df.ls_g - df.to_ls * COSTS["repo 8.5"]
    a_lo, s_lo, _ = stats(net_lo); a_ls, s_ls, _ = stats(net_ls); a_ha, s_ha, _ = stats(df["all"])
    print(f"  item1 long-short: Sharpe {s_ls:.2f} (>=0.5?) ann {a_ls:+.1%} (>=+5%?)")
    print(f"  item2 long-only Sharpe {s_lo:.2f} vs hold-all {s_ha:.2f} (>= +0.3?)")
    df.to_parquet("/tmp/mom_xs.parquet")


if __name__ == "__main__":
    main()
