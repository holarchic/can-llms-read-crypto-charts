"""Step 3 — time-series (per-coin) trend leg of the momentum edge test.

The TS analog of XS, and the construction the eventual scanner SETUP maps to: for each
PIT-eligible coin, is it trending (close > 4wk SMA)? Hold the equal-weight basket of all
currently-trending coins; weekly rebalance; enter next-open, exit next-open; net of costs.
The decisive read is the SPREAD = trending-basket fwd return minus non-trending-basket fwd
return — the continuation signal isolated from market beta (does "this coin is trending"
predict higher forward return than "this coin is not"). Read-only.

Per plan: a TS single-config in the first cut so an XS fail can't kill continuation alone.
"""
import importlib.util
import os

import numpy as np
import pandas as pd

ps = importlib.util.spec_from_file_location("panel", os.path.join(os.path.dirname(__file__), "panel.py"))
panel = importlib.util.module_from_spec(ps); ps.loader.exec_module(panel)
xs_s = importlib.util.spec_from_file_location("xs", os.path.join(os.path.dirname(__file__), "xs_momentum.py"))
xs = importlib.util.module_from_spec(xs_s); xs_s.loader.exec_module(xs)

LOOKBACK, HOLD, DVOL_MIN = 28, 7, 5e6
COSTS = xs.COSTS
PER_YEAR = xs.PER_YEAR


def backtest():
    o, c, dvm, v = panel.load()
    O, C, DV, V = o.to_numpy(), c.to_numpy(), dvm.to_numpy(), v.to_numpy()
    cols = np.array(c.columns); dates = c.index; n = len(dates)
    rebs = range(LOOKBACK, n - HOLD - 1, HOLD)
    recs = []
    w_prev = np.zeros(len(cols))
    for d in rebs:
        win_valid = V[d - LOOKBACK:d + 1].all(axis=0)
        liquid = np.nan_to_num(DV[d]) >= DVOL_MIN
        tradable = ~np.isnan(O[d + 1]) & ~np.isnan(O[d + 1 + HOLD])
        elig = win_valid & liquid & tradable
        if elig.sum() < 10:
            continue
        idx = np.where(elig)[0]
        sma = np.nanmean(C[d - LOOKBACK + 1:d + 1, idx], axis=0)
        trending = C[d, idx] > sma                         # price above 4wk average
        fwd = O[d + 1 + HOLD, idx] / O[d + 1, idx] - 1.0
        up, dn = idx[trending], idx[~trending]
        if len(up) == 0 or len(dn) == 0:
            continue
        w = np.zeros(len(cols)); w[up] = 1.0 / len(up)
        recs.append({
            "date": dates[d], "year": dates[d].year, "n_up": len(up), "n_elig": len(idx),
            "trend_g": fwd[trending].mean(), "nontrend_g": fwd[~trending].mean(),
            "all": fwd.mean(), "spread": fwd[trending].mean() - fwd[~trending].mean(),
            "turn": np.abs(w - w_prev).sum(),
        })
        w_prev = w
    return pd.DataFrame(recs)


def main():
    df = backtest()
    print(f"TS trend (close>4wk SMA) | {len(df)} rebalances | mean {df.n_up.mean():.0f}/{df.n_elig.mean():.0f} "
          f"trending\n")
    print(f"{'strategy / cost':22}{'net ann':>9}{'Sharpe':>8}{'maxDD':>8}{'turnover':>10}")
    for cname, cps in COSTS.items():
        net = df.trend_g - df.turn * cps
        a, s, dd = xs.stats(net)
        print(f"  {'trend-basket '+cname:20}{a:+8.1%}{s:>8.2f}{dd:>8.1%}{df.turn.mean():>9.2f}x")
    for lab, col in [("non-trend (gross)", "nontrend_g"), ("hold-all (gross)", "all")]:
        a, s, dd = xs.stats(df[col]); print(f"  {lab:20}{a:+8.1%}{s:>8.2f}{dd:>8.1%}")

    print("\n=== SPREAD: trending minus non-trending (the continuation signal, beta-free) ===")
    a, s, dd = xs.stats(df.spread)
    print(f"  spread gross: {a:+.1%}/yr  Sharpe {s:.2f}  | mean/reb {df.spread.mean()*100:+.2f}%  "
          f">0 in {(df.spread>0).mean():.0%} of weeks")
    print("  per-year spread:", end="")
    for yr, g in df.groupby("year"):
        if len(g) >= 20:
            print(f"  {yr}:{g.spread.mean()*100:+.2f}%/wk", end="")
    print()
    print("\n  read: if the spread is reliably >0, 'this coin is trending' carries continuation")
    print("  info — the SETUP has a basis. If ~0, the trend filter adds nothing over hold-all.")
    df.to_parquet("/tmp/mom_ts.parquet")


if __name__ == "__main__":
    main()
