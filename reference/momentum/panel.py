"""Step 1 — point-in-time daily panel for the momentum edge test.

Builds daily OHLC + $-volume matrices (dates x coins) from the 15m archive, with a
bar-integrity mask (a day counts only with near-full 15m coverage), and exposes the
per-rebalance POINT-IN-TIME eligible universe (coins with completed formation history
ending at t-1 AND median $-volume >= threshold as of t). No `>=540d` survivorship filter
— the universe expands as coins list. Read-only. Saves matrices to /tmp/mom_panel/.

Per plan `plans/momentum-edge-test.md` (Codex GREEN): PIT universe, bar-integrity, $-vol filter.
"""
import os
import sqlite3

import numpy as np
import pandas as pd

ARCHIVE = "./data/binance-backtest.db"
OUT = "/tmp/mom_panel"
FULL_DAY_BARS = 96          # 24h / 15m
MIN_BARS = 90               # a daily bar is "complete" with >= this many 15m bars
DVOL_WINDOW = 30            # rolling window (days) for the liquidity filter
DEFAULT_DVOL_MIN = 5e6      # $5M median daily volume eligibility


def build():
    con = sqlite3.connect(f"file:{ARCHIVE}?mode=ro", uri=True)
    coins = [r[0] for r in con.execute("SELECT DISTINCT coin FROM candles WHERE interval='15m' ORDER BY coin")]
    o, c, dv, nb = {}, {}, {}, {}
    for coin in coins:
        df = pd.read_sql_query(
            "SELECT timestamp,open,high,low,close,volume FROM candles "
            "WHERE coin=? AND interval='15m' ORDER BY timestamp", con, params=[coin])
        if df.empty:
            continue
        df["t"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("t")
        g = df.resample("1D")
        d = g.agg({"open": "first", "close": "last", "volume": "sum"})
        d["n"] = df["close"].resample("1D").size()
        d = d[d["n"] > 0]
        o[coin] = d["open"]; c[coin] = d["close"]; dv[coin] = d["close"] * d["volume"]; nb[coin] = d["n"]
    con.close()

    open_m = pd.DataFrame(o).sort_index()
    close_m = pd.DataFrame(c).sort_index()
    dvol_m = pd.DataFrame(dv).sort_index()
    nbar_m = pd.DataFrame(nb).reindex_like(close_m)
    valid_m = (nbar_m >= MIN_BARS) & close_m.notna()        # bar-integrity mask
    dvol_med = dvol_m.where(valid_m).rolling(DVOL_WINDOW, min_periods=10).median()

    os.makedirs(OUT, exist_ok=True)
    for name, m in [("open", open_m), ("close", close_m), ("dvol_med", dvol_med), ("valid", valid_m)]:
        m.to_parquet(f"{OUT}/{name}.parquet")
    return open_m, close_m, dvol_med, valid_m


def load():
    return tuple(pd.read_parquet(f"{OUT}/{n}.parquet") for n in ("open", "close", "dvol_med", "valid"))


def eligible(close_m, dvol_med, valid_m, t, lookback_days, dvol_min=DEFAULT_DVOL_MIN):
    """PIT-eligible coins at rebalance date t: completed formation history ending at the last
    bar BEFORE t, valid bars over the window, and median $-vol >= threshold as of that bar."""
    prior = close_m.index[close_m.index < t]
    if len(prior) < lookback_days + 1:
        return pd.Index([])
    last = prior[-1]                                        # last completed bar before t
    start = prior[-(lookback_days + 1)]                    # formation window start
    win = valid_m.loc[start:last]
    has_hist = win.all(axis=0) & valid_m.loc[last]         # full valid coverage over formation
    liquid = dvol_med.loc[last] >= dvol_min
    return close_m.columns[has_hist & liquid.reindex(close_m.columns).fillna(False)]


def main():
    open_m, close_m, dvol_med, valid_m = build()
    print(f"panel: {close_m.shape[1]} coins | {close_m.index.min().date()} .. {close_m.index.max().date()} "
          f"| {close_m.shape[0]} days")
    print(f"complete-bar days: {valid_m.values.sum()} / {valid_m.size} ({valid_m.values.mean():.0%})")
    print("\n=== PIT eligible universe size over time (4wk lookback, $5M $-vol) ===")
    rebs = pd.date_range(close_m.index.min(), close_m.index.max(), freq="W-MON", tz="UTC")
    for t in rebs[::13]:                                    # ~quarterly samples
        n = len(eligible(close_m, dvol_med, valid_m, t, 28))
        print(f"  {t.date()}: {n} eligible")
    print(f"\n  fixed-start cohort (coins valid on {close_m.index.min().date()}): "
          f"{int(valid_m.iloc[0].sum())}")
    print(f"saved -> {OUT}/")


if __name__ == "__main__":
    main()
