"""T-029.1 — momentum continuation as a DISCRETE setup->trigger trade (edge-existence test).

Sibling to the portfolio test (xs_momentum/ts_trend): those proved a *relative rank-and-rebalance*
factor; this asks whether a single trending coin entered on a defined trigger with a hard stop is +EV
PER TRADE — the only form the scanner can express. Plan: plans/momentum-discrete-trade-test.md
(Codex plan-review r1-r4 YELLOW -> r5 GREEN). Offline, read-only, $0.

The decisive control is BETA: crypto rose 2023-26, so naive long-breakout R is positive for free. The
gate statistic is therefore the SAME-DAY MATCHED difference d_d = mean net_R(trigger trades that day) -
mean net_R(in-uptrend non-triggered "control" trades that day), block-bootstrapped over time, on a
fixed-start cohort, across regime buckets. Arms (trigger/control/market/short) are simulated with
INDEPENDENT per-coin occupancy so trigger trades never starve the control sample.
"""
import importlib.util
import os
import sqlite3

import numpy as np
import pandas as pd

ps = importlib.util.spec_from_file_location("panel", os.path.join(os.path.dirname(__file__), "panel.py"))
panel = importlib.util.module_from_spec(ps)
ps.loader.exec_module(panel)

ARCHIVE = panel.ARCHIVE
DVOL_MIN = 5e6
SMA_N = 28
FAST_N = 10                  # fast MA for the pullback-and-resume trigger
ATR_N = 14
DONCH_N = 20
H = 20                       # time-stop (trading days)
STOP_MULT = 1.5              # primary risk = 1.5*ATR
TARGET_R = 2.0               # bracket fixed target (2R)
TRAIL_MULT = 3.0             # runner chandelier trail = extreme - 3*ATR
COSTS = {"fee 3.5": 3.5e-4, "repo 8.5": 8.5e-4, "stress 25": 25e-4}   # per side
PRIMARY_COST = "repo 8.5"
L_BLOCK = 20                 # bootstrap block length (trading days)
B = 3000
OUT = "/tmp/mom_discrete"


# ----- data (PIT daily OHLC incl high/low, built fresh; valid/dvol_med reused from panel) -----

def load_ohlc():
    _, c_pan, dvol_med, valid = panel.load()
    idx, cols = c_pan.index, list(c_pan.columns)
    con = sqlite3.connect(f"file:{ARCHIVE}?mode=ro", uri=True)
    o, h, l, c = {}, {}, {}, {}
    for coin in cols:
        df = pd.read_sql_query(
            "SELECT timestamp,open,high,low,close FROM candles WHERE coin=? AND interval='15m' "
            "ORDER BY timestamp", con, params=[coin])
        if df.empty:
            continue
        df["t"] = pd.to_datetime(df["timestamp"], utc=True)
        g = df.set_index("t").resample("1D")
        d = g.agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        o[coin], h[coin], l[coin], c[coin] = d["open"], d["high"], d["low"], d["close"]
    con.close()
    O = pd.DataFrame(o).reindex(idx)[cols]
    Hh = pd.DataFrame(h).reindex(idx)[cols]
    Ll = pd.DataFrame(l).reindex(idx)[cols]
    C = pd.DataFrame(c).reindex(idx)[cols]
    return O, Hh, Ll, C, dvol_med[cols], valid[cols], idx, cols


# ----- per-coin trade resolution (one arm, independent occupancy) -----

def resolve_arm(entry_mask, direction, O, H_, L_, C, valid, atr, dates, cols,
                stop_mult=STOP_MULT, exit_mode="bracket"):
    """direction: +1 long, -1 short. exit_mode: 'bracket' (-1R/+2R) or 'runner' (chandelier trail).
    Returns list of trade dicts."""
    Om, Hm, Lm, Cm = O.to_numpy(), H_.to_numpy(), L_.to_numpy(), C.to_numpy()
    Vm, Am, Em = valid.to_numpy(), atr.to_numpy(), entry_mask.to_numpy()
    n = len(dates)
    isoyw = [(t.isocalendar().year, t.isocalendar().week) for t in dates]
    rows = []
    for j, coin in enumerate(cols):
        free_d = 0
        for d in range(n):
            if not Em[d, j] or d < free_d:
                continue
            if d > n - H - 2:
                continue
            a = Am[d, j]
            if not np.isfinite(a) or a <= 0:
                continue
            if not Vm[d + 1:d + 2 + H, j].all():            # forward integrity: entry+hold all valid
                continue
            e = Om[d + 1, j]
            if not np.isfinite(e) or e <= 0:
                continue
            risk = stop_mult * a
            risk_frac = risk / e
            stop = e - direction * risk
            gross_r, exit_px, bars, outcome = None, None, None, None
            if exit_mode == "bracket":
                target = e + direction * TARGET_R * risk
                for t in range(d + 1, d + 2 + H):
                    hi, lo = Hm[t, j], Lm[t, j]
                    if direction > 0:
                        if lo <= stop:
                            gross_r, exit_px, bars, outcome = -1.0, stop, t - d, "stop"
                            break
                        if hi >= target:
                            gross_r, exit_px, bars, outcome = TARGET_R, target, t - d, "target"
                            break
                    else:
                        if hi >= stop:
                            gross_r, exit_px, bars, outcome = -1.0, stop, t - d, "stop"
                            break
                        if lo <= target:
                            gross_r, exit_px, bars, outcome = TARGET_R, target, t - d, "target"
                            break
            else:   # runner: initial stop, then a chandelier trail ratcheting from the running extreme
                run_stop, ext = stop, e
                for t in range(d + 1, d + 2 + H):
                    hi, lo = Hm[t, j], Lm[t, j]
                    if direction > 0:
                        if lo <= run_stop:                  # pessimistic: stop checked before this bar ratchets
                            gross_r, exit_px, bars, outcome = (run_stop - e) / risk, run_stop, t - d, "trail"
                            break
                        ext = max(ext, hi)
                        run_stop = max(run_stop, ext - TRAIL_MULT * a)
                    else:
                        if hi >= run_stop:
                            gross_r, exit_px, bars, outcome = (e - run_stop) / risk, run_stop, t - d, "trail"
                            break
                        ext = min(ext, lo)
                        run_stop = min(run_stop, ext + TRAIL_MULT * a)
            if gross_r is None:
                exit_px, bars, outcome = Cm[d + 1 + H, j], H, "time"
                gross_r = direction * (exit_px - e) / risk
            edi = d + 1                                      # entry-day panel index
            et = dates[edi]
            rows.append({
                "coin": coin, "dec_day": d, "entry_di": edi, "entry_date": et,
                "iso": isoyw[edi], "ybucket": ybucket(et), "arm": "?",
                "gross_r": gross_r, "risk_frac": risk_frac, "exit_over_e": exit_px / e,
                "bars": bars, "outcome": outcome,
            })
            free_d = d + bars                                # occupied entry..exit; next decision >= exit_t
    return rows


def ybucket(t):
    if t < pd.Timestamp("2024-04-01", tz="UTC"):
        return "Y1"
    if t < pd.Timestamp("2025-04-01", tz="UTC"):
        return "Y2"
    return "Y3"


def add_net(df):
    for name, cps in COSTS.items():
        df[f"net_{name}"] = df.gross_r - cps * (1.0 + df.exit_over_e) / df.risk_frac
    return df


# ----- inference: circular moving-block bootstrap over day-blocks -----

def _block_idx(n_days, rng):
    nb = int(np.ceil(n_days / L_BLOCK))
    starts = rng.integers(0, n_days, size=(B, nb))
    off = np.arange(L_BLOCK)
    idx = (starts[:, :, None] + off[None, None, :]) % n_days     # (B, nb, L)
    return idx.reshape(B, -1)[:, :n_days]                        # (B, n_days)


def boot_lower_trade_weighted(per_day_sum, per_day_cnt, rng, q=2.5):
    """Item 1: resample day-blocks, score each replicate on the trade-weighted grand mean."""
    n = len(per_day_sum)
    idx = _block_idx(n, rng)
    s = per_day_sum[idx].sum(axis=1)
    c = per_day_cnt[idx].sum(axis=1)
    means = s / np.where(c == 0, np.nan, c)
    return np.nanpercentile(means, q), np.nanmean(means)


def boot_lower_day_weighted(per_day_val, rng, q=2.5):
    """Item 2: resample day-blocks, score each replicate on the day-weighted mean (NaN = unmatched day)."""
    n = len(per_day_val)
    idx = _block_idx(n, rng)
    drawn = per_day_val[idx]                                     # (B, n) with NaNs
    means = np.nanmean(drawn, axis=1)
    return np.nanpercentile(means, q), np.nanmean(means)


def per_day_arrays(trig, ctrl, d_lo, d_hi):
    """Build day-indexed arrays over [d_lo, d_hi]: trigger sum/cnt, and matched daily effect delta_d."""
    days = np.arange(d_lo, d_hi + 1)
    pos = {d: i for i, d in enumerate(days)}
    n = len(days)
    t_sum, t_cnt = np.zeros(n), np.zeros(n)
    c_sum, c_cnt = np.zeros(n), np.zeros(n)
    netcol = f"net_{PRIMARY_COST}"
    for di, r in zip(trig.entry_di, trig[netcol]):
        t_sum[pos[di]] += r
        t_cnt[pos[di]] += 1
    for di, r in zip(ctrl.entry_di, ctrl[netcol]):
        c_sum[pos[di]] += r
        c_cnt[pos[di]] += 1
    matched = (t_cnt > 0) & (c_cnt > 0)
    delta = np.full(n, np.nan)
    delta[matched] = t_sum[matched] / t_cnt[matched] - c_sum[matched] / c_cnt[matched]
    return t_sum, t_cnt, delta, int(matched.sum())


# ----- report -----

def arm_stat(df, cost=PRIMARY_COST):
    net = df[f"net_{cost}"]
    return len(df), df.coin.nunique(), net.mean(), (df.gross_r > 0).mean(), df.bars.mean()


def r_dist(label, df):
    """Full R-distribution — so the tail is visible, not just the mean (the point of the runner exit)."""
    g = df.gross_r.to_numpy()
    net = df[f"net_{PRIMARY_COST}"].to_numpy()
    pct = np.percentile(g, [10, 25, 50, 75, 90, 95, 99])
    print(f"  {label:9} n={len(df):>5}  SumR(net) {net.sum():>+8.0f}  meanR {net.mean():>+.3f}  "
          f"win% {(g > 0).mean():>3.0%} | R p10..p99 " +
          "/".join(f"{p:+.1f}" for p in pct) + f"  max {g.max():+.1f}")


def gate_report(trig, ctrl, valid, C, dates, cols, rng, label, market=None):
    """Run the predeclared 5-item gate for one trigger config (its matched control supplies the beta null).
    If `market` is given, also report whether the entry beats a random-long baseline on the same exit."""
    netcol = f"net_{PRIMARY_COST}"
    d_lo = int(min(trig.entry_di.min(), ctrl.entry_di.min()))
    d_hi = int(max(trig.entry_di.max(), ctrl.entry_di.max()))
    t_sum, t_cnt, delta, n_matched = per_day_arrays(trig, ctrl, d_lo, d_hi)

    print(f"\n=== GATE [{label}] (primary long, net @ {PRIMARY_COST}) ===")
    # item 1 — per-trade EV (trade-weighted grand mean over day-blocks)
    lo1, pt1 = boot_lower_trade_weighted(t_sum, t_cnt, rng)
    n1, nc1 = len(trig), trig.coin.nunique()
    p1 = bool(lo1 > 0) and n1 >= 150 and nc1 >= 20
    print(f"  [1] edge exists   : meanR {pt1:+.3f}  block-boot 95% lo {lo1:+.3f}  "
          f"n={n1} coins={nc1}  -> {'PASS' if p1 else 'FAIL'}")
    # item 2 — beats beta (same-day matched effect, day-weighted)
    lo2, pt2 = boot_lower_day_weighted(delta, rng)
    if n_matched < 60:
        p2, verdict2 = None, f"INCONCLUSIVE (matched days {n_matched} < 60)"
    else:
        p2 = bool(lo2 > 0)
        verdict2 = "PASS" if p2 else "FAIL"
    print(f"  [2] beats beta    : mean(d_d) {pt2:+.3f}  block-boot 95% lo {lo2:+.3f}  "
          f"matched_days={n_matched}  -> {verdict2}")
    if market is not None:                              # beats a random-long baseline on the same exit?
        tm, mm = trig[netcol].mean(), market[netcol].mean()
        print(f"  [base] beats long : trig {tm:+.3f} vs random-long {mm:+.3f}  Δ {tm - mm:+.3f}  "
              f"-> {'PASS' if tm > mm else 'FAIL'}")
    # item 3 — regime-robust
    by = trig.groupby("ybucket")[netcol].mean()
    yrs_pos = sum(by.get(y, float("nan")) > 0 for y in ("Y1", "Y2", "Y3"))
    btc_bear = (C["BTC"] < C["BTC"].rolling(200).mean())
    trig_bear = trig[trig.dec_day.map(lambda d: bool(btc_bear.iloc[d]))]
    if len(trig_bear) < 40:
        bear_ok, bear_str = None, f"inconclusive (n={len(trig_bear)}<40)"
    else:
        bear_ok = bool(trig_bear[netcol].mean() > 0)
        bear_str = f"{trig_bear[netcol].mean():+.3f} (n={len(trig_bear)}) {'PASS' if bear_ok else 'FAIL'}"
    p3 = bool(yrs_pos >= 2) and (bear_ok is None or bear_ok)
    print("  [3] regime-robust : buckets " +
          " ".join(f"{y}:{by.get(y, float('nan')):+.3f}" for y in ("Y1", "Y2", "Y3")) +
          f" ({yrs_pos}/3>0) | BTC-bear {bear_str}  -> {'PASS' if p3 else 'FAIL'}")
    # item 4 — survivorship (fixed-start cohort)
    cohort = set(np.array(cols)[valid.iloc[0].to_numpy()])
    trig_co = trig[trig.coin.isin(cohort)]
    p4 = bool(trig_co[netcol].mean() > 0)
    print(f"  [4] survivorship  : fixed-start cohort meanR {trig_co[netcol].mean():+.3f} "
          f"(n={len(trig_co)}, {len(cohort)} coins)  -> {'PASS' if p4 else 'FAIL'}")
    # item 5 — cost headroom
    be = trig.gross_r.mean() / ((1.0 + trig.exit_over_e) / trig.risk_frac).mean() * 1e4
    p5 = bool(be >= 8.5)
    print(f"  [5] cost headroom : break-even {be:.1f} bps/side (>= 8.5?)  -> {'PASS' if p5 else 'FAIL'}")

    marginal = (abs(pt1) < 0.05) or (n_matched >= 60 and abs(pt2) < 0.05)
    decided = all(x is True for x in [p1, p2, p3, p4, p5])
    print(f"\n  GATE [{label}]: {'ALL PASS -> discrete setup has a basis' if decided else 'NOT ALL PASS'}"
          f"{'  [MARGINAL: ±0.05R -> 15m re-resolution advised]' if marginal else ''}")

    # descriptive cross-checks (non-gating)
    wk = {}
    for i, v in enumerate(delta):
        if np.isnan(v):
            continue
        ic = dates[d_lo + i].isocalendar()
        wk.setdefault((ic.year, ic.week), []).append(v)
    wvals = np.array([np.mean(v) for v in wk.values()])
    if len(wvals) >= 5:
        bs = rng.choice(wvals, size=(B, len(wvals)), replace=True).mean(axis=1)
        print(f"    desc: weekly d_w iid-boot mean {wvals.mean():+.3f} 95% lo {np.percentile(bs, 2.5):+.3f} "
              f"(n_weeks={len(wvals)})  |  cost bands " +
              " ".join(f"{k}:{trig[f'net_{k}'].mean():+.3f}" for k in COSTS))
    return decided


def main():
    O, H_, L_, C, dvol_med, valid, dates, cols = load_ohlc()
    sma = C.rolling(SMA_N).mean()
    sma_fast = C.rolling(FAST_N).mean()
    prev_c = C.shift()
    tr = pd.DataFrame(
        np.maximum.reduce([(H_ - L_).to_numpy(),
                           (H_ - prev_c).abs().to_numpy(),
                           (L_ - prev_c).abs().to_numpy()]),
        index=C.index, columns=C.columns)
    atr = tr.ewm(alpha=1.0 / ATR_N, adjust=False, min_periods=ATR_N).mean()
    donch_hi = H_.rolling(DONCH_N).max().shift(1)
    donch_lo = L_.rolling(DONCH_N).min().shift(1)
    form_valid = valid.astype(int).rolling(SMA_N).min().fillna(0).astype(bool)
    liquid = dvol_med >= DVOL_MIN
    in_up = C > sma
    in_dn = C < sma
    brk_up = C > donch_hi
    brk_dn = C < donch_lo
    # pullback-and-resume: in an uptrend, RECLAIM the fast MA from below —
    #   close[d-1] <= SMA_fast[d-1]  (yesterday at/below the 10-MA = the dip)
    #   close[d]   >  SMA_fast[d]    (today closed back above = the resume)
    # every term reads bars <= d; entry is open[d+1] -> no look-ahead. The mechanical OPPOSITE
    # of the breakout (buy weakness reclaiming, not strength extending).
    reclaim = (C.shift(1) <= sma_fast.shift(1)) & (C > sma_fast)
    base = form_valid & liquid

    masks = {
        "bo_trig": base & in_up & brk_up,
        "bo_ctrl": base & in_up & ~brk_up,
        "pb_trig": base & in_up & reclaim,
        "pb_ctrl": base & in_up & ~reclaim,
        "market":  base,
        "short":   base & in_dn & brk_dn,
    }
    longs = {"bo_trig", "bo_ctrl", "pb_trig", "pb_ctrl", "market"}

    def build(mode):
        out = {}
        for name, m in masks.items():
            if mode == "runner" and name == "short":
                continue                                       # short is a bracket-only diagnostic
            rows = resolve_arm(m, 1 if name in longs else -1, O, H_, L_, C, valid, atr, dates, cols,
                               exit_mode=mode)
            out[name] = add_net(pd.DataFrame(rows).assign(arm=name))
        return out

    bracket, runner = build("bracket"), build("runner")
    rng = np.random.default_rng(20260605)
    labels = {"bo_trig": "breakout", "bo_ctrl": "bo-ctrl", "pb_trig": "pullback",
              "pb_ctrl": "pb-ctrl", "market": "market", "short": "short"}
    print(f"=== T-029.1 discrete momentum trade | {dates.min().date()}..{dates.max().date()} | "
          f"{len(cols)} coins ===")

    print("\n########## BRACKET EXIT (-1R / +2R, the original test) ##########")
    print(f"{'arm':10}{'n':>7}{'coins':>7}{'meanR@repo':>12}{'win%':>7}{'avgbars':>9}")
    for name in ("bo_trig", "bo_ctrl", "pb_trig", "pb_ctrl", "market", "short"):
        n, nc, mr, wr, mb = arm_stat(bracket[name])
        print(f"  {labels[name]:8}{n:>7}{nc:>7}{mr:>+12.3f}{wr:>7.0%}{mb:>9.1f}")
    gate_report(bracket["bo_trig"], bracket["bo_ctrl"], valid, C, dates, cols, rng, "breakout/2R")
    gate_report(bracket["pb_trig"], bracket["pb_ctrl"], valid, C, dates, cols, rng, "pullback/2R")

    print(f"\n########## RUNNER EXIT (chandelier: extreme - {TRAIL_MULT:g}*ATR, no fixed cap) ##########")
    for name in ("bo_trig", "bo_ctrl", "pb_trig", "pb_ctrl", "market"):
        r_dist(labels[name], runner[name])
    gate_report(runner["bo_trig"], runner["bo_ctrl"], valid, C, dates, cols, rng, "breakout/runner",
                market=runner["market"])
    gate_report(runner["pb_trig"], runner["pb_ctrl"], valid, C, dates, cols, rng, "pullback/runner",
                market=runner["market"])

    os.makedirs(OUT, exist_ok=True)
    for name, df in bracket.items():
        df.to_parquet(f"{OUT}/{name}.parquet")
    for name, df in runner.items():
        df.to_parquet(f"{OUT}/{name}_runner.parquet")
    print(f"\nsaved per-trade tables -> {OUT}/")


if __name__ == "__main__":
    main()
