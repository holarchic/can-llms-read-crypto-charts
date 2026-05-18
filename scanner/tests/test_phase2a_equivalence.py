"""T-016 Phase 2a equivalence battery (SPEC #22).

LESSONS-CODING "Green ≠ verified": a relocation + latent-correctness change
is only verified by driving the NEW storage-owned read model and a FROZEN
verbatim copy of the pre-2a `analytics.get_closed_trades` / `compute_live_metrics`
over real data, with an anti-vacuity oracle that is non-vacuous *by
construction* (the prod corpus provably does NOT diverge — bare `LIMIT 1`
== first-by-`created_at` for all 72/72 prod multi-SL scans — so a
"new != old on prod" tripwire would never fire and prove nothing).

Two partitions, two orthogonal oracles:

(a) **Faithful-relocation** over the *prod* corpus — new == frozen-old
    byte-identical on every `TradeRecord` field and every `MetricsWindow`
    field. Non-vacuous because any relocation perturbation breaks it. A
    rowid-order guard first asserts the in-memory load reproduces prod's
    bare-`LIMIT 1` selection (recorded `limit1_sl`), so this really is a
    prod-equivalent oracle and extraction order can't silently erase the
    equivalence.

(b) **Correctness + anti-vacuity by construction** over a hand-crafted
    *adversarial* scan whose `stop_loss` rows are inserted so rowid /
    insertion order ≠ `created_at` order (mirrors the real latent failure:
    SL-replacement rows + a VACUUM/reindex reordering rowids). Frozen-old's
    bare `LIMIT 1` provably selects the non-original SL (asserted directly,
    competing SL rows kept at the same status so order is the only
    discriminator); new's `original_stop_price` selects the original;
    `realized_r` diverges and new == an independently hand-derived value.
    This tripwire fires by construction, so a regression that re-introduces
    undefined-order SL selection fails closed here.
"""

import dataclasses
import json as _json
import sqlite3
from datetime import datetime
from pathlib import Path

from scanner.execution.analytics import compute_live_metrics
from scanner.execution.storage import (
    EXIT_ORDER_TYPES_SQL,
    ensure_orders_table,
    get_closed_trades,
    original_stop_price,
)
from scanner.models import TradeRecord

_CORPUS = _json.loads(
    (Path(__file__).parent / "fixtures" / "closed_trades_corpus.json").read_text()
)


# ---------------------------------------------------------------------------
# FROZEN copy of the pre-2a read path (analytics.py @ fcb3620): bare `LIMIT 1`
# (no ORDER BY) SL lookup + dict-returning compute_live_metrics + the `if days:`
# close-window branch. Field-output faithful — every value-bearing path is
# verbatim; the only deviations are non-value-bearing: the missing-exit-price
# `log.warning(...)` side-effect is omitted (does not affect any TradeRecord
# field) and records are built from the relocated `scanner.models.TradeRecord`
# (same shape). This is the immutable baseline — do NOT "fix" it.
# ---------------------------------------------------------------------------

def _frozen_exit_price(exit_order: dict) -> float | None:
    for field in ("fill_price", "trigger_price"):
        val = exit_order.get(field)
        if val is not None:
            return float(val)
    return None


def _frozen_slippage_r(entry, exit_order, entry_fill, exit_fill, risk_amount, conn) -> float:
    if risk_amount <= 0:
        return 0
    scan_row = conn.execute(
        "SELECT machine_block FROM scans WHERE scan_id = ?", (entry["scan_id"],),
    ).fetchone()
    if not scan_row or not scan_row["machine_block"]:
        return 0
    try:
        mb = _json.loads(scan_row["machine_block"])
    except (ValueError, TypeError):
        return 0
    expected_entry = (mb.get("entry_low", 0) + mb.get("entry_high", 0)) / 2
    if expected_entry <= 0:
        return 0
    entry_slip = abs(entry_fill - expected_entry) * (entry.get("quantity", 0) or 0)
    return entry_slip / risk_amount if risk_amount > 0 else 0


def _frozen_get_closed_trades(conn, execution_mode=None, days=None) -> list[TradeRecord]:
    clauses = ["e.order_type = 'entry'", "e.status = 'FILLED'"]
    params = []
    if execution_mode:
        clauses.append("e.execution_mode = ?")
        params.append(execution_mode)
    if days:  # verbatim pre-2a close-window branch (fcb3620)
        clauses.append(f"""EXISTS (
            SELECT 1 FROM (
                SELECT scan_id, execution_mode, exchange,
                       MAX(datetime(COALESCE(filled_at, created_at))) AS final_exit_at
                FROM orders
                WHERE order_type IN {EXIT_ORDER_TYPES_SQL} AND status = 'FILLED'
                GROUP BY scan_id, execution_mode, exchange
            ) x
            WHERE x.scan_id = e.scan_id
              AND x.execution_mode = e.execution_mode
              AND x.exchange = e.exchange
              AND datetime(x.final_exit_at) > datetime('now', '-{days} days')
        )""")
    where = " AND ".join(clauses)
    entries = conn.execute(
        f"SELECT e.* FROM orders e WHERE {where} ORDER BY e.created_at", params,
    ).fetchall()
    trades = []
    for entry in entries:
        entry = dict(entry)
        exits = conn.execute(
            f"""SELECT * FROM orders WHERE scan_id = ? AND order_type IN {EXIT_ORDER_TYPES_SQL}
            AND status = 'FILLED' AND execution_mode = ? AND exchange = ?
            ORDER BY datetime(COALESCE(filled_at, created_at))""",
            (entry["scan_id"], entry["execution_mode"], entry["exchange"]),
        ).fetchall()
        if not exits:
            continue
        exits = [dict(x) for x in exits]
        entry_fill = entry.get("fill_price") or entry.get("entry_price") or 0
        entry_qty = entry.get("quantity", 0)
        filled_exit_qty = sum(x.get("quantity", 0) for x in exits)
        if entry_qty - filled_exit_qty > 1e-9:
            continue
        realized_pnl = 0
        missing_price = False
        for exit_order in exits:
            exit_fill = _frozen_exit_price(exit_order)
            if exit_fill is None:
                missing_price = True
                break
            exit_qty = exit_order.get("quantity", 0)
            if entry["direction"] == "LONG":
                realized_pnl += (exit_fill - entry_fill) * exit_qty
            else:
                realized_pnl += (entry_fill - exit_fill) * exit_qty
        if missing_price:
            continue
        last_exit = exits[-1]
        last_exit_fill = _frozen_exit_price(last_exit)
        assert last_exit_fill is not None
        funding_row = conn.execute(
            """SELECT COALESCE(SUM(funding_paid), 0) as total FROM orders
            WHERE scan_id = ? AND execution_mode = ? AND exchange = ?""",
            (entry["scan_id"], entry["execution_mode"], entry["exchange"]),
        ).fetchone()
        funding_paid = float(funding_row["total"])
        # PRE-2a BUG: bare LIMIT 1, no ORDER BY — undefined SQLite row order.
        stop_order = conn.execute(
            """SELECT trigger_price FROM orders
            WHERE scan_id = ? AND order_type = 'stop_loss'
            AND execution_mode = ? AND exchange = ?
            LIMIT 1""",
            (entry["scan_id"], entry["execution_mode"], entry["exchange"]),
        ).fetchone()
        stop_price = float(stop_order["trigger_price"]) if stop_order else 0
        stop_distance = abs(entry_fill - stop_price)
        risk_amount = stop_distance * entry_qty if stop_distance > 0 else 0
        realized_r = realized_pnl / risk_amount if risk_amount > 0 else 0
        r_mult = float(entry.get("risk_mult") or 1.0)
        if r_mult < 1.0 - 1e-9:
            realized_r *= r_mult
        entry_time = entry.get("filled_at") or entry.get("placed_at") or entry.get("created_at")
        exit_time = last_exit.get("filled_at") or last_exit.get("created_at")
        holding_hours = 0
        if entry_time and exit_time:
            try:
                t1 = datetime.fromisoformat(entry_time)
                t2 = datetime.fromisoformat(exit_time)
                holding_hours = (t2 - t1).total_seconds() / 3600
            except (ValueError, TypeError):
                pass
        slippage_r = _frozen_slippage_r(entry, last_exit, entry_fill, last_exit_fill, risk_amount, conn)
        exit_type_map = {
            "stop_loss": "SL", "tp1": "TP1", "tp2": "TP2",
            "timeout_close": "TIMEOUT", "emergency_close": "EMERGENCY", "drift_close": "DRIFT",
        }
        exit_type = exit_type_map.get(last_exit["order_type"], last_exit["order_type"])
        trades.append(TradeRecord(
            scan_id=entry["scan_id"], coin=entry["coin"], direction=entry["direction"],
            entry_fill=entry_fill, exit_fill=last_exit_fill, exit_type=exit_type,
            realized_pnl=realized_pnl, realized_r=realized_r, risk_amount=risk_amount,
            holding_hours=holding_hours, funding_paid=funding_paid, slippage_r=slippage_r,
            execution_mode=entry.get("execution_mode", "testnet"),
            usable_risk=risk_amount > 0,
            exchange=entry.get("exchange", "hyperliquid"),
        ))
    return trades


def _frozen_compute_quality_warnings(trades) -> list:  # verbatim pre-2a (list)
    warnings = ["All metrics estimated from DB — not reconciled with exchange"]
    if trades:
        no_funding = sum(1 for t in trades if t.funding_paid == 0)
        if 0 < no_funding < len(trades):
            warnings.append(f"{no_funding}/{len(trades)} trades missing funding data")
        elif no_funding == len(trades):
            warnings.append("Funding costs not captured — PnL overstated")
        approx_fills = sum(1 for t in trades if t.entry_fill == 0 or t.exit_fill == 0)
        if approx_fills:
            warnings.append(f"{approx_fills} trades with missing fill prices — using trigger prices")
    return warnings


def _frozen_compute_live_metrics(trades) -> dict:
    if not trades:
        return {"count": 0, "win_rate": 0, "expectancy_r": 0, "profit_factor": 0,
                "total_pnl": 0, "max_drawdown_pct": 0, "avg_holding_hours": 0,
                "avg_slippage_r": 0, "total_funding_r": 0, "fill_rate": 0,
                "data_source": "db_only",
                "quality_warnings": _frozen_compute_quality_warnings([])}
    wins = [t for t in trades if t.realized_r > 0]
    losses = [t for t in trades if t.realized_r <= 0]
    win_rate = len(wins) / len(trades)
    expectancy_r = sum(t.realized_r for t in trades) / len(trades)
    total_pnl = sum(t.realized_pnl for t in trades)
    gross_profit = sum(t.realized_pnl for t in wins)
    gross_loss = abs(sum(t.realized_pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0
    total_risk = sum(t.risk_amount for t in trades)
    running = peak = max_dd = 0
    for t in trades:
        running += t.realized_pnl
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd
    if peak > 0:
        max_dd_pct = max_dd / peak * 100
    elif total_risk > 0:
        max_dd_pct = max_dd / total_risk * 100
    else:
        max_dd_pct = 0
    avg_holding = sum(t.holding_hours for t in trades) / len(trades)
    avg_slippage = sum(t.slippage_r for t in trades) / len(trades)
    total_funding = sum(t.funding_paid for t in trades)
    total_funding_r = total_funding / total_risk if total_risk > 0 else 0
    return {"count": len(trades), "win_rate": round(win_rate, 4),
            "expectancy_r": round(expectancy_r, 4), "profit_factor": round(profit_factor, 4),
            "total_pnl": round(total_pnl, 2), "max_drawdown_pct": round(max_dd_pct, 2),
            "avg_holding_hours": round(avg_holding, 1), "avg_slippage_r": round(avg_slippage, 4),
            "total_funding_r": round(total_funding_r, 4), "fill_rate": 0,
            "data_source": "db_only",
            "quality_warnings": _frozen_compute_quality_warnings(trades)}


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def _scans_table(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS scans (scan_id TEXT PRIMARY KEY, "
        "machine_block TEXT, cost_usd REAL, created_at TEXT)"
    )


def _load_prod_corpus() -> sqlite3.Connection:
    """In-memory DB with prod orders inserted in prod ROWID ORDER (list
    order) so the frozen-old bare-`LIMIT 1` reproduces prod SL selection."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_orders_table(conn)
    _scans_table(conn)
    cols = _CORPUS["order_columns"]
    qs = ", ".join("?" * len(cols))
    collist = ", ".join(cols)
    for o in _CORPUS["orders"]:  # list order == prod rowid order
        conn.execute(
            f"INSERT INTO orders ({collist}) VALUES ({qs})",
            [o.get(c) for c in cols],
        )
    for s in _CORPUS["scans"]:
        conn.execute(
            "INSERT OR REPLACE INTO scans (scan_id, machine_block, cost_usd, created_at) "
            "VALUES (?, ?, ?, ?)",
            (s["scan_id"], s.get("machine_block"), s.get("cost_usd"), s.get("created_at")),
        )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# (a) Faithful-relocation over the prod corpus
# ---------------------------------------------------------------------------

def test_rowid_order_reproduces_prod_limit1_selection():
    """Guard: the in-memory load must reproduce prod's bare-LIMIT-1 SL pick
    (recorded `limit1_sl`) for every multi-SL scan — otherwise the
    faithful-relocation oracle below is not a prod-equivalent oracle."""
    conn = _load_prod_corpus()
    assert _CORPUS["multi_sl"], "corpus must contain multi-SL scans"
    for m in _CORPUS["multi_sl"]:
        row = conn.execute(
            """SELECT trigger_price FROM orders WHERE scan_id = ?
            AND order_type = 'stop_loss' AND execution_mode = ? AND exchange = ?
            LIMIT 1""",
            (m["scan_id"], m["execution_mode"], m["exchange"]),
        ).fetchone()
        assert row is not None
        assert float(row["trigger_price"]) == float(m["limit1_sl"]), (
            f"in-memory bare-LIMIT-1 for {m['scan_id']} != recorded prod limit1_sl"
        )


def test_faithful_relocation_trade_records():
    """new get_closed_trades == FROZEN-old, field-for-field, every prod row.
    (Holds because prod limit1 == first-by-created_at for all 72/72; this
    catches ANY relocation perturbation.)"""
    conn = _load_prod_corpus()
    old = {t.scan_id: t for t in _frozen_get_closed_trades(conn)}
    new = {t.scan_id: t for t in get_closed_trades(conn)}
    assert old.keys() == new.keys()
    assert len(new) > 0
    for sid in old:
        assert dataclasses.asdict(new[sid]) == dataclasses.asdict(old[sid]), sid


def test_metrics_relocation_faithful_except_documented_24_exclusion():
    """T-016 correctness-fix (SPEC #24, KL#2 v) deliberately supersedes Phase
    2a's "metrics byte-identical" contract: zero-risk trades (risk_amount==0 —
    no/NULL SL or degenerate entry==SL) are now EXCLUDED from R-normalized
    metrics instead of being silently counted as break-even losses. On the
    prod corpus this is NOT a no-op: 5/221 trades have entry_fill == original
    SL (zero risk distance, R undefined; ~$0 scratches), shifting win_rate
    0.3303→0.338 and expectancy 0.0551→0.0564.

    This test now pins that the relocated `compute_live_metrics` changed by
    EXACTLY the documented #24 exclusion and nothing else: NEW == frozen-old
    logic with R-fields over `risked` and non-R fields over all trades. The
    pure read-model faithful-relocation guarantee still lives, unweakened, in
    `test_faithful_relocation_trade_records` (TradeRecords byte-identical)."""
    conn = _load_prod_corpus()
    ft = _frozen_get_closed_trades(conn)
    risked = [t for t in ft if t.risk_amount > 0]
    n_zero = sum(1 for t in ft if t.risk_amount == 0)
    assert n_zero == 5, f"prod corpus zero-risk count drifted: {n_zero}"

    # R-normalized fields: frozen-old logic fed only `risked` reproduces NEW.
    exp_r = _frozen_compute_live_metrics(risked)
    # Non-R fields: frozen-old over ALL trades (unchanged by #24).
    exp_all = _frozen_compute_live_metrics(ft)
    new = compute_live_metrics(get_closed_trades(conn))

    for k in ("win_rate", "expectancy_r", "profit_factor",
              "avg_slippage_r", "total_funding_r"):
        assert getattr(new, k) == exp_r[k], f"R-field {k}: new={getattr(new,k)} exp={exp_r[k]}"
    for k in ("count", "total_pnl", "max_drawdown_pct", "avg_holding_hours",
              "fill_rate", "data_source"):
        assert getattr(new, k) == exp_all[k], f"non-R {k}: new={getattr(new,k)} exp={exp_all[k]}"

    # quality_warnings: frozen content + the #24 no-usable-risk line.
    exp_w = tuple(_frozen_compute_quality_warnings(ft)) + (
        f"{n_zero} trades with no usable risk anchor — excluded from R metrics",
    )
    assert new.quality_warnings == exp_w, f"warnings: new={new.quality_warnings} exp={exp_w}"

    # Anti-vacuity: the documented exclusion actually moves prod metrics.
    assert new.win_rate != exp_all["win_rate"]
    assert new.expectancy_r != exp_all["expectancy_r"]


def test_faithful_relocation_days_window():
    """The pre-2a `if days:` close-window SQL branch was relocated verbatim
    too — exercise it in BOTH frozen-old and new and assert identical, at
    days=3650 (corpus age < 10y → all rows) and days=1 (corpus is old →
    none). Deterministic regardless of wall clock; covers the branch the
    no-arg oracle never reaches."""
    conn = _load_prod_corpus()
    for days in (3650, 1):
        old = {t.scan_id: t for t in _frozen_get_closed_trades(conn, days=days)}
        new = {t.scan_id: t for t in get_closed_trades(conn, days=days)}
        assert old.keys() == new.keys(), f"days={days}"
        for sid in old:
            assert dataclasses.asdict(new[sid]) == dataclasses.asdict(old[sid]), (days, sid)
    # sanity: the two windows are not both empty (3650 must include rows)
    assert len(get_closed_trades(conn, days=3650)) > 0


# ---------------------------------------------------------------------------
# (b) Correctness — original_stop_price == first-by-created_at
# ---------------------------------------------------------------------------

def test_original_stop_price_is_first_by_created_at():
    """For every multi-SL prod scan, the storage-owned original_stop_price
    returns the recorded first-by-`created_at` SL (the risk anchor)."""
    conn = _load_prod_corpus()
    checked = 0
    for m in _CORPUS["multi_sl"]:
        got = original_stop_price(
            m["scan_id"], conn, m["execution_mode"], m["exchange"],
        )
        assert got == float(m["first_sl"]), (
            f"{m['scan_id']}: original_stop_price={got} != first_sl={m['first_sl']}"
        )
        checked += 1
    assert checked == len(_CORPUS["multi_sl"]) >= 72


# ---------------------------------------------------------------------------
# (c) Anti-vacuity by construction — adversarial scan
# ---------------------------------------------------------------------------

_ADV = "adv_phase2a"
_ADV_INSERT_ORDER = [
    # entry first (earliest created_at)
    {"order_type": "entry", "status": "FILLED", "entry_price": 105.0,
     "fill_price": 105.0, "quantity": 1.0, "created_at": "2026-01-01T00:00:00+00:00",
     "filled_at": "2026-01-01T00:00:01+00:00"},
    # SL_drift: LATER created_at but inserted SECOND-of-the-SLs FIRST →
    # lower rowid → a bare LIMIT 1 (rowid order) selects THIS (non-original).
    {"order_type": "stop_loss", "status": "CANCELLED", "trigger_price": 95.0,
     "quantity": 1.0, "created_at": "2026-01-01T06:00:00+00:00"},
    # SL_orig: EARLIER created_at, inserted AFTER → higher rowid. This is the
    # true original (first by created_at) the fix must select.
    {"order_type": "stop_loss", "status": "CANCELLED", "trigger_price": 100.0,
     "quantity": 1.0, "created_at": "2026-01-01T00:00:00+00:00"},
    # exit: drift_close FILLED, fully closes the position.
    {"order_type": "drift_close", "status": "FILLED", "fill_price": 102.0,
     "quantity": 1.0, "created_at": "2026-01-01T12:00:00+00:00",
     "filled_at": "2026-01-01T12:00:00+00:00"},
]


def _load_adversarial() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_orders_table(conn)
    _scans_table(conn)
    for i, o in enumerate(_ADV_INSERT_ORDER):
        conn.execute(
            """INSERT INTO orders (order_id, scan_id, coin, direction, order_type,
            status, entry_price, fill_price, quantity, trigger_price,
            filled_at, exchange, execution_mode, created_at)
            VALUES (?, ?, 'ADV', 'LONG', ?, ?, ?, ?, ?, ?, ?, 'hyperliquid', 'live', ?)""",
            (f"{_ADV}_{i}", _ADV, o["order_type"], o["status"],
             o.get("entry_price"), o.get("fill_price"), o.get("quantity"),
             o.get("trigger_price"), o.get("filled_at"), o["created_at"]),
        )
    conn.commit()
    return conn


def test_adversarial_frozen_old_selects_non_original_sl():
    """Proves the divergence is CAUSED by the undefined-order bug: frozen-old's
    bare LIMIT 1 picks SL_drift (95.0, lower rowid), not the original
    SL_orig (100.0)."""
    conn = _load_adversarial()
    row = conn.execute(
        """SELECT trigger_price FROM orders WHERE scan_id = ?
        AND order_type = 'stop_loss' AND execution_mode = ? AND exchange = ?
        LIMIT 1""",
        (_ADV, "live", "hyperliquid"),
    ).fetchone()
    assert float(row["trigger_price"]) == 95.0  # the NON-original (bug)
    # new owner selects the true original (earliest created_at)
    assert original_stop_price(_ADV, conn, "live", "hyperliquid") == 100.0


def test_adversarial_anti_vacuity_realized_r_diverges():
    """Tripwire fires BY CONSTRUCTION: frozen-old vs new realized_r differ,
    and new == the hand-derived original-SL R. A regression that re-introduces
    undefined-order SL selection fails closed here.

    LONG entry 105, drift_close fill 102, qty 1 → pnl = (102-105)*1 = -3.
      frozen-old risk = |105 - 95| * 1 = 10  → R = -3/10 = -0.3
      new       risk = |105 -100| * 1 =  5  → R = -3/ 5 = -0.6
    """
    conn = _load_adversarial()
    old = {t.scan_id: t for t in _frozen_get_closed_trades(conn)}
    new = {t.scan_id: t for t in get_closed_trades(conn)}
    assert _ADV in old and _ADV in new
    o, n = old[_ADV], new[_ADV]
    assert o.realized_r == -0.3          # bug: anchored on non-original SL
    assert n.realized_r == -0.6          # fix: anchored on original SL
    assert n.realized_r != o.realized_r  # anti-vacuity: provably diverges
    assert n.risk_amount == 5.0          # |105 - 100| * 1, hand-derived
    assert o.risk_amount == 10.0         # |105 -  95| * 1
