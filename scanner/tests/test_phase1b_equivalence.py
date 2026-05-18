"""T-016 Phase 1b equivalence guard (SPEC #21).

LESSONS-CODING L5: a green suite proves nothing for a type-narrowing refactor
unless a test passes the NEW type through each consumer over real data. This
drives the REAL `execute_trade` consumer with the request from the new
`entry_monitor._build_execution_request` AND with a request derived from a
FROZEN verbatim copy of the deleted `_reconstruct_scan_result` mapping, over
every one of the 706 prod-synced corpus machine_blocks, and asserts the two
produce a byte-identical `ExecutionResult`. Exchange/network stubbed; order
ids, the builder `staleness_ref`, AND the executor's own `datetime.now` all
frozen to the same instant so the staleness guard does not short-circuit
every row (which would make the consumer oracle vacuous) and so equality is
deterministic; a fresh in-memory DB per `execute_trade` call so the paired
runs cannot contaminate each other.

The frozen-old helper is the immutable baseline — it reproduces exactly what
the pre-Phase-1b `execute_trade` read off the synthetic `ScanResult`
(`_reconstruct_scan_result` + `_empty_setup_signal`). If the new builder's
wiring drifts from that mapping for any real block, the paired `execute_trade`
outcomes diverge and this fails.
"""

import copy
import dataclasses
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scanner.config import ScannerConfig
from scanner.execution.config import ExecutionConfig
from scanner.execution.entry_monitor import _build_execution_request
from scanner.execution.executor import execute_trade
from scanner.execution.storage import ensure_orders_table
from scanner.models import Direction, ExecutionRequest, MachineBlock

_CORPUS = json.loads(
    (Path(__file__).parent / "fixtures" / "machine_block_corpus.json").read_text()
)
_ROWS = [r for t in ("scans", "pending_entries") for r in _CORPUS[t]]

_FROZEN_NOW = datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)


def _trigger_close_for(raw: str) -> float:
    """Trigger close = the block's OWN entry midpoint so `reanchored()`'s
    delta is ≈0 and the (valid) corpus geometry is preserved. A fixed
    trigger_close would shift tiny-priced coins (entry ~0.00174) by ~100,
    fabricating universally-invalid geometry that never reaches
    sizing/placement — i.e. a shallow consumer oracle (the depth concern
    codex Phase-1b correctness raised). Reanchor parity itself is covered by
    test_machine_block; here we want the executor body exercised."""
    d = json.loads(raw)
    lo, hi = d.get("entry_low"), d.get("entry_high")
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo:
        return (lo + hi) / 2
    return 100.0


def _old_request_frozen(pending: dict, trigger_close: float) -> ExecutionRequest:
    """Frozen verbatim copy of the deleted entry_monitor._reconstruct_scan_result
    + _empty_setup_signal logic, projected to exactly the seven values the
    pre-Phase-1b execute_trade read off the synthetic ScanResult. DO NOT
    "simplify" toward _build_execution_request — this is the independent
    baseline; its whole value is being a separate derivation.
    """
    raw = json.loads(pending["machine_block"])
    mb = MachineBlock.from_raw(raw)
    mb_out = mb.reanchored(trigger_close) if mb is not None else None

    # Old: setup_signal = _empty_setup_signal(coin, Direction(sd)) only when sd
    # is truthy AND a valid Direction; on ValueError setup_signal stayed None.
    # execute_trade then read `result.setup_signal.direction.value if
    # result.setup_signal else ""`. As an ExecutionRequest projection (where
    # execute_trade does `req.screener_direction or ""`): valid+truthy →
    # Direction(sd).value; else → None (None or "" == "" == the old "").
    screener_dir = pending.get("screener_direction")
    old_screener = None
    if screener_dir:
        try:
            old_screener = Direction(screener_dir).value
        except ValueError:
            old_screener = None
    return ExecutionRequest(
        scan_id=pending["scan_id"],
        coin=pending["coin"],
        machine_block=mb_out,
        screener_direction=old_screener,
        atr=0.0,            # _empty_setup_signal context.atr default
        max_leverage=None,  # SetupSignal.max_leverage default
        staleness_ref=_FROZEN_NOW,
    )


def _exec_config() -> ExecutionConfig:
    return ExecutionConfig(
        enabled=True,
        mode="testnet",
        private_key="0xtest",
        wallet_address="0xwallet",
        risk_pct=1.0,
        btc_vol_gate_enabled=False,
        discord_ops_webhook_url="https://discord.test/ops",
        discord_urgent_webhook_url="https://discord.test/urgent",
    )


def _entry_mid(req: ExecutionRequest) -> float:
    mb = req.machine_block
    if mb is None:
        return 100.0
    return (mb.entry_low + mb.entry_high) / 2


def _run_execute(req: ExecutionRequest) -> dict:
    """Drive the real execute_trade with a deterministic stubbed exchange and
    a fresh in-memory DB. Returns dataclasses.asdict(ExecutionResult) so the
    paired old/new runs are compared by value.

    executor.datetime is pinned to _FROZEN_NOW so the staleness guard sees
    `req.staleness_ref` (also _FROZEN_NOW) as age≈0 and the consumer actually
    proceeds into machine_block / sanity / sizing / order-placement. Without
    this pin, real wall-clock vs the frozen ref makes EVERY row exit at
    `stale_signal` and the oracle is vacuous (codex Phase-1b correctness
    review — the exact LESSONS-CODING L5 trap this guard exists to defeat).
    _FROZEN_NOW (a real datetime) keeps `.isoformat()` working for the
    order-row timestamps executor also derives from datetime.now().
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_orders_table(conn)
    exchange = MagicMock()
    exchange.create_order.return_value = {
        "id": "EX", "filled": None, "average": None, "price": None,
    }
    exchange.amount_to_precision.side_effect = lambda s, q: str(q)
    exchange.fetch_ticker.return_value = {"last": _entry_mid(req)}
    with patch("scanner.execution.executor.datetime") as edt, \
         patch("scanner.execution.executor.get_open_positions", return_value=[]), \
         patch("scanner.execution.executor.get_auth_exchange", return_value=exchange), \
         patch("scanner.execution.executor.fetch_balance", return_value=1_000_000.0), \
         patch("scanner.execution.executor.fetch_current_price",
               return_value=_entry_mid(req)), \
         patch("scanner.execution.executor.get_exchange_open_orders", return_value=[]), \
         patch("scanner.execution.executor.is_killed", return_value=False), \
         patch("scanner.execution.executor.new_order_id", return_value="OID"), \
         patch("scanner.execution.executor.make_cloid", return_value="CLOID"), \
         patch("scanner.execution.alerts.requests.post",
               return_value=MagicMock(status_code=204)):
        edt.now.return_value = _FROZEN_NOW
        try:
            er = execute_trade(req, _exec_config(), ScannerConfig(), conn)
        finally:
            conn.close()
    return dataclasses.asdict(er)


@pytest.mark.parametrize(
    "raw", _ROWS, ids=[f"row{i}" for i in range(len(_ROWS))]
)
def test_corpus_execute_trade_old_eq_new(raw):
    """For every prod corpus block: execute_trade driven by the new builder's
    request == execute_trade driven by the frozen-old derivation's request."""
    pending = {
        "scan_id": "scan-eq",
        "coin": "BTC",
        "screener_direction": json.loads(raw).get("direction"),
        "machine_block": raw,
    }
    tc = _trigger_close_for(raw)
    with patch("scanner.execution.entry_monitor.datetime") as mdt:
        mdt.now.return_value = _FROZEN_NOW
        new_req = _build_execution_request(copy.deepcopy(pending), tc)
    old_req = _old_request_frozen(copy.deepcopy(pending), tc)

    # The builder wiring must reproduce the frozen-old projection exactly.
    assert new_req == old_req

    # And the NEW type must drive the real consumer to a byte-identical
    # outcome (passes the new type through execute_trade over real data).
    assert _run_execute(new_req) == _run_execute(old_req)


def test_corpus_size():
    assert len(_ROWS) == 706, "corpus must be 556 scans + 150 pending_entries"


def test_consumer_oracle_is_not_stale_vacuous():
    """Anti-vacuity tripwire (codex Phase-1b correctness review): prove the
    paired execute_trade oracle actually traverses the consumer past the
    staleness guard. If a future change reintroduces the frozen-ref-vs-
    wall-clock skew, every row would exit at `stale_signal`, the per-row
    old==new assertions would still pass (both stale), and the guard would
    silently stop testing anything. This fails loudly if that happens.
    """
    outcomes = []
    for raw in _ROWS[:60]:
        pending = {
            "scan_id": "scan-eq",
            "coin": "BTC",
            "screener_direction": json.loads(raw).get("direction"),
            "machine_block": raw,
        }
        with patch("scanner.execution.entry_monitor.datetime") as mdt:
            mdt.now.return_value = _FROZEN_NOW
            req = _build_execution_request(
                copy.deepcopy(pending), _trigger_close_for(raw)
            )
        er = _run_execute(req)
        outcomes.append("executed" if er["executed"] else er["skip_reason"])

    assert not all(o == "stale_signal" for o in outcomes), (
        "consumer oracle is stale-vacuous — every row short-circuited at the "
        "staleness guard; the execute_trade comparison is testing nothing"
    )
    # Stronger than not-all-stale: at least one row must reach the deep
    # consumer body (step-2 validate or beyond — only reachable past the
    # step-0 staleness + step-0.6 reconciliation guards). Geometry-preserving
    # trigger_close keeps valid corpus geometry so blocks land in
    # validate/sanity/sizing/placement, not a shallow early skip. (Asserting
    # exactly `executed` would force brittle full-fill exchange mocking for
    # zero added oracle strength — the per-row old==new test over all 706
    # rows is the actual equivalence proof; this only guards vacuity.)
    _DEEP = {"executed", "entry_failed", "sizing_fail", "sanity_fail",
             "price_outside_zone", "invalid_params", "insufficient_margin",
             "notional_limit"}
    assert _DEEP & set(outcomes), (
        f"oracle never reached the deep consumer body (validate/sizing/"
        f"placement) — only shallow outcomes {set(outcomes)}; the "
        f"execute_trade comparison is not exercising the executor"
    )
