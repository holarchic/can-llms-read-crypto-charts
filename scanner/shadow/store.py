"""Shadow-signal capture store (T-020.1, SPEC #41).

`save_shadow_signal` persists ONE post-validator CONFIRM — the faithful tradeable
cohort, captured at scan time after `validate_confirms` (NOT a raw `scans` row, which
loses `BoundarySignal.confluence` and can hold `machine_block=NULL`). The full
`machine_block` geometry + confluence count are recorded so the forward resolver can
walk each signal to a realized R without re-running the LLM. Idempotent on `scan_id`
(`INSERT OR IGNORE`) so a re-captured scan never double-counts.
"""

import json
from datetime import datetime
from typing import NamedTuple

from scanner.models import MachineBlock, ScanResult, ShadowOutcome

from .resolver import RESOLVED, RETRYABLE

__all__ = [
    "save_shadow_signal",
    "LoadedShadowSignal",
    "load_unresolved_signals",
    "save_resolution",
    "resolved_r_values",
]


class LoadedShadowSignal(NamedTuple):
    """A captured signal rehydrated for forward resolution."""

    id: int
    coin: str
    exchange: str
    anchor_ts: datetime
    machine_block: MachineBlock


def save_shadow_signal(
    result: ScanResult, anchor_ts: datetime, exchange: str, conn,
) -> bool:
    """Persist `result` (a post-validator CONFIRM) to `shadow_signals`.

    `anchor_ts` is the scan-run moment — the time live would have acted, and the
    forward-resolution start. `exchange` (from `cfg.exchange`) pins the venue so the
    resolver fetches the right book's OHLCV. Returns True if a new row was inserted,
    False if an existing `scan_id` was ignored (idempotent capture).
    """
    mb = result.machine_block
    cur = conn.execute(
        "INSERT OR IGNORE INTO shadow_signals "
        "(scan_id, coin, exchange, anchor_ts, setup_type, direction, "
        " prompt_version, confluence_count, machine_block, validator_verdict) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            result.scan_id,
            result.coin,
            exchange,
            anchor_ts.isoformat(),
            mb.setup_type,
            mb.direction,
            result.prompt_version,
            len(result.setup_signal.confluence),
            json.dumps(mb.to_raw()),
            result.validator_verdict,
        ),
    )
    conn.commit()
    return cur.rowcount == 1


def load_unresolved_signals(
    conn, exit_policy_version: str, entry_policy_version: str,
) -> list[LoadedShadowSignal]:
    """Captured signals with no terminal resolution yet for this (entry, exit) policy pair —
    never resolved, or last left in a `RETRYABLE` state (OPEN / DATA_INCOMPLETE) so the tape
    stays re-runnable as forward data arrives. Reads fields by name; a persisted
    `machine_block` that fails to rehydrate is a corrupt capture and fails loud here."""
    placeholders = ", ".join("?" for _ in RETRYABLE)
    rows = conn.execute(
        "SELECT s.id AS id, s.coin AS coin, s.exchange AS exchange, "
        "       s.anchor_ts AS anchor_ts, s.machine_block AS machine_block "
        "FROM shadow_signals s "
        "LEFT JOIN shadow_resolutions r "
        "  ON r.shadow_signal_id = s.id "
        "  AND r.exit_policy_version = ? AND r.entry_policy_version = ? "
        f"WHERE r.id IS NULL OR r.status IN ({placeholders})",
        (exit_policy_version, entry_policy_version, *sorted(RETRYABLE)),
    ).fetchall()
    out: list[LoadedShadowSignal] = []
    for row in rows:
        mb = MachineBlock.from_raw(json.loads(row["machine_block"]))
        if mb is None:
            raise ValueError(
                f"shadow_signals id={row['id']}: persisted machine_block failed to "
                "rehydrate (corrupt capture) — refusing to resolve it"
            )
        out.append(LoadedShadowSignal(
            id=row["id"], coin=row["coin"], exchange=row["exchange"],
            anchor_ts=datetime.fromisoformat(row["anchor_ts"]), machine_block=mb,
        ))
    return out


def save_resolution(
    conn,
    shadow_signal_id: int,
    outcome: ShadowOutcome,
    exit_policy_version: str,
    entry_policy_version: str,
    resolved_at: datetime,
) -> None:
    """Upsert one resolution. Idempotent on (signal, exit_policy, entry_policy): a
    re-run overwrites an earlier `OPEN` with its terminal outcome, and a different
    policy pair adds a row beside it (the re-runnable tape)."""
    o = outcome
    conn.execute(
        "INSERT INTO shadow_resolutions "
        "(shadow_signal_id, exit_policy_version, entry_policy_version, status, "
        " entry_fill, exit_price, exit_reason, realized_r, fees_r, slippage_r, "
        " bars_held, low_fidelity, resolved_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (shadow_signal_id, exit_policy_version, entry_policy_version) "
        "DO UPDATE SET status=excluded.status, entry_fill=excluded.entry_fill, "
        " exit_price=excluded.exit_price, exit_reason=excluded.exit_reason, "
        " realized_r=excluded.realized_r, fees_r=excluded.fees_r, "
        " slippage_r=excluded.slippage_r, bars_held=excluded.bars_held, "
        " low_fidelity=excluded.low_fidelity, resolved_at=excluded.resolved_at",
        (
            shadow_signal_id, exit_policy_version, entry_policy_version, o.status,
            o.entry_fill, o.exit_price, o.exit_reason, o.realized_r, o.fees_r,
            o.slippage_r, o.bars_held, int(o.low_fidelity), resolved_at.isoformat(),
        ),
    )
    conn.commit()


def resolved_r_values(
    conn, exit_policy_version: str, entry_policy_version: str,
) -> list[float]:
    """Per-trade net realized R for every RESOLVED trade under this policy pair — the
    expectancy sample. 0R outcomes stay in; non-trade statuses never have realized_r."""
    rows = conn.execute(
        "SELECT realized_r AS realized_r FROM shadow_resolutions "
        "WHERE exit_policy_version = ? AND entry_policy_version = ? "
        "AND status = ? AND realized_r IS NOT NULL",
        (exit_policy_version, entry_policy_version, RESOLVED),
    ).fetchall()
    return [float(row["realized_r"]) for row in rows]
