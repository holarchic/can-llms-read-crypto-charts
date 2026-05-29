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

from scanner.models import ScanResult

__all__ = ["save_shadow_signal"]


def save_shadow_signal(
    result: ScanResult, anchor_ts: datetime, exchange: str, conn,
) -> None:
    """Persist `result` (a post-validator CONFIRM) to `shadow_signals`.

    `anchor_ts` is the scan-run moment — the time live would have acted, and the
    forward-resolution start. `exchange` (from `cfg.exchange`) pins the venue so the
    resolver fetches the right book's OHLCV.
    """
    mb = result.machine_block
    conn.execute(
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
