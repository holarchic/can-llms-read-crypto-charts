"""Shadow forward-resolution orchestration (T-020.1 / T-020.12, SPEC #41/#51).

`run_resolution` is the load → resolve → save → expectancy flow behind
`scripts/shadow-resolve.py`, extracted into the package so the money-path policy threading is
unit-tested directly (no importlib of a hyphenated script). The single `exit_policy` argument
keys EVERY store call (load / save / report) and the resolver, so a v1 vs v2 tape can never be
cross-keyed into a false Gate-#10 sample. Returns `(report, by_status)`; the caller renders.
"""

from typing import NamedTuple

import ccxt

from scanner.config import get_exchange_meta
from scanner.expectancy import ExpectancyReport, expectancy_report
from scanner.shadow.resolver import ENTRY_POLICY_VERSION, resolve_signal
from scanner.shadow.store import load_unresolved_signals, resolved_r_values, save_resolution

__all__ = ["ResolutionRun", "run_resolution"]


class ResolutionRun(NamedTuple):
    """The outcome of one resolve pass — named so callers read `.report`/`.by_status` rather
    than a positional tuple (and `.13` can add fields without breaking call sites)."""

    report: ExpectancyReport
    by_status: dict[str, int]


def run_resolution(conn, *, exit_policy, bars, fetch, now, log=print) -> ResolutionRun:
    """Resolve every unresolved shadow signal forward under `exit_policy`; return a
    `ResolutionRun(report, by_status)`.

    `fetch(coin, tf, bars, exchange_id=…)` supplies the OHLCV frame, `now` stamps each
    resolution, `log` receives per-coin progress lines (both injected so the flow is pure of
    network + clock + argparse). `exit_policy` is threaded to `load_unresolved_signals`,
    `resolve_signal`, `save_resolution`, AND `resolved_r_values` — one value, four call-sites,
    no mismatch. `setup_tf` rides from each loaded signal into the resolver's v2 trail.
    """
    signals = load_unresolved_signals(conn, exit_policy, ENTRY_POLICY_VERSION)
    log(f"resolving {len(signals)} unresolved shadow signals (1H, {bars} bars, exit={exit_policy})...")

    by_status: dict[str, int] = {}
    for s in signals:
        try:
            frame = fetch(s.coin, "1h", bars, exchange_id=s.exchange)
        except (ValueError, TypeError, ccxt.BaseError) as exc:  # feed/validation failure:
            log(f"  {s.coin}: OHLCV fetch failed — {exc}")           # skip coin, continue batch
            continue
        outcome = resolve_signal(
            s.machine_block, s.anchor_ts, frame,
            exit_policy=exit_policy, setup_tf=s.setup_tf,
            taker_fee=get_exchange_meta(s.exchange)["fee_rate"],
        )
        save_resolution(conn, s.id, outcome, exit_policy, ENTRY_POLICY_VERSION, now)
        by_status[outcome.status] = by_status.get(outcome.status, 0) + 1

    report = expectancy_report(resolved_r_values(conn, exit_policy, ENTRY_POLICY_VERSION))
    return ResolutionRun(report=report, by_status=by_status)
