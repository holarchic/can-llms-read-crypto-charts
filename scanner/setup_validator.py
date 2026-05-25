"""Post-LLM structural gate for the v10 path (T-019 Chunk 8a, SPEC #33).

The v10 design assumption — "assume the LLM confirms anything plausible" — is only
acceptable behind a deterministic structural gate that rejects over-eager CONFIRMs.
``validate_confirms`` is that gate, wired as one interception in ``pipeline.run_scan``
between stage-2 analyze and stage-3 alert. It demotes a failed ``SETUP_VERSION='v2'``
CONFIRM by re-typing its ``ScanResult.decision`` to ``"REJECT"`` — so **both**
``stage3_alert`` and stage-4 execution (each filtering ``is_confirm``) skip it from one
interception — and tags ``ScanResult.validator_verdict`` so the demotion stays
distinguishable from an LLM REJECT (and partitions the scan summary).

Two gates, evaluated floor-first:

1. **Confluence-count floor** — ``len(signal.confluence) < config.v2_min_confluence``.
   The detectors emit at confluence ≥1 and per-coin selection keeps the survivor
   regardless of absolute count, so a marginal setup (e.g. a lone-``SWEEP_RECLAIM``
   ``sfp_range``) reaches the trigger-happy LLM and gets confirmed — only this floor
   rejects it. Floors the raw factor *count* (the pre-documented intent at
   ``models.ConfluenceFactor``), not the ``_stage1_score`` (whose weights are a
   Chunk-8b placeholder ranker). v2-only: the v1 ``confidence`` is the anti-predictive
   screener score (decision #2); the v2 confluence count is a structural measure.

2. **Structural-precondition recheck** — a fail-closed integrity guard over the
   carried ``SpineContext``: non-finite ``confidence`` / ``atr`` ≤0 / ``current_price``
   ≤0 / empty ``regime`` → ``REJECT_DEGRADED_CONTEXT``. This catches a degraded /
   default-constructed projection that cannot stand behind a CONFIRM (no risk
   geometry, unclassified regime). It is *pure integrity* — it deliberately does NOT
   re-assert each setup's regime contract: the detector already hard-gates regime
   (``setups_v2`` returns ``[]`` on a wrong regime) and ``context.regime`` is fixed at
   detection, so a per-setup regime recheck would be a within-scan tautology whose only
   possible effect is a false reject on policy drift — duplicated detector policy, not
   added safety (arch-pass R1). The band/trend-state MA preconditions are likewise not
   re-checkable here — they are not projected onto ``SpineContext``.

Fail-closed: any unexpected exception while evaluating a CONFIRM demotes it
(``REJECT_VALIDATOR_ERROR``, a distinct token so a Chunk-8b replay can tell a validator
defect from a degraded context) rather than letting it through or crashing the scan —
a CONFIRM we cannot validate must not alert or execute. Pure compute: no DB, no LLM.
v1 byte-identical (early return on ``setup_version != 'v2'``). Defaults calibrated in
Chunk 8b against seeded over-eager confirms (Acceptance #9 b).
"""

import math

from .config import ScannerConfig
from .models import BoundarySignal, ScanResult

__all__ = [
    "validate_confirms",
    "REJECT_CONFLUENCE_FLOOR",
    "REJECT_DEGRADED_CONTEXT",
    "REJECT_VALIDATOR_ERROR",
]

REJECT_CONFLUENCE_FLOOR = "REJECT_CONFLUENCE_FLOOR"
REJECT_DEGRADED_CONTEXT = "REJECT_DEGRADED_CONTEXT"
REJECT_VALIDATOR_ERROR = "REJECT_VALIDATOR_ERROR"


def _finite_pos(x: float) -> bool:
    return math.isfinite(x) and x > 0


def _floor_verdict(sig: BoundarySignal, min_confluence: int) -> str | None:
    if len(sig.confluence) < min_confluence:
        return REJECT_CONFLUENCE_FLOOR
    return None


def _structural_verdict(sig: BoundarySignal) -> str | None:
    ctx = sig.context
    if not math.isfinite(sig.confidence):
        return REJECT_DEGRADED_CONTEXT
    if not _finite_pos(ctx.atr) or not _finite_pos(ctx.current_price):
        return REJECT_DEGRADED_CONTEXT
    if not ctx.regime:
        return REJECT_DEGRADED_CONTEXT
    return None


def _verdict(result: ScanResult, min_confluence: int) -> str | None:
    """Floor first, then the structural recheck; None ⇒ the CONFIRM stands."""
    sig = result.setup_signal
    return _floor_verdict(sig, min_confluence) or _structural_verdict(sig)


def validate_confirms(
    results: list[ScanResult],
    config: ScannerConfig,
    *,
    verbose: bool = False,
) -> list[ScanResult]:
    """Demote each over-eager v2 CONFIRM in place; return the same list.

    No-op (returns ``results`` unchanged) on the v1 path and for any non-CONFIRM
    result. Demotion sets ``decision='REJECT'`` + ``validator_verdict``.
    """
    if config.setup_version != "v2":
        return results
    for r in results:
        if not r.is_confirm:
            continue
        try:
            verdict = _verdict(r, config.v2_min_confluence)
        except Exception as e:  # fail-closed — an unvalidatable CONFIRM must not trade
            print(f"  setup_validator error on {r.coin}: {type(e).__name__}: {e}")
            verdict = REJECT_VALIDATOR_ERROR
        if verdict:
            r.decision = "REJECT"
            r.validator_verdict = verdict
            if verbose:
                print(f"  setup_validator demoted {r.coin}: {verdict}")
    return results
