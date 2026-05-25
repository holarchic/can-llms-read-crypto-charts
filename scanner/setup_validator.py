"""Post-LLM structural gate for the v10 path (T-019 Chunk 8a, SPEC #33).

The v10 design assumption — "assume the LLM confirms anything plausible" — is only
acceptable behind a deterministic structural gate that rejects over-eager CONFIRMs.
``validate_confirms`` is that gate, wired as one interception in ``pipeline.run_scan``
between stage-2 analyze and stage-3 alert. It demotes a failed ``SETUP_VERSION='v2'``
CONFIRM by re-typing its ``ScanResult.decision`` to ``"REJECT"`` — so **both**
``stage3_alert`` and stage-4 execution (each filtering ``is_confirm``) skip it from one
interception — and tags ``ScanResult.validator_verdict`` so the demotion stays
distinguishable from an LLM REJECT (and partitions the scan summary).

**The gate is the confluence-count floor:** ``len(signal.confluence) <
config.v2_min_confluence``. The detectors emit at confluence ≥1 and per-coin selection
keeps the survivor regardless of absolute count, so a marginal setup (e.g. a
lone-``SWEEP_RECLAIM`` ``sfp_range``) reaches the trigger-happy LLM and gets confirmed —
only this floor rejects it. It floors the raw factor *count* (the pre-documented intent
at ``models.ConfluenceFactor``), not the ``_stage1_score`` (whose weights are a Chunk-8b
placeholder ranker). v2-only: the v1 ``confidence`` is the anti-predictive screener
score (decision #2); the v2 confluence count is a structural measure.

This is the *one genuinely post-LLM, additive* check. The structural preconditions
(finite ATR / price, regime, MA-state) are enforced at the detector/producer boundary
where the signal is constructed — the v2 detectors ``return []`` on bad ATR / wrong
regime, and ``screener_v2`` only emits well-formed ``BoundarySignal``s — so a
downstream re-check here would be redundant re-validation of a trusted typed input
(arch-pass R1/R2; rules #27/#31). The validator trusts its typed input and does not
catch broadly: a malformed signal is a producer defect that must fail loudly, not a
business rejection (rule #35). Pure compute: no DB, no LLM. v1 byte-identical (early
return on ``setup_version != 'v2'``). The floor is calibrated in Chunk 8b against
seeded over-eager confirms (Acceptance #9 b).
"""

from .config import ScannerConfig
from .models import ScanResult

__all__ = ["validate_confirms", "REJECT_CONFLUENCE_FLOOR"]

REJECT_CONFLUENCE_FLOOR = "REJECT_CONFLUENCE_FLOOR"


def validate_confirms(
    results: list[ScanResult],
    config: ScannerConfig,
) -> list[ScanResult]:
    """Demote each over-eager v2 CONFIRM in place; return the same list.

    No-op (returns ``results`` unchanged) on the v1 path and for any non-CONFIRM
    result. A demotion sets ``decision='REJECT'`` + ``validator_verdict``. Pure
    compute — the demotion is surfaced at the pipeline boundary via the summary's
    ``validator_rejects`` count (``_print_summary``/``_post_scan_ops_log``), not
    logged here (arch-pass R3).
    """
    if config.setup_version != "v2":
        return results
    for r in results:
        if not r.is_confirm:
            continue
        if len(r.setup_signal.confluence) < config.v2_min_confluence:
            r.decision = "REJECT"
            r.validator_verdict = REJECT_CONFLUENCE_FLOOR
    return results
