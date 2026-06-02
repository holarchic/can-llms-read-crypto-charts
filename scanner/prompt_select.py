"""Version-gated prompt selection (T-019 Chunk 7b.3 / 8d).

Single owner of the `SETUP_VERSION`-gated LLM artifacts — the persisted `prompt_version`
and the assembled system prompt (the per-version system prompt + ruleset + scanner-mode
suffix) — so a cutover flips one seam, not scattered `if setup_version` checks. v1 is
frozen at 9.11; v2 = 10.01 (T-020.4 collapsed the LLM to a REJECT-only veto + dropped the
`setup_quality` self-label; the structural predicate is the gate — SPEC #44).

All three taxonomy-bearing surfaces are version-branched here: the system prompt
(`SYSTEM_PROMPT` / `SYSTEM_PROMPT_V2`), the scanner-mode suffix (`SCANNER_MODE_SUFFIX` /
`SCANNER_MODE_SUFFIX_V2`), and the injected ruleset file (`unified-ta-ruleset.md` /
`-v2.md`, from `SETUP_VERSION_REGISTRY`). The v1 constants are untouched — the default
path stays byte-identical.
"""

from dataclasses import dataclass

from .config import SETUP_VERSION_REGISTRY, SetupVersion
from .llm import (
    SCANNER_MODE_SUFFIX,
    SCANNER_MODE_SUFFIX_V2,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_V2,
    load_strategy,
)


@dataclass(frozen=True)
class _PromptProse:
    """The per-version prose pair: the `system_prompt` template (with a `{ruleset}` slot the
    selected ruleset fills) and the `scanner_suffix` appended after."""

    system_prompt: str
    scanner_suffix: str


# setup_version → prose pair. The per-version PROSE that can't live in
# `config.SETUP_VERSION_REGISTRY` (which holds the persisted version + ruleset path) without a
# config→llm cycle. `config.SETUP_VERSION_REGISTRY` stays authoritative for "which versions exist";
# this map is pinned to it by the assertion below so the two cannot drift.
_PROMPT_PROSE: dict[SetupVersion, _PromptProse] = {
    SetupVersion.V1: _PromptProse(SYSTEM_PROMPT, SCANNER_MODE_SUFFIX),
    SetupVersion.V2: _PromptProse(SYSTEM_PROMPT_V2, SCANNER_MODE_SUFFIX_V2),
}

# Single-source-of-truth guard: every registered version has exactly one prose pair, and vice
# versa. A version added to one map but not the other fails loudly at import — never silently the
# wrong cohort at scan time. Two invariants (explicit `raise`, not `assert`, which `python -O`
# strips, so they hold in every runtime):
#   1. VALUE parity — same key SET (rendered `.value` so a future drift prints `['v1','v2']`).
#   2. KEY TYPE — both maps keyed by `SetupVersion`, never raw `str`. Needed because `SetupVersion`
#      is a str-subclass: a string key and an enum key are set-equal AND hash-equal, so check (1)
#      alone would pass silently if either map regressed to raw strings.
if set(_PROMPT_PROSE) != set(SETUP_VERSION_REGISTRY):
    raise RuntimeError(
        f"_PROMPT_PROSE versions {sorted(v.value for v in _PROMPT_PROSE)} out of sync with "
        f"SETUP_VERSION_REGISTRY {sorted(v.value for v in SETUP_VERSION_REGISTRY)}"
    )
for _name, _m in (("_PROMPT_PROSE", _PROMPT_PROSE), ("SETUP_VERSION_REGISTRY", SETUP_VERSION_REGISTRY)):
    if not all(isinstance(k, SetupVersion) for k in _m):
        raise RuntimeError(f"{_name} keys must all be SetupVersion, not raw str")


@dataclass(frozen=True)
class PromptSpec:
    """The version-gated prompt artifacts for one scan: the assembled `system_text` sent
    to the LLM and the `prompt_version` persisted alongside every result so v1/v2 cohorts
    stay separable (Acceptance #8)."""

    prompt_version: str
    system_text: str


def select_prompt(setup_version: SetupVersion) -> PromptSpec:
    """Resolve the prompt artifacts for ``setup_version`` — the persisted version + ruleset
    file from `config.SETUP_VERSION_REGISTRY`, the system-prompt/scanner-suffix pair from
    `_PROMPT_PROSE`.

    `load_strategy` injects the per-version ruleset into the per-version system prompt, then
    appends the per-version scanner-mode suffix. An unknown version raises `KeyError`
    (fail-closed) — `config` already validates the flag, so a miss here is a programmer error,
    never silently the v1 cohort."""
    artifacts = SETUP_VERSION_REGISTRY[setup_version]
    prose = _PROMPT_PROSE[setup_version]
    system_text = prose.system_prompt.format(ruleset=load_strategy(artifacts.ruleset_path)) + prose.scanner_suffix
    return PromptSpec(prompt_version=artifacts.prompt_version, system_text=system_text)
