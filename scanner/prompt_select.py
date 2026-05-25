"""Version-gated prompt selection (T-019 Chunk 7b.3 / 8d).

Single owner of the `SETUP_VERSION`-gated LLM artifacts — the persisted `prompt_version`
and the assembled system prompt (the per-version system prompt + ruleset + scanner-mode
suffix) — so a cutover flips one seam, not scattered `if setup_version` checks. v1 is
frozen at 9.11; v2 = 10.00.

All three taxonomy-bearing surfaces are version-branched here: the system prompt
(`SYSTEM_PROMPT` / `SYSTEM_PROMPT_V2`), the scanner-mode suffix (`SCANNER_MODE_SUFFIX` /
`SCANNER_MODE_SUFFIX_V2`), and the injected ruleset file (`unified-ta-ruleset.md` /
`-v2.md`, from `SETUP_VERSION_REGISTRY`). The v1 constants are untouched — the default
path stays byte-identical.
"""

from dataclasses import dataclass

from .config import SETUP_VERSION_REGISTRY
from .llm import (
    SCANNER_MODE_SUFFIX,
    SCANNER_MODE_SUFFIX_V2,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_V2,
    load_strategy,
)

# setup_version → (system prompt, scanner-mode suffix). The ruleset file comes from
# SETUP_VERSION_REGISTRY; this map owns the version-branched prose. The single place a
# cutover (or a new version) adds its prompt pair.
_PROMPT_CONSTANTS: dict[str, tuple[str, str]] = {
    "v1": (SYSTEM_PROMPT, SCANNER_MODE_SUFFIX),
    "v2": (SYSTEM_PROMPT_V2, SCANNER_MODE_SUFFIX_V2),
}


@dataclass(frozen=True)
class PromptSpec:
    """The version-gated prompt artifacts for one scan: the assembled `system_text` sent
    to the LLM and the `prompt_version` persisted alongside every result so v1/v2 cohorts
    stay separable (Acceptance #8)."""

    prompt_version: str
    system_text: str


def select_prompt(setup_version: str) -> PromptSpec:
    """Resolve the prompt artifacts for ``setup_version`` — the persisted version + ruleset
    file from `config.SETUP_VERSION_REGISTRY`, the system-prompt/scanner-suffix pair from
    `_PROMPT_CONSTANTS`.

    `load_strategy` injects the per-version ruleset into the per-version system prompt, then
    appends the per-version scanner-mode suffix. An unknown version raises `KeyError`
    (fail-closed) — `config` already validates the flag, so a miss here is a programmer error,
    never silently the v1 cohort."""
    prompt_version, ruleset_path = SETUP_VERSION_REGISTRY[setup_version]
    system_prompt, scanner_suffix = _PROMPT_CONSTANTS[setup_version]
    system_text = system_prompt.format(ruleset=load_strategy(ruleset_path)) + scanner_suffix
    return PromptSpec(prompt_version=prompt_version, system_text=system_text)
