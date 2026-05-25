"""Version-gated prompt selection (T-019 Chunk 7b.3).

Single owner of the `SETUP_VERSION`-gated LLM artifacts — the persisted `prompt_version`
and the assembled system prompt (`SYSTEM_PROMPT` + the per-version ruleset +
`SCANNER_MODE_SUFFIX`) — so a cutover flips one seam, not scattered `if setup_version`
checks. v1 is frozen at 9.11; v2 = 10.00.

7b.3 makes only the SELECTION version-aware: the v2 `SYSTEM_PROMPT` / scanner-mode prose
are still the shared v1 prose (rewritten for the v2 snake_case setup vocabulary in
Chunk 8, where the per-version branch will live), so v2 differs today only in the ruleset
file (`unified-ta-ruleset-v2.md`, a shipped stub) and the persisted version string.
"""

from dataclasses import dataclass

from .llm import SCANNER_MODE_SUFFIX, SYSTEM_PROMPT, load_strategy

_PROMPT_VERSION = {"v1": "9.11", "v2": "10.00"}


@dataclass(frozen=True)
class PromptSpec:
    """The version-gated prompt artifacts for one scan: the assembled `system_text` sent
    to the LLM and the `prompt_version` persisted alongside every result so v1/v2 cohorts
    stay separable (Acceptance #8)."""

    prompt_version: str
    system_text: str


def select_prompt(setup_version: str) -> PromptSpec:
    """Resolve the prompt artifacts for ``setup_version``.

    `load_strategy` injects the per-version ruleset; the `SYSTEM_PROMPT` + scanner-mode
    prose are shared until Chunk 8 authors the v2 vocabulary (this is the function that
    branch will live in). Unknown versions fall back to v1 — `config` validates the flag
    upstream, so this is defence-in-depth, not the primary gate."""
    ruleset = load_strategy(setup_version)
    system_text = SYSTEM_PROMPT.format(ruleset=ruleset) + SCANNER_MODE_SUFFIX
    return PromptSpec(
        prompt_version=_PROMPT_VERSION.get(setup_version, _PROMPT_VERSION["v1"]),
        system_text=system_text,
    )
