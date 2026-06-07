"""T-020.17 (SPEC #54) — shadow-resolve.py source guards.

The hyphenated CLI can't be imported, so its argparse combo-guards are pinned via subprocess. The
guards encode a money-relevant correctness rule: a binance-db tape must NOT be resolved against the
wrong venue (live HL), and the live scanner.db (real-HL captures) must NOT be resolved with the
Binance archive. Both must `ap.error` (exit 2) BEFORE touching any DB or the network."""

import subprocess
import sys
from pathlib import Path

import pytest

_RESOLVE = Path(__file__).resolve().parents[2] / "scripts" / "shadow-resolve.py"

# scripts/shadow-resolve.py is not shipped in the OSS scaffold — skip its CLI contract tests.
pytestmark = pytest.mark.skipif(
    not _RESOLVE.exists(),
    reason="scripts/shadow-resolve.py not shipped in the OSS scaffold",
)


@pytest.mark.parametrize("args, needle", [
    (["--source", "binance-db"], "requires --replay"),        # binance-db without --replay
    (["--replay"], "requires --source"),                       # --replay without an explicit --source
])
def test_resolve_source_guards_exit_2(args, needle):
    proc = subprocess.run(
        [sys.executable, str(_RESOLVE), *args],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 2                                # argparse usage error, before any work
    assert needle in proc.stderr
