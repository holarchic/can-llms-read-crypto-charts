"""Shadow-mode v2 (T-020.1, SPEC #41) — capture post-validator CONFIRMs (no orders)
and resolve them forward against price for the Acceptance-#10 +R cutover gate.

Capture (`store.save_shadow_signal`, wired via `pipeline.stage4_shadow`) runs at scan
time; `resolver.resolve_signal` walks each captured signal forward, `store` persists and
reloads the re-runnable tape, and `expectancy` emits the Acceptance-#10 verdict
(driven by the `shadow-scan` / `shadow-resolve` CLIs).
"""
