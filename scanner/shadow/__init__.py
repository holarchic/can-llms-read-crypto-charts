"""Shadow-mode v2 (T-020.1, SPEC #41) — capture post-validator CONFIRMs (no orders)
and resolve them forward against price for the Acceptance-#10 +R cutover gate.

Capture (`store.save_shadow_signal`, wired via `pipeline.stage4_shadow`) runs at scan
time; the forward resolver + expectancy readout are the next increment.
"""
