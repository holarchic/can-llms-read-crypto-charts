"""3-stage scan pipeline: screen → analyze → alert."""

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from uuid import uuid4

import anthropic

from .alerts import COLOR_ERROR, COLOR_OK, COLOR_WARN, format_alert_text, send_alert, send_ops_log
from .analysis import analyze_for_scanner
from .charts import cleanup_charts, generate_charts_for_signal
from .config import ScannerConfig, get_exchange_meta
from .data import fetch_candles, fetch_candles_batch, fetch_pairs_with_volume, get_max_leverage
from .data_plane_v2 import fetch_multi_tf
from .db import get_connection
from .indicators import build_context
from .models import BoundarySignal, ExecutionRequest, ScanResult
from .scan_queries import daily_cost, is_duplicate
from .scan_store import mark_alerted, save_result, save_scan_universe, save_signals
from .screener import detect_all_setups
from .screener_v2 import Stage1ArtifactsV2, chart_inputs, screen_v2
from .setup_validator import validate_confirms
from .strategy.store import get_strategy


def _select_with_diversity(
    signals: list[BoundarySignal], max_count: int, decay: float = 0.85,
) -> list[BoundarySignal]:
    """Greedy selection with diminishing returns per setup type.

    Scores each candidate as confidence * decay^n where n = number of that
    setup type already selected.  Random shuffle before calling ensures
    confidence only acts as a tie-breaker within the diversity framework
    (confidence alone is anti-predictive).

    Returns selected + remaining so callers that rely on positional slicing
    (e.g. save_signals marks first N as sent_to_llm) keep working.
    """
    if len(signals) <= max_count:
        return signals

    type_counts: dict[str, int] = {}
    selected: list[BoundarySignal] = []
    remaining = list(signals)

    while remaining and len(selected) < max_count:
        best_idx = 0
        best_score = -1.0
        for i, sig in enumerate(remaining):
            n = type_counts.get(sig.setup_type.value, 0)
            score = sig.confidence * (decay ** n)
            if score > best_score:
                best_score = score
                best_idx = i
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        type_counts[chosen.setup_type.value] = type_counts.get(chosen.setup_type.value, 0) + 1

    return selected + remaining


def _screen_v1(
    coins: list[str],
    leverage_map: dict[str, int | None],
    disabled_types: set[str] | None,
    config: ScannerConfig,
    conn,
    verbose: bool,
) -> tuple[list[BoundarySignal], dict]:
    """v1 screener: batch-fetch 4H → build_context → detect_all_setups → lift → one
    non-duplicate signal per coin. The bundle is the 4H candle map (stage-2 reuse)."""
    if verbose:
        print(f"Fetching 4H candles for {len(coins)} pairs...")
    candle_map = fetch_candles_batch(coins, "4h", config.candle_counts["4h"], config)
    if verbose:
        print(f"Fetched {len(candle_map)}/{len(coins)} pairs")

    all_signals: list[BoundarySignal] = []
    for coin in coins:
        df = candle_map.get(coin)
        if df is None or len(df) < 50:
            continue

        try:
            ctx = build_context(df, config.swing_lookback, config.sr_cluster_pct)
            signals = detect_all_setups(
                coin, df, ctx, config.max_setups_per_coin, config.proximity_atr_mult,
                disabled_types=disabled_types,
            )
            # Option C: lift v1 detector output into the version-agnostic spine record
            # (stage-1 scans 4H). The v2 producer targets the same type via screen_v2.
            signals = [BoundarySignal.from_setup_signal(s, setup_tf="4h") for s in signals]

            # Pick best non-duplicate signal for this coin
            for signal in signals:
                signal.max_leverage = leverage_map.get(coin)
                if is_duplicate(coin, signal.setup_type.value, signal.direction.value,
                                config.dedup_hours, conn, exchange=config.exchange):
                    if verbose:
                        print(f"  {coin} {signal.setup_type.value} — skipped (duplicate)")
                    continue
                all_signals.append(signal)
                break  # One signal per coin

        except Exception as e:
            if verbose:
                print(f"  {coin} — error: {e}")

    return all_signals, candle_map


def _btc_daily_structure(config: ScannerConfig, verbose: bool) -> str | None:
    """BTC Daily swing structure for market context (Daily = filter TF for 4H execution).
    Version-independent context metadata persisted alongside every signal."""
    try:
        btc_daily_df = fetch_candles("BTC", "1d", config.candle_counts["1d"], config)
        if btc_daily_df is not None and len(btc_daily_df) >= 50:
            btc_context = build_context(btc_daily_df, config.swing_lookback, config.sr_cluster_pct)
            if verbose:
                print(f"BTC Daily structure: {btc_context.structure.value}")
            return btc_context.structure.value
    except Exception as e:
        if verbose:
            print(f"BTC Daily fetch failed: {e}")
    return None


def stage1_screen(
    config: ScannerConfig,
    coin_filter: str | None = None,
    verbose: bool = True,
    strategy_id: str = "v0",
    testnet_coins: set[str] | None = None,
) -> tuple[list[BoundarySignal], Stage1ArtifactsV2 | dict]:
    """Stage 1: resolve pairs → version-dispatch the screener → select + persist.

    `setup_version` selects the producer: v1 (`_screen_v1`, shuffle + diversity) or v2
    (`screen_v2`, deterministic score order, already one-per-coin). Pairs resolution,
    BTC-structure context, and persistence are shared; the returned bundle is the
    producer's stage-2 handoff: v2 wraps a typed `Stage1ArtifactsV2`; v1 keeps its
    untyped 4H candle_map dict (the v1-side retires at the cutover)."""
    conn = get_connection()

    # Load strategy for disabled_types
    strategy = get_strategy(strategy_id, conn)
    disabled_types = strategy.disabled_types_set if strategy else None

    meta = get_exchange_meta(config.exchange)

    # Fetch pairs
    if coin_filter:
        coin_upper = coin_filter.upper()
        max_lev = get_max_leverage(coin_upper, config.exchange)
        symbol = meta["symbol_template"].format(coin=coin_upper)
        pairs = [{"coin": coin_upper, "symbol": symbol, "max_leverage": max_lev}]
        if verbose:
            print(f"Single coin mode: {coin_upper} on {meta['label']} (max leverage: {max_lev or '?'}x)")
    else:
        pairs = fetch_pairs_with_volume(config)
        if verbose:
            print(f"Found {len(pairs)} {meta['label']} pairs above ${config.min_volume_usd:,.0f} volume")

    if testnet_coins is not None:
        before = len(pairs)
        pairs = [p for p in pairs if p["coin"] in testnet_coins]
        filtered = before - len(pairs)
        if verbose and filtered:
            print(f"  Filtered {filtered} pairs not on testnet ({len(pairs)} remaining)")

    coins = [p["coin"] for p in pairs]
    leverage_map = {p["coin"]: p.get("max_leverage") for p in pairs}

    if config.setup_version == "v2":
        all_signals, bundle = screen_v2(coins, leverage_map, disabled_types, config, conn, verbose)
    else:
        all_signals, bundle = _screen_v1(coins, leverage_map, disabled_types, config, conn, verbose)

    btc_structure = _btc_daily_structure(config, verbose)

    if config.setup_version == "v2":
        # screen_v2 already ordered by score (one-per-coin); no shuffle/diversity.
        capped = all_signals[:config.max_llm_calls_per_scan]
    else:
        # Shuffle for unbiased LLM selection (confidence is anti-predictive — see calibration analysis)
        random.shuffle(all_signals)
        # Diversity penalty: penalize repeated setup types so underrepresented types get LLM slots
        all_signals = _select_with_diversity(all_signals, config.max_llm_calls_per_scan)
        capped = all_signals[:config.max_llm_calls_per_scan]

    # Persist all signals for backtesting (with scoring components + BTC structure)
    batch_id = uuid4().hex[:8]
    timestamp = datetime.now(timezone.utc).isoformat()
    save_scan_universe(batch_id, pairs, timestamp, conn, exchange=config.exchange)
    save_signals(all_signals, batch_id, len(capped), conn, strategy_id=strategy_id,
                 btc_structure=btc_structure, exchange=config.exchange)

    conn.close()

    if verbose:
        print(f"Stage 1 complete: {len(capped)} signals (from {len(all_signals)} pre-cap)")

    return capped, bundle


def _v2_chart_inputs(
    signal: BoundarySignal, prefetched: Stage1ArtifactsV2 | None, config: ScannerConfig,
) -> tuple[dict, dict]:
    """Resolve the coin's stage-1 bundle entry (defensive refetch if absent — the
    producer bundles every emitted coin) and hand it to the producer-owned
    `screener_v2.chart_inputs`, which owns the frame/HTF-EMA selection. Stage 2 holds
    only the bundle-miss I/O recovery, not the bundle's internal shape."""
    candles = prefetched.candles_by_coin.get(signal.coin) if prefetched else None
    if candles is None:
        candles = fetch_multi_tf(signal.coin, config)
    return chart_inputs(candles)


def _analyze_one(
    signal: BoundarySignal,
    config: ScannerConfig,
    prefetched_candles: Stage1ArtifactsV2 | dict | None = None,
) -> ScanResult | None:
    """Fetch multi-TF candles, generate charts, run LLM analysis for one signal."""
    try:
        if config.setup_version == "v2":
            candle_sets, htf_emas = _v2_chart_inputs(signal, prefetched_candles, config)
            chart_paths = generate_charts_for_signal(signal, candle_sets, enrich=True, htf_emas=htf_emas)
        else:
            candle_sets = {}
            for tf in ["1d", "4h", "1h"]:
                if tf == "4h" and prefetched_candles and signal.coin in prefetched_candles:
                    candle_sets[tf] = prefetched_candles[signal.coin]
                else:
                    candle_sets[tf] = fetch_candles(signal.coin, tf, config.candle_counts[tf], config)
            chart_paths = generate_charts_for_signal(signal, candle_sets)

        result = analyze_for_scanner(signal, chart_paths, config)
        return result
    except Exception as e:
        # 529 SDK-retry-exhaustion gets its own structured ops embed; suppress
        # the generic errors alert to avoid duplicate paging (SPEC #37).
        if isinstance(e, anthropic.APIStatusError) and e.status_code == 529:
            print(f"  LLM 529 exhausted for {signal.coin}: {e}")
            send_ops_log(
                "LLM 529 exhaustion",
                {
                    "Coin": signal.coin,
                    "Setup": signal.setup_type.value,
                    "Request ID": e.request_id or "unknown",
                    "Stage": "stage2_analyze",
                },
                COLOR_ERROR,
                config,
            )
            return None
        print(f"  Error analyzing {signal.coin}: {e}")
        from .alerts import send_errors_alert
        send_errors_alert(f"Stage2 LLM failed for {signal.coin}: {type(e).__name__}: {e}", config)
        return None


def stage2_analyze(
    signals: list[BoundarySignal],
    config: ScannerConfig,
    verbose: bool = True,
    strategy_id: str = "v0",
    prefetched_candles: Stage1ArtifactsV2 | dict | None = None,
) -> list[ScanResult]:
    """Stage 2: Parallel LLM analysis — fetch charts + analyze concurrently."""
    conn = get_connection()
    max_workers = config.analysis_max_workers

    # Check cost cap before starting
    cost_today = daily_cost(conn)
    if cost_today >= config.daily_cost_cap:
        if verbose:
            print(f"Daily cost cap reached (${cost_today:.2f} >= ${config.daily_cost_cap:.2f}). Stopping.")
        conn.close()
        return []

    # Budget remaining LLM calls
    est_cost_per = 0.12  # ~$0.12 per analysis
    budget_remaining = config.daily_cost_cap - cost_today
    max_affordable = int(budget_remaining / est_cost_per)
    to_analyze = signals[:min(len(signals), max_affordable)]

    if verbose:
        print(f"\nStage 2: Analyzing {len(to_analyze)} signals "
              f"(budget: ${budget_remaining:.2f} remaining, ~{max_workers} parallel)")

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_analyze_one, signal, config, prefetched_candles): signal
            for signal in to_analyze
        }
        for future in as_completed(futures):
            signal = futures[future]
            result = future.result()
            if result:
                if verbose:
                    decision_str = result.decision or 'PARSE_FAIL'
                    print(f"  {result.coin:>8} {signal.setup_type.value} {signal.direction.value:>5} → "
                          f"{decision_str:>10}  ${result.cost_usd:.4f}")
                    if not result.decision:
                        preview = (result.analysis_text or '')[:200].replace('\n', ' ')
                        print(f"    ⚠ Parse failed. Raw output: {preview}")
                save_result(result, conn, strategy_id=strategy_id, exchange=config.exchange)
                results.append(result)

    conn.close()
    return results


def stage3_alert(
    results: list[ScanResult],
    config: ScannerConfig,
    skip_discord: bool = False,
    verbose: bool = True,
) -> int:
    """Stage 3: Filter CONFIRMs → post to Discord → cleanup charts."""
    conn = get_connection()
    confirms = [r for r in results if r.is_confirm]
    alerted_count = 0

    if verbose:
        print(f"\nStage 3: {len(confirms)} CONFIRMs from {len(results)} analyses")

    for result in confirms:
        if verbose:
            print(format_alert_text(result))

        if not skip_discord and config.discord_ops_webhook_url:
            exchange_label = get_exchange_meta(config.exchange)["label"]
            success = send_alert(result, config, exchange_label=exchange_label)
            if success:
                mark_alerted(result.scan_id, conn)
                alerted_count += 1
                if verbose:
                    print(f"  Discord alert sent for {result.coin}")
                time.sleep(1)  # Rate limit Discord webhooks
            else:
                if verbose:
                    print(f"  Discord alert FAILED for {result.coin}")
        elif verbose and not skip_discord:
            print("  No Discord webhook configured — skipping alert")

    # Cleanup charts
    cleaned = cleanup_charts()
    if verbose and cleaned:
        print(f"\nCleaned up {cleaned} chart files")

    conn.close()
    return alerted_count


def stage4_execute(
    confirms: list[ScanResult],
    scanner_config: ScannerConfig,
    exec_config,
    verbose: bool = True,
):
    """Stage 4: Execute trades for CONFIRMs with valid machine blocks.

    Each CONFIRM attempted independently. Errors caught — NEVER crashes scanner.
    Returns list of ExecutionResult.
    """
    from .execution.executor import ExecutionResult, execute_trade, preflight_check, send_skip_alert

    if not exec_config.enabled:
        return []

    # Preflight: validate exchange connectivity before attempting any trades
    if exec_config.mode == "live":
        issues = preflight_check(exec_config)
        if issues:
            print(f"  Preflight FAILED: {'; '.join(issues)}")
            return []

    conn = get_connection()
    results = []

    for confirm in confirms:
        try:
            # Route conditional entries to pending if feature enabled
            mb = confirm.machine_block
            if (exec_config.conditional_entry_enabled
                    and mb is not None
                    and mb.entry_trigger
                    and mb.entry_trigger_price
                    and mb.invalidation_price):
                from .execution.entry_monitor import save_conditional_entry
                saved = save_conditional_entry(confirm, exec_config, conn)
                direction = mb.direction
                if saved:
                    print(f"  {confirm.coin} {direction}: PENDING conditional entry "
                          f"({mb.entry_trigger} @ {mb.entry_trigger_price})")
                    results.append(ExecutionResult(
                        scan_id=confirm.scan_id,
                        coin=confirm.coin,
                        direction=direction,
                        skip_reason="conditional_pending",
                    ))
                else:
                    # save_conditional_entry returned False (invalid params,
                    # counter-trend R:R, active-pending/open-position dedupe, or
                    # price-sanity reject). The LLM asked for a conditional entry
                    # at a trigger price — falling through to execute_trade() here
                    # would place an immediate market order that bypasses pending
                    # dedupe and double-exposes when the existing pending later
                    # triggers. A rejected conditional entry is a skip, never a
                    # market fill.
                    print(f"  {confirm.coin} {direction}: SKIPPED — conditional_rejected")
                    results.append(ExecutionResult(
                        scan_id=confirm.scan_id,
                        coin=confirm.coin,
                        direction=direction,
                        skip_reason="conditional_rejected",
                    ))
                continue

            sig = confirm.setup_signal
            req = ExecutionRequest(
                scan_id=confirm.scan_id,
                coin=confirm.coin,
                machine_block=confirm.machine_block,
                screener_direction=sig.direction.value if sig else None,
                atr=sig.context.atr if sig else 0.0,
                max_leverage=sig.max_leverage if sig else None,
                staleness_ref=confirm.detected_at or confirm.timestamp,
            )
            er = execute_trade(req, exec_config, scanner_config, conn)
            results.append(er)
            # Always print execution outcomes — critical ops info, never silenced by -q
            if er.executed:
                print(f"  {confirm.coin} {er.direction}: EXECUTED "
                      f"(entry={er.entry_order_id}, sl={er.sl_order_id})")
            elif er.skip_reason:
                print(f"  {confirm.coin} {er.direction}: SKIPPED — {er.skip_reason}"
                      f"{f' ({er.error})' if er.error else ''}")
                send_skip_alert(er, exec_config)
        except Exception as exc:
            print(f"  {confirm.coin}: execution error — {exc}")
            results.append(ExecutionResult(
                scan_id=confirm.scan_id,
                coin=confirm.coin,
                skip_reason="exception",
                error=str(exc),
            ))

    conn.close()
    return results


def _resolve_testnet_coins(cfg, exec_config, exchange_label, verbose):
    """Pre-filter: load testnet markets to avoid wasting candle fetches + LLM calls."""
    testnet_coins = None
    if exec_config and exec_config.mode == "testnet":
        try:
            from .data import get_testnet_coins
            testnet_coins = get_testnet_coins(cfg.exchange)
            if verbose:
                print(f"Testnet filter: {len(testnet_coins)} pairs on {exchange_label} testnet")
        except Exception as exc:
            if verbose:
                print(f"Testnet pair filter failed: {exc} — proceeding unfiltered")
    return testnet_coins


def _report_dry_run(signals, verbose):
    """Stage-1-only dry run: print detected signals, return the dry-run summary."""
    if verbose:
        print(f"\n--- DRY RUN — {len(signals)} signals detected ---")
        for s in signals:
            print(f"  {s.coin:>8}  {s.setup_type.value}  {s.direction.value:>5}  "
                  f"conf={s.confidence:.2f}  level=${s.key_level:,.2f}")
            print(f"           {s.description[:80]}")
    return {"signals": len(signals), "dry_run": True}


def _empty_summary():
    """The summary returned when a scan short-circuits with no signals to analyze."""
    return {"signals": 0, "results": 0, "confirms": 0, "alerted": 0}


def _query_pending_conditional_coins(exec_config) -> set[str]:
    """Pure DB lookup — coins with an active pending conditional entry.

    Connection lifetime preserved from the pre-split inline block:
    `pending_conn` is closed on the success path only — if
    `get_active_pending_entries` raises it is left open. This is pre-existing
    behaviour deliberately NOT hardened here (resource-handling fix is its
    own task).
    """
    from .execution.storage import get_active_pending_entries
    pending_conn = get_connection()
    pending = get_active_pending_entries(pending_conn, exec_config.exchange, exec_config.mode)
    pending_conn.close()
    return {p["coin"] for p in pending}


def _filter_pending_conditional(signals, exec_config, verbose):
    """Skip LLM for coins with an active pending conditional entry.

    Lookup is delegated to `_query_pending_conditional_coins`; this function
    owns the filter + verbose reporting seam.
    """
    pending_coins = _query_pending_conditional_coins(exec_config)
    if pending_coins:
        before = len(signals)
        signals = [s for s in signals if s.coin not in pending_coins]
        skipped = before - len(signals)
        if skipped and verbose:
            print(f"Skipped {skipped} signal(s) — pending conditional entry: "
                  f"{', '.join(sorted(pending_coins))}")
    if not signals:
        if verbose:
            print("\nNo signals remaining after pending filter.")
    return signals


def _dry_execute_preview(confirms, verbose):
    """Dry-execute: preview sizing without exchange connectivity.

    Uses hardcoded balance=$10K, risk=1.0% — intentional for
    credential-less preview. Skips: price fetch, kill switch,
    position limits, BTC vol gate, margin guard.
    """
    if verbose:
        print(f"\nStage 4 (dry-execute): {len(confirms)} CONFIRMs")
    from .execution.executor import validate_execution_params
    from .execution.sizing import calculate_position
    for c in confirms:
        mb = c.machine_block
        if mb is None:
            if verbose:
                print(f"  {c.coin}: SKIPPED — no machine_block")
            continue
        errors = validate_execution_params(mb)
        if errors:
            if verbose:
                print(f"  {c.coin} {mb.direction}: SKIPPED — {'; '.join(errors)}")
            continue
        entry_mid = (mb.entry_low + mb.entry_high) / 2
        pos = calculate_position(mb, 10000, 1.0)
        if pos:
            if verbose:
                print(f"  {c.coin} {mb.direction}: WOULD EXECUTE — "
                      f"{pos.quantity:.6f} units @ ${entry_mid:,.2f}, "
                      f"SL ${mb.stop_loss:,.2f}, TP ${mb.tp1:,.2f}")
        else:
            if verbose:
                print(f"  {c.coin} {mb.direction}: SKIPPED — sizing failed")


def _build_summary(signals, results, exec_results, alerted, elapsed):
    """Aggregate the scan-outcome summary dict.

    `alerted` (stage3 return) and the raw `elapsed` float are threaded
    explicitly — they are run_scan locals the pre-split inline block closed
    over; `elapsed_sec` keeps the rounded form while the verbose print reads
    the raw `elapsed` (see `_print_summary`).
    """
    total_cost = sum(r.cost_usd for r in results)
    n_executed = sum(1 for er in exec_results if er.executed) if exec_results else 0
    # `validator_verdict` is set only by the v2 post-LLM gate (SPEC #33). `rejects`
    # stays LLM-rejects-only; validator demotions partition into `validator_rejects`,
    # added as a CONDITIONAL key (>0 only) so the v1 summary shape is byte-identical.
    n_validator_rejects = sum(1 for r in results if r.validator_verdict)
    summary = {
        "signals": len(signals),
        "results": len(results),
        "confirms": sum(1 for r in results if r.is_confirm),
        "rejects": sum(1 for r in results if r.is_reject and not r.validator_verdict),
        "waits": sum(1 for r in results if r.is_wait),
        "alerted": alerted,
        "executed": n_executed,
        "cost_usd": round(total_cost, 4),
        "elapsed_sec": round(elapsed, 1),
    }
    if n_validator_rejects:
        summary["validator_rejects"] = n_validator_rejects
    return summary


def _print_summary(summary, elapsed):
    """Verbose scan-complete report.

    Uses the RAW `elapsed` float for the duration line — byte-identical with
    the pre-split block, since `f"{round(x,1):.0f}"` differs from `f"{x:.0f}"`
    at rounding boundaries.
    """
    print(f"\n{'=' * 60}")
    print(f"Scan complete in {elapsed:.0f}s")
    print(f"  Signals: {summary['signals']} | Analyzed: {summary['results']}")
    decision_line = f"  CONFIRM: {summary['confirms']} | REJECT: {summary['rejects']} | WAIT: {summary['waits']}"
    if summary.get("validator_rejects", 0):
        decision_line += f" | Validator-rejected: {summary['validator_rejects']}"
    print(decision_line)
    print(f"  Alerted: {summary['alerted']} | Cost: ${summary['cost_usd']:.4f}")


def _post_scan_ops_log(summary, results, exchange_label, cfg):
    """Post the scan-complete ops log to Discord."""
    parse_fails = sum(1 for r in results if not r.decision)
    warn = summary["signals"] == 0 or parse_fails > 0
    ops_fields = {
        "Signals": str(summary["signals"]),
        "Analyzed": str(summary["results"]),
        "CONFIRM": str(summary["confirms"]),
        "REJECT": str(summary["rejects"]),
        "WAIT": str(summary["waits"]),
        "Alerted": str(summary["alerted"]),
        "Cost": f"${summary['cost_usd']:.2f}",
        "Duration": f"{summary['elapsed_sec']:.0f}s",
    }
    if parse_fails:
        ops_fields["Parse Fails"] = str(parse_fails)
    if summary.get("validator_rejects", 0):
        ops_fields["Validator Rejected"] = str(summary["validator_rejects"])
    send_ops_log(f"Scan Complete [{exchange_label}]", ops_fields, COLOR_WARN if warn else COLOR_OK, cfg)


def run_scan(
    config: ScannerConfig | None = None,
    coin: str | None = None,
    dry_run: bool = False,
    skip_discord: bool = False,
    verbose: bool = True,
    strategy_id: str = "v0",
    exec_config=None,
    dry_execute: bool = False,
) -> dict:
    """Full 3-stage scan pipeline — thin dispatcher.

    prep → stage1 → guards → (pending filter) → stage2 → stage3 → stage4 →
    summary. Each step is one call; the accreted stage/policy/presentation
    bodies live in the `_`-helpers above (T-016 Phase 4b, SPEC #28). Returns
    the summary dict.
    """
    cfg = config or ScannerConfig()
    start = datetime.now(timezone.utc)

    exchange_label = get_exchange_meta(cfg.exchange)["label"]
    if verbose:
        print(f"Scanner started at {start.strftime('%Y-%m-%d %H:%M:%S UTC')} [{exchange_label}]")
        print("=" * 60)

    testnet_coins = _resolve_testnet_coins(cfg, exec_config, exchange_label, verbose)

    # Stage 1
    signals, candle_map = stage1_screen(
        cfg, coin_filter=coin, verbose=verbose, strategy_id=strategy_id,
        testnet_coins=testnet_coins,
    )

    if dry_run:
        return _report_dry_run(signals, verbose)

    if not signals:
        if verbose:
            print("\nNo signals to analyze.")
        return _empty_summary()

    if exec_config and exec_config.conditional_entry_enabled:
        signals = _filter_pending_conditional(signals, exec_config, verbose)
        if not signals:
            return _empty_summary()

    # Stage 2
    results = stage2_analyze(signals, cfg, verbose=verbose, strategy_id=strategy_id,
                                 prefetched_candles=candle_map)

    # Post-LLM structural gate (v2 only) — demote over-eager CONFIRMs so they neither
    # alert (stage 3) nor execute (stage 4). No-op on v1 (SPEC #33). The demotion count
    # surfaces in the scan summary (`validator_rejects`).
    results = validate_confirms(results, cfg)

    # Stage 3
    alerted = stage3_alert(results, cfg, skip_discord=skip_discord, verbose=verbose)

    # Stage 4: Execution (if enabled)
    confirms = [r for r in results if r.is_confirm]
    exec_results = []

    if dry_execute and confirms:
        _dry_execute_preview(confirms, verbose)
    elif exec_config and exec_config.enabled and confirms:
        # Always print — execution status is critical ops info
        print(f"\nStage 4: Executing {len(confirms)} CONFIRMs")
        exec_results = stage4_execute(confirms, cfg, exec_config, verbose=verbose)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    summary = _build_summary(signals, results, exec_results, alerted, elapsed)

    if verbose:
        _print_summary(summary, elapsed)

    if not dry_run and not skip_discord:
        _post_scan_ops_log(summary, results, exchange_label, cfg)

    return summary
