"""3-stage scan pipeline: screen → analyze → alert."""

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from uuid import uuid4

from .alerts import COLOR_OK, COLOR_WARN, format_alert_text, send_alert, send_ops_log
from .analysis import PROMPT_VERSION, analyze_for_scanner
from .backtest.candles import cache_candles
from .charts import cleanup_charts, generate_charts_for_signal
from .config import ScannerConfig, get_exchange_meta
from .data import fetch_candles, fetch_candles_batch, fetch_pairs_with_volume, get_max_leverage
from .indicators import build_context
from .models import ScanResult, SetupSignal
from .screener import detect_all_setups
from .storage import daily_cost, get_connection, is_duplicate, mark_alerted, save_result, save_scan_universe, save_signals
from .strategy.store import get_strategy


def _select_with_diversity(
    signals: list[SetupSignal], max_count: int, decay: float = 0.85,
) -> list[SetupSignal]:
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
    selected: list[SetupSignal] = []
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


def _collect_candles(
    coins: list[str],
    config: ScannerConfig,
    verbose: bool = True,
) -> None:
    """Collect candles across all configured exchanges and timeframes.

    Runs exchanges in parallel via ThreadPoolExecutor — each gets its own
    DB connection (SQLite WAL supports concurrent writers).
    Skips (primary exchange, 4H) since that's already fetched and cached by stage1.
    Errors per exchange are caught and logged — one failing doesn't block the rest.
    """
    def _collect_one_exchange(eid: str) -> None:
        label = get_exchange_meta(eid)["label"]
        ex_conn = get_connection()
        for tf in config.collection_timeframes:
            if eid == config.exchange and tf == "4h":
                continue
            count = config.candle_counts.get(tf, 200)
            if verbose:
                print(f"Collecting {tf} candles from {label}...")
            try:
                tf_map = fetch_candles_batch(
                    coins, tf, count, config, exchange_id=eid,
                )
                tf_cached = 0
                for coin_name, df in tf_map.items():
                    if df is not None and not df.empty:
                        tf_cached += cache_candles(coin_name, tf, df, ex_conn, exchange=eid)
                if verbose:
                    print(f"  {label} {tf}: {len(tf_map)} pairs, {tf_cached} rows cached")
            except Exception as e:
                if verbose:
                    print(f"  {label} {tf}: failed — {e}")
        ex_conn.close()

    if not config.candle_exchanges:
        return
    with ThreadPoolExecutor(max_workers=len(config.candle_exchanges)) as executor:
        list(executor.map(_collect_one_exchange, config.candle_exchanges))


def stage1_screen(
    config: ScannerConfig,
    coin_filter: str | None = None,
    verbose: bool = True,
    strategy_id: str = "v0",
) -> list[SetupSignal]:
    """Stage 1: Fetch pairs → batch fetch 4H candles → run screener → dedup."""
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

    # Batch fetch all 4H candles in parallel
    coins = [p["coin"] for p in pairs]
    if verbose:
        print(f"Fetching 4H candles for {len(coins)} pairs...")
    candle_map = fetch_candles_batch(coins, "4h", config.candle_counts["4h"], config)
    if verbose:
        print(f"Fetched {len(candle_map)}/{len(coins)} pairs")

    # Cache 4H candles for replay (primary exchange — already fetched above)
    cached = 0
    for coin, df in candle_map.items():
        if df is not None and not df.empty:
            cached += cache_candles(coin, "4h", df, conn, exchange=config.exchange)
    if verbose and cached:
        print(f"Cached {cached} 4H candle rows")

    # Collect candles across all configured exchanges and timeframes (data pipeline)
    if not coin_filter:
        _collect_candles(coins, config, verbose)

    # Build leverage lookup from pairs data
    leverage_map = {p["coin"]: p.get("max_leverage") for p in pairs}

    # Screen each pair — one signal per coin (highest confidence)
    all_signals = []
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

    # Compute BTC 4H structure for market context (used in LLM prompts + stored per signal)
    btc_structure = None
    btc_df = candle_map.get("BTC")
    if btc_df is not None and len(btc_df) >= 50:
        btc_ctx = build_context(btc_df, config.swing_lookback, config.sr_cluster_pct)
        btc_structure = btc_ctx.structure.value
        if verbose:
            print(f"BTC 4H structure: {btc_structure}")

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

    return capped, btc_structure, candle_map


def _analyze_one(
    signal: SetupSignal,
    config: ScannerConfig,
    btc_structure: str | None = None,
    prefetched_candles: dict | None = None,
) -> ScanResult | None:
    """Fetch multi-TF candles, generate charts, run LLM analysis for one signal."""
    try:
        candle_sets = {}
        for tf in ["1d", "4h", "1h"]:
            if tf == "4h" and prefetched_candles and signal.coin in prefetched_candles:
                candle_sets[tf] = prefetched_candles[signal.coin]
            else:
                candle_sets[tf] = fetch_candles(signal.coin, tf, config.candle_counts[tf], config)

        # Cache all timeframes for experiment replay (4H already cached in stage1)
        conn = get_connection()
        for tf in ["1d", "1h"]:
            if tf in candle_sets:
                cache_candles(signal.coin, tf, candle_sets[tf], conn, exchange=config.exchange)
        conn.close()

        chart_paths = generate_charts_for_signal(signal, candle_sets)
        result = analyze_for_scanner(signal, chart_paths, config, btc_structure)
        return result
    except Exception as e:
        print(f"  Error analyzing {signal.coin}: {e}")
        return None


def stage2_analyze(
    signals: list[SetupSignal],
    config: ScannerConfig,
    verbose: bool = True,
    strategy_id: str = "v0",
    btc_structure: str | None = None,
    prefetched_candles: dict | None = None,
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
            executor.submit(_analyze_one, signal, config, btc_structure, prefetched_candles): signal
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
                save_result(result, conn, strategy_id=strategy_id, exchange=config.exchange,
                           prompt_version=PROMPT_VERSION)
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
    confirms = [r for r in results if r.decision == "CONFIRM"]
    alerted_count = 0

    if verbose:
        print(f"\nStage 3: {len(confirms)} CONFIRMs from {len(results)} analyses")

    for result in confirms:
        if verbose:
            print(format_alert_text(result))

        if not skip_discord and config.discord_webhook_url:
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
    from .execution.executor import ExecutionResult, execute_trade

    if not exec_config.enabled:
        return []

    conn = get_connection()
    results = []

    for confirm in confirms:
        try:
            er = execute_trade(confirm, exec_config, scanner_config, conn)
            results.append(er)
            # Always print execution outcomes — critical ops info, never silenced by -q
            if er.executed:
                print(f"  {confirm.coin} {er.direction}: EXECUTED "
                      f"(entry={er.entry_order_id}, sl={er.sl_order_id})")
            elif er.skip_reason:
                print(f"  {confirm.coin} {er.direction}: SKIPPED — {er.skip_reason}"
                      f"{f' ({er.error})' if er.error else ''}")
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
    """Full 3-stage scan pipeline. Returns summary dict."""
    cfg = config or ScannerConfig()
    start = datetime.now(timezone.utc)

    exchange_label = get_exchange_meta(cfg.exchange)["label"]
    if verbose:
        print(f"Scanner started at {start.strftime('%Y-%m-%d %H:%M:%S UTC')} [{exchange_label}]")
        print("=" * 60)

    # Stage 1
    signals, btc_structure, candle_map = stage1_screen(cfg, coin_filter=coin, verbose=verbose, strategy_id=strategy_id)

    if dry_run:
        if verbose:
            print(f"\n--- DRY RUN — {len(signals)} signals detected ---")
            for s in signals:
                print(f"  {s.coin:>8}  {s.setup_type.value}  {s.direction.value:>5}  "
                      f"conf={s.confidence:.2f}  level=${s.key_level:,.2f}")
                print(f"           {s.description[:80]}")
        return {"signals": len(signals), "dry_run": True}

    if not signals:
        if verbose:
            print("\nNo signals to analyze.")
        return {"signals": 0, "results": 0, "confirms": 0, "alerted": 0}

    # Stage 2
    results = stage2_analyze(signals, cfg, verbose=verbose, strategy_id=strategy_id,
                                 btc_structure=btc_structure, prefetched_candles=candle_map)

    # Stage 3
    alerted = stage3_alert(results, cfg, skip_discord=skip_discord, verbose=verbose)

    # Stage 4: Execution (if enabled)
    confirms = [r for r in results if r.decision == "CONFIRM"]
    exec_results = []

    if dry_execute and confirms:
        if verbose:
            print(f"\nStage 4 (dry-execute): {len(confirms)} CONFIRMs")
        from .execution.executor import validate_execution_params
        from .execution.sizing import calculate_position
        for c in confirms:
            mb = c.machine_block
            if not mb:
                if verbose:
                    print(f"  {c.coin}: SKIPPED — no machine_block")
                continue
            errors = validate_execution_params(mb)
            if errors:
                if verbose:
                    print(f"  {c.coin} {mb.get('direction','?')}: SKIPPED — {'; '.join(errors)}")
                continue
            entry_mid = (mb.get("entry_low", 0) + mb.get("entry_high", 0)) / 2
            pos = calculate_position(mb, 10000, 1.0)
            if pos:
                if verbose:
                    print(f"  {c.coin} {mb.get('direction','?')}: WOULD EXECUTE — "
                          f"{pos.quantity:.6f} units @ ${entry_mid:,.2f}, "
                          f"SL ${mb.get('stop_loss',0):,.2f}, TP ${mb.get('tp1',0):,.2f}")
            else:
                if verbose:
                    print(f"  {c.coin} {mb.get('direction','?')}: SKIPPED — sizing failed")

    elif exec_config and exec_config.enabled and confirms:
        # Always print — execution status is critical ops info
        print(f"\nStage 4: Executing {len(confirms)} CONFIRMs")
        exec_results = stage4_execute(confirms, cfg, exec_config, verbose=verbose)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    total_cost = sum(r.cost_usd for r in results)

    n_executed = sum(1 for er in exec_results if er.executed) if exec_results else 0

    summary = {
        "signals": len(signals),
        "results": len(results),
        "confirms": sum(1 for r in results if r.decision == "CONFIRM"),
        "rejects": sum(1 for r in results if r.decision == "REJECT"),
        "waits": sum(1 for r in results if r.decision == "WAIT"),
        "alerted": alerted,
        "executed": n_executed,
        "cost_usd": round(total_cost, 4),
        "elapsed_sec": round(elapsed, 1),
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Scan complete in {elapsed:.0f}s")
        print(f"  Signals: {summary['signals']} | Analyzed: {summary['results']}")
        print(f"  CONFIRM: {summary['confirms']} | REJECT: {summary['rejects']} | WAIT: {summary['waits']}")
        print(f"  Alerted: {summary['alerted']} | Cost: ${summary['cost_usd']:.4f}")

    # Post ops log to Discord
    if not dry_run and not skip_discord:
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
        send_ops_log(f"Scan Complete [{exchange_label}]", ops_fields, COLOR_WARN if warn else COLOR_OK, cfg)

    return summary
