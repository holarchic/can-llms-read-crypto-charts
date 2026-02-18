"""3-stage scan pipeline: screen → analyze → alert."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import ScannerConfig, CHARTS_DIR
from .data import fetch_pairs_with_volume, fetch_candles, fetch_candles_batch, get_max_leverage
from .indicators import build_context
from .screener import detect_all_setups
from .charts import generate_charts_for_signal, cleanup_charts
from .analysis import analyze_for_scanner
from .storage import get_connection, save_result, save_signals, is_duplicate, mark_alerted, daily_cost
from .alerts import send_alert, format_alert_text
from .models import SetupSignal, ScanResult
from .backtest.candles import cache_candles
from .strategy.store import get_strategy


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

    # Fetch pairs
    if coin_filter:
        coin_upper = coin_filter.upper()
        max_lev = get_max_leverage(coin_upper)
        pairs = [{"coin": coin_upper, "symbol": f"{coin_upper}/USDC:USDC", "max_leverage": max_lev}]
        if verbose:
            print(f"Single coin mode: {coin_upper} (max leverage: {max_lev or '?'}x)")
    else:
        pairs = fetch_pairs_with_volume(config)
        if verbose:
            print(f"Found {len(pairs)} pairs above ${config.min_volume_usd:,.0f} volume")

    # Batch fetch all 4H candles in parallel
    coins = [p["coin"] for p in pairs]
    if verbose:
        print(f"Fetching 4H candles for {len(coins)} pairs...")
    candle_map = fetch_candles_batch(coins, "4h", config.candle_counts["4h"], config)
    if verbose:
        print(f"Fetched {len(candle_map)}/{len(coins)} pairs")

    # Cache 4H candles for replay
    cached = 0
    for coin, df in candle_map.items():
        if df is not None and not df.empty:
            cached += cache_candles(coin, "4h", df, conn)
    if verbose and cached:
        print(f"Cached {cached} 4H candle rows")

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
                                config.dedup_hours, conn):
                    if verbose:
                        print(f"  {coin} {signal.setup_type.value} — skipped (duplicate)")
                    continue
                all_signals.append(signal)
                break  # One signal per coin

        except Exception as e:
            if verbose:
                print(f"  {coin} — error: {e}")

    # Sort by confidence, cap at max LLM calls
    all_signals.sort(key=lambda s: s.confidence, reverse=True)
    capped = all_signals[:config.max_llm_calls_per_scan]

    # Persist all signals for backtesting
    batch_id = uuid4().hex[:8]
    save_signals(all_signals, batch_id, len(capped), conn, strategy_id=strategy_id)

    # Compute BTC 4H structure for market context (used in LLM prompts)
    btc_structure = None
    btc_df = candle_map.get("BTC")
    if btc_df is not None and len(btc_df) >= 50:
        btc_ctx = build_context(btc_df, config.swing_lookback, config.sr_cluster_pct)
        btc_structure = btc_ctx.structure.value
        if verbose:
            print(f"BTC 4H structure: {btc_structure}")

    conn.close()

    if verbose:
        print(f"Stage 1 complete: {len(capped)} signals (from {len(all_signals)} pre-cap)")

    return capped, btc_structure


def _analyze_one(
    signal: SetupSignal,
    config: ScannerConfig,
    btc_structure: str | None = None,
) -> ScanResult | None:
    """Fetch multi-TF candles, generate charts, run LLM analysis for one signal."""
    try:
        candle_sets = {}
        for tf in ["1d", "4h", "1h"]:
            candle_sets[tf] = fetch_candles(signal.coin, tf, config.candle_counts[tf], config)

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
    max_workers: int = 4,
    strategy_id: str = "v0",
    btc_structure: str | None = None,
) -> list[ScanResult]:
    """Stage 2: Parallel LLM analysis — fetch charts + analyze concurrently."""
    conn = get_connection()

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
            executor.submit(_analyze_one, signal, config, btc_structure): signal
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
                save_result(result, conn, strategy_id=strategy_id)
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
            success = send_alert(result, config)
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
            print(f"  No Discord webhook configured — skipping alert")

    # Cleanup charts
    cleaned = cleanup_charts()
    if verbose and cleaned:
        print(f"\nCleaned up {cleaned} chart files")

    conn.close()
    return alerted_count


def run_scan(
    config: ScannerConfig | None = None,
    coin: str | None = None,
    dry_run: bool = False,
    skip_discord: bool = False,
    verbose: bool = True,
    strategy_id: str = "v0",
) -> dict:
    """Full 3-stage scan pipeline. Returns summary dict."""
    cfg = config or ScannerConfig()
    start = datetime.now(timezone.utc)

    if verbose:
        print(f"Scanner started at {start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("=" * 60)

    # Stage 1
    signals, btc_structure = stage1_screen(cfg, coin_filter=coin, verbose=verbose, strategy_id=strategy_id)

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
    results = stage2_analyze(signals, cfg, verbose=verbose, strategy_id=strategy_id, btc_structure=btc_structure)

    # Stage 3
    alerted = stage3_alert(results, cfg, skip_discord=skip_discord, verbose=verbose)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    total_cost = sum(r.cost_usd for r in results)

    summary = {
        "signals": len(signals),
        "results": len(results),
        "confirms": sum(1 for r in results if r.decision == "CONFIRM"),
        "rejects": sum(1 for r in results if r.decision == "REJECT"),
        "waits": sum(1 for r in results if r.decision == "WAIT"),
        "alerted": alerted,
        "cost_usd": round(total_cost, 4),
        "elapsed_sec": round(elapsed, 1),
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Scan complete in {elapsed:.0f}s")
        print(f"  Signals: {summary['signals']} | Analyzed: {summary['results']}")
        print(f"  CONFIRM: {summary['confirms']} | REJECT: {summary['rejects']} | WAIT: {summary['waits']}")
        print(f"  Alerted: {summary['alerted']} | Cost: ${summary['cost_usd']:.4f}")

    return summary
