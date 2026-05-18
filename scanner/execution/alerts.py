"""Alert delivery helpers with retry + durable local log.

URGENT alerts (naked position, SL swap failed, emergency close failed) must
survive transient Discord outages: a single 5xx or network blip at the wrong
moment otherwise silently eats the @everyone page. Every URGENT post is
mirrored to data/logs/urgent-alerts.log so an operator can grep after the
fact even if Discord never received it.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger(__name__)

URGENT_LOG_PATH = Path(__file__).resolve().parents[2] / "data" / "logs" / "urgent-alerts.log"


def _append_urgent_log(content: str) -> None:
    """Append content to the durable urgent-alerts log. Never raises."""
    try:
        URGENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with URGENT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {content}\n")
    except Exception as exc:
        log.error("urgent-alerts.log write failed: %s", exc)


def send_urgent(webhook: str, content: str, max_retries: int = 3) -> bool:
    """Post to webhook with retry + local log. Returns True on Discord ack.

    Always writes to urgent-alerts.log regardless of Discord outcome so a
    Discord outage never eats the paper trail.
    """
    _append_urgent_log(content)
    if not webhook:
        return False

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(webhook, json={"content": content}, timeout=10)
            if resp.status_code == 204:
                return True
            log.warning(
                "URGENT alert non-204 (%s) attempt %d/%d",
                resp.status_code, attempt, max_retries,
            )
        except Exception as exc:
            log.warning(
                "URGENT alert attempt %d/%d failed: %s",
                attempt, max_retries, exc,
            )
        if attempt < max_retries:
            time.sleep(2 ** (attempt - 1))  # 1s, 2s
    log.error("URGENT alert exhausted — see urgent-alerts.log for content")
    return False


def post_embed(
    webhook: str | None, embed: dict, label: str, *,
    failure_logger: logging.Logger, failure_level: int = logging.WARNING,
) -> bool:
    """Single owner of the Discord webhook delivery + 204-interpretation
    decision for embed alerts (T-016 Phase 3, SPEC #26 e). Returns True on a
    204 ack, False on no-webhook / non-204 / exception.

    The send failure is logged THROUGH the caller's logger
    (`failure_logger`) at the caller's level (`failure_level`) so the buffered
    `LogRecord.name`, level, and message stem stay byte-identical — the
    `discord_logging` ERROR-buffer keys its operational embed on
    `record.name`, so centralising the log call here would mis-attribute it.
    """
    if not webhook:
        return False
    try:
        resp = requests.post(webhook, json={"embeds": [embed]}, timeout=10)
        return resp.status_code == 204
    except Exception as exc:
        failure_logger.log(failure_level, "%s failed: %s", label, exc)
        return False


def post_content(
    webhook: str | None, content: str, label: str, *,
    failure_logger: logging.Logger, failure_level: int = logging.WARNING,
) -> bool:
    """Single owner of the Discord webhook delivery + 204-interpretation
    decision for plain-content alerts (T-016 Phase 3, SPEC #26 e). Failure
    logging semantics identical to `post_embed`."""
    if not webhook:
        return False
    try:
        resp = requests.post(webhook, json={"content": content}, timeout=10)
        return resp.status_code == 204
    except Exception as exc:
        failure_logger.log(failure_level, "%s failed: %s", label, exc)
        return False
