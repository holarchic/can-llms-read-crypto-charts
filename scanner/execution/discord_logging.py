"""Buffered Discord logging for the execution monitor.

Collects ERROR+ records during a monitor run, then flushes once as a
single Discord embed at the end. Avoids per-error HTTP calls and keeps
the Ops channel readable.
"""

import logging
from datetime import datetime, timezone

import requests

MAX_BUFFER = 10
COLOR_ERROR = 0xEF5350


class DiscordBufferHandler(logging.Handler):
    """Buffer ERROR+ log records and flush to Discord as a single embed."""

    def __init__(self, webhook_url: str, level: int = logging.ERROR):
        super().__init__(level)
        self.webhook_url = webhook_url
        self.buffer: list[logging.LogRecord] = []
        self.dropped_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if len(self.buffer) < MAX_BUFFER:
            self.buffer.append(record)
        else:
            self.dropped_count += 1

    def flush_to_discord(self) -> bool:
        """Format buffered records as a Discord embed and POST.

        Named explicitly (not overriding flush()) to avoid accidental
        network calls from logging.shutdown().

        Returns True if sent successfully, False otherwise.
        """
        if not self.buffer or not self.webhook_url:
            return False

        lines = []
        for record in self.buffer:
            ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%H:%M:%S")
            lines.append(f"[{ts}] {record.name}: {record.getMessage()}")

        if self.dropped_count:
            lines.append(f"... and {self.dropped_count} more")

        description = "\n".join(lines)
        embed = {
            "title": f"Monitor Errors ({len(self.buffer)})",
            "description": description,
            "color": COLOR_ERROR,
            "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")},
        }

        try:
            resp = requests.post(
                self.webhook_url,
                json={"embeds": [embed]},
                timeout=10,
            )
            resp.raise_for_status()
            return True
        except Exception:
            return False
        finally:
            self.buffer.clear()
            self.dropped_count = 0


def configure_monitor_logging(webhook_url: str = "") -> DiscordBufferHandler | None:
    """Set up logging for the standalone monitor entry point.

    Mirrors cli.py:57-63 — StreamHandler at INFO, root at WARNING,
    scanner.execution at INFO. Adds a DiscordBufferHandler if webhook provided.

    Returns the Discord handler (for manual flush) or None.
    """
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))

    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.WARNING)

    logging.getLogger("scanner.execution").setLevel(logging.INFO)

    if not webhook_url:
        return None

    discord_handler = DiscordBufferHandler(webhook_url)
    logging.getLogger("scanner.execution").addHandler(discord_handler)
    return discord_handler
