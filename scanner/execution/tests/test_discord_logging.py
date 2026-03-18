"""Tests for Discord buffer logging handler."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from scanner.execution.discord_logging import (
    MAX_BUFFER,
    DiscordBufferHandler,
    configure_monitor_logging,
)


@pytest.fixture
def handler():
    return DiscordBufferHandler("https://discord.com/api/webhooks/test")


def _emit_record(handler, level=logging.ERROR, name="scanner.execution.monitor", msg="test error"):
    record = logging.LogRecord(
        name=name, level=level, pathname="", lineno=0,
        msg=msg, args=(), exc_info=None,
    )
    handler.emit(record)
    return record


class TestBufferCapture:
    def test_buffer_captures_error(self, handler):
        _emit_record(handler, level=logging.ERROR)
        assert len(handler.buffer) == 1

    def test_buffer_ignores_warning(self, handler):
        # Handler level is ERROR — but emit() doesn't filter (logging framework does).
        # We test that the handler's level is set correctly via the logging system.
        h = DiscordBufferHandler("https://example.com")
        assert h.level == logging.ERROR

    def test_buffer_caps_at_max(self, handler):
        for i in range(MAX_BUFFER + 5):
            _emit_record(handler, msg=f"error {i}")
        assert len(handler.buffer) == MAX_BUFFER
        assert handler.dropped_count == 5


class TestFlush:
    @patch("scanner.execution.discord_logging.requests.post")
    def test_flush_sends_embed(self, mock_post, handler):
        mock_post.return_value = MagicMock(status_code=200)
        _emit_record(handler, msg="sync failed: timeout")
        _emit_record(handler, msg="check failed: missing SL")

        result = handler.flush_to_discord()

        assert result is True
        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]
        embed = payload["embeds"][0]
        assert embed["title"] == "Monitor Errors (2)"
        assert "sync failed: timeout" in embed["description"]
        assert "check failed: missing SL" in embed["description"]
        assert embed["color"] == 0xEF5350
        assert "footer" in embed

    @patch("scanner.execution.discord_logging.requests.post")
    def test_flush_includes_dropped_count(self, mock_post, handler):
        mock_post.return_value = MagicMock(status_code=200)
        for i in range(MAX_BUFFER + 3):
            _emit_record(handler, msg=f"error {i}")

        handler.flush_to_discord()

        payload = mock_post.call_args.kwargs["json"]
        assert "... and 3 more" in payload["embeds"][0]["description"]

    def test_flush_noop_empty(self, handler):
        with patch("scanner.execution.discord_logging.requests.post") as mock_post:
            result = handler.flush_to_discord()
            assert result is False
            mock_post.assert_not_called()

    def test_flush_noop_no_webhook(self):
        h = DiscordBufferHandler("")
        _emit_record(h)
        result = h.flush_to_discord()
        assert result is False

    @patch("scanner.execution.discord_logging.requests.post", side_effect=ConnectionError("boom"))
    def test_flush_swallows_exception(self, mock_post, handler):
        _emit_record(handler, msg="some error")
        result = handler.flush_to_discord()
        assert result is False
        # Buffer cleared even on failure
        assert len(handler.buffer) == 0

    @patch("scanner.execution.discord_logging.requests.post")
    def test_flush_clears_buffer(self, mock_post, handler):
        mock_post.return_value = MagicMock(status_code=200)
        _emit_record(handler, msg="error 1")
        handler.flush_to_discord()
        assert len(handler.buffer) == 0
        assert handler.dropped_count == 0


class TestConfigureMonitorLogging:
    def test_returns_handler_with_webhook(self):
        h = configure_monitor_logging("https://discord.com/api/webhooks/test")
        assert isinstance(h, DiscordBufferHandler)
        # Cleanup — remove handlers we added to avoid polluting other tests
        logging.getLogger().handlers.clear()
        logging.getLogger("scanner.execution").handlers.clear()

    def test_returns_none_without_webhook(self):
        result = configure_monitor_logging("")
        assert result is None
        logging.getLogger().handlers.clear()

    def test_configures_stream_handler(self):
        configure_monitor_logging("")
        root = logging.getLogger()
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) >= 1
        # Cleanup
        root.handlers.clear()


class TestMonitorCrashWrapper:
    """Tests for crash handling in _cmd_monitor."""

    @patch("scanner.execution.executor._send_errors_alert")
    @patch("scanner.execution.monitor.sync_order_status", side_effect=RuntimeError("exchange down"))
    def test_monitor_crash_sends_ops_alert(self, mock_sync, mock_alert):
        from scanner.execution.__main__ import _cmd_monitor
        from scanner.execution.config import ExecutionConfig

        config = ExecutionConfig(enabled=False, mode="testnet")
        _cmd_monitor(config)  # should not raise
        mock_alert.assert_called_once()
        msg = mock_alert.call_args[0][0]
        assert "Monitor CRASHED" in msg
        assert "exchange down" in msg

    @patch("scanner.execution.executor._send_errors_alert")
    @patch("scanner.execution.monitor.sync_order_status", side_effect=RuntimeError("boom"))
    def test_monitor_crash_still_flushes(self, mock_sync, mock_alert):
        from scanner.execution.__main__ import _cmd_monitor
        from scanner.execution.config import ExecutionConfig

        config = ExecutionConfig(enabled=False, mode="testnet")
        mock_handler = MagicMock()
        _cmd_monitor(config, discord_handler=mock_handler)
        mock_handler.flush_to_discord.assert_called_once()
