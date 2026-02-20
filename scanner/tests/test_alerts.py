"""Tests for scanner.alerts — ops logging and report embeds."""

from unittest.mock import patch, MagicMock

from scanner.alerts import send_ops_log, send_report_embeds, COLOR_OK, COLOR_WARN, COLOR_ERROR
from scanner.config import ScannerConfig


def _cfg(ops_url="https://ops.hook", reports_url="", main_url=""):
    return ScannerConfig(
        anthropic_api_key="",
        discord_webhook_url=main_url,
        discord_reports_webhook_url=reports_url,
        discord_ops_webhook_url=ops_url,
    )


class TestSendOpsLog:
    @patch("scanner.alerts.requests.post")
    def test_posts_embed_with_fields(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        result = send_ops_log(
            "Scan Complete",
            {"Signals": "14", "Cost": "$1.68"},
            COLOR_OK,
            _cfg(),
        )

        assert result is True
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        embed = payload["embeds"][0]
        assert embed["title"] == "Scan Complete"
        assert embed["color"] == COLOR_OK
        assert len(embed["fields"]) == 2
        assert embed["fields"][0]["name"] == "Signals"
        assert embed["fields"][0]["inline"] is True
        assert "UTC" in embed["footer"]["text"]

    @patch("scanner.alerts.requests.post")
    def test_falls_back_to_reports_webhook(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        send_ops_log("Test", {"k": "v"}, config=_cfg(ops_url="", reports_url="https://reports.hook"))

        url = mock_post.call_args[0][0]
        assert url == "https://reports.hook"

    def test_no_webhook_returns_false(self):
        result = send_ops_log("Test", {"k": "v"}, config=_cfg(ops_url="", reports_url=""))
        assert result is False

    @patch("scanner.alerts.requests.post", side_effect=Exception("network error"))
    def test_swallows_exceptions(self, mock_post):
        result = send_ops_log("Test", {"k": "v"}, config=_cfg())
        assert result is False

    @patch("scanner.alerts.requests.post")
    def test_error_color(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        send_ops_log("Scan FAILED", {"Error": "boom"}, COLOR_ERROR, _cfg())

        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert embed["color"] == COLOR_ERROR


class TestSendReportEmbeds:
    @patch("scanner.alerts.requests.post")
    def test_posts_embeds(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        embeds = [{"title": "Test", "color": 0x26A69A, "fields": []}]
        result = send_report_embeds(embeds, _cfg(ops_url="", reports_url="https://reports.hook"))

        assert result is True
        payload = mock_post.call_args[1]["json"]
        assert payload["embeds"] == embeds

    def test_no_webhook_returns_false(self):
        result = send_report_embeds([{"title": "X"}], _cfg(ops_url="", reports_url="", main_url=""))
        assert result is False

    @patch("scanner.alerts.requests.post")
    def test_falls_back_to_main_webhook(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        send_report_embeds([{"title": "X"}], _cfg(ops_url="", reports_url="", main_url="https://main.hook"))

        url = mock_post.call_args[0][0]
        assert url == "https://main.hook"
