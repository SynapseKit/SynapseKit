from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from synapsekit.agents.tools.google_calendar import GoogleCalendarTool


class TestGoogleCalendarTool:
    @pytest.mark.asyncio
    async def test_list_events(self):
        events_service = MagicMock()
        events_service.list.return_value.execute.return_value = {
            "items": [
                {
                    "id": "evt1",
                    "summary": "Standup",
                    "start": {"dateTime": "2026-03-24T09:00:00+05:30"},
                    "end": {"dateTime": "2026-03-24T09:15:00+05:30"},
                }
            ]
        }
        service = MagicMock()
        service.events.return_value = events_service

        google_auth_mod = MagicMock()
        google_auth_mod.default.return_value = (MagicMock(), None)
        discovery_mod = MagicMock()
        discovery_mod.build.return_value = service
        google_mod = MagicMock()
        google_mod.auth = google_auth_mod

        with patch.dict(
            "sys.modules",
            {
                "google": google_mod,
                "google.auth": google_auth_mod,
                "googleapiclient": MagicMock(discovery=discovery_mod),
                "googleapiclient.discovery": discovery_mod,
            },
        ):
            tool = GoogleCalendarTool()
            result = await tool.run(action="list_events", calendar_id="primary")

        assert not result.is_error
        assert "Standup" in result.output
        discovery_mod.build.assert_called_once_with(
            "calendar",
            "v3",
            credentials=google_auth_mod.default.return_value[0],
            cache_discovery=False,
        )
        assert "timeMin" not in events_service.list.call_args.kwargs
        assert "timeMax" not in events_service.list.call_args.kwargs

    @pytest.mark.asyncio
    async def test_create_event(self):
        events_service = MagicMock()
        events_service.insert.return_value.execute.return_value = {
            "summary": "Planning",
            "htmlLink": "https://example.com/event",
        }
        service = MagicMock()
        service.events.return_value = events_service

        google_auth_mod = MagicMock()
        google_auth_mod.default.return_value = (MagicMock(), None)
        discovery_mod = MagicMock()
        discovery_mod.build.return_value = service
        google_mod = MagicMock()
        google_mod.auth = google_auth_mod

        with patch.dict(
            "sys.modules",
            {
                "google": google_mod,
                "google.auth": google_auth_mod,
                "googleapiclient": MagicMock(discovery=discovery_mod),
                "googleapiclient.discovery": discovery_mod,
            },
        ):
            tool = GoogleCalendarTool()
            result = await tool.run(
                action="create_event",
                summary="Planning",
                start="2026-03-24T10:00:00+05:30",
                end="2026-03-24T11:00:00+05:30",
            )

        assert not result.is_error
        assert "Created event: Planning" in result.output
        body = events_service.insert.call_args.kwargs["body"]
        assert body["summary"] == "Planning"
        assert body["start"]["dateTime"] == "2026-03-24T10:00:00+05:30"

    @pytest.mark.asyncio
    async def test_delete_event(self):
        events_service = MagicMock()
        events_service.delete.return_value.execute.return_value = {}
        service = MagicMock()
        service.events.return_value = events_service

        google_auth_mod = MagicMock()
        google_auth_mod.default.return_value = (MagicMock(), None)
        discovery_mod = MagicMock()
        discovery_mod.build.return_value = service
        google_mod = MagicMock()
        google_mod.auth = google_auth_mod

        with patch.dict(
            "sys.modules",
            {
                "google": google_mod,
                "google.auth": google_auth_mod,
                "googleapiclient": MagicMock(discovery=discovery_mod),
                "googleapiclient.discovery": discovery_mod,
            },
        ):
            tool = GoogleCalendarTool()
            result = await tool.run(action="delete_event", event_id="evt-123")

        assert not result.is_error
        assert "Deleted event evt-123" in result.output
        events_service.delete.assert_called_once_with(calendarId="primary", eventId="evt-123")
