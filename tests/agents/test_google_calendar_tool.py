from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from synapsekit.agents.tools.google_calendar import GoogleCalendarTool


def _make_google_mocks(service: MagicMock) -> dict[str, MagicMock]:
    """Build the sys.modules dict needed to mock google-api-python-client."""
    google_auth_mod = MagicMock()
    google_auth_mod.default.return_value = (MagicMock(), None)
    discovery_mod = MagicMock()
    discovery_mod.build.return_value = service
    google_mod = MagicMock()
    google_mod.auth = google_auth_mod
    return {
        "google": google_mod,
        "google.auth": google_auth_mod,
        "googleapiclient": MagicMock(discovery=discovery_mod),
        "googleapiclient.discovery": discovery_mod,
    }


def _make_service(events_service: MagicMock) -> MagicMock:
    service = MagicMock()
    service.events.return_value = events_service
    return service


class TestGoogleCalendarTool:
    # ------------------------------------------------------------------
    # list_events
    # ------------------------------------------------------------------

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
        service = _make_service(events_service)

        with patch.dict("sys.modules", _make_google_mocks(service)):
            tool = GoogleCalendarTool()
            result = await tool.run(action="list_events", calendar_id="primary")

        assert not result.is_error
        assert "Standup" in result.output
        assert "evt1" in result.output
        assert "timeMin" not in events_service.list.call_args.kwargs
        assert "timeMax" not in events_service.list.call_args.kwargs

    @pytest.mark.asyncio
    async def test_list_events_empty(self):
        events_service = MagicMock()
        events_service.list.return_value.execute.return_value = {"items": []}
        service = _make_service(events_service)

        with patch.dict("sys.modules", _make_google_mocks(service)):
            tool = GoogleCalendarTool()
            result = await tool.run(action="list_events")

        assert not result.is_error
        assert "No events found" in result.output

    @pytest.mark.asyncio
    async def test_list_events_with_time_bounds(self):
        events_service = MagicMock()
        events_service.list.return_value.execute.return_value = {"items": []}
        service = _make_service(events_service)

        with patch.dict("sys.modules", _make_google_mocks(service)):
            tool = GoogleCalendarTool()
            await tool.run(
                action="list_events",
                time_min="2026-03-01T00:00:00Z",
                time_max="2026-03-31T23:59:59Z",
            )

        call_kwargs = events_service.list.call_args.kwargs
        assert call_kwargs["timeMin"] == "2026-03-01T00:00:00Z"
        assert call_kwargs["timeMax"] == "2026-03-31T23:59:59Z"

    @pytest.mark.asyncio
    async def test_list_events_all_day(self):
        """Events with 'date' instead of 'dateTime' should render correctly."""
        events_service = MagicMock()
        events_service.list.return_value.execute.return_value = {
            "items": [
                {
                    "id": "allday1",
                    "summary": "Holiday",
                    "start": {"date": "2026-12-25"},
                    "end": {"date": "2026-12-26"},
                }
            ]
        }
        service = _make_service(events_service)

        with patch.dict("sys.modules", _make_google_mocks(service)):
            tool = GoogleCalendarTool()
            result = await tool.run(action="list_events")

        assert "Holiday" in result.output
        assert "2026-12-25" in result.output

    @pytest.mark.asyncio
    async def test_list_events_untitled(self):
        """Events without a summary should show 'Untitled'."""
        events_service = MagicMock()
        events_service.list.return_value.execute.return_value = {
            "items": [
                {
                    "id": "evt2",
                    "start": {"dateTime": "2026-01-01T00:00:00Z"},
                    "end": {"dateTime": "2026-01-01T01:00:00Z"},
                }
            ]
        }
        service = _make_service(events_service)

        with patch.dict("sys.modules", _make_google_mocks(service)):
            tool = GoogleCalendarTool()
            result = await tool.run(action="list_events")

        assert "Untitled" in result.output

    @pytest.mark.asyncio
    async def test_list_events_custom_max_results(self):
        events_service = MagicMock()
        events_service.list.return_value.execute.return_value = {"items": []}
        service = _make_service(events_service)

        with patch.dict("sys.modules", _make_google_mocks(service)):
            tool = GoogleCalendarTool()
            await tool.run(action="list_events", max_results=5)

        assert events_service.list.call_args.kwargs["maxResults"] == 5

    # ------------------------------------------------------------------
    # create_event
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_create_event(self):
        events_service = MagicMock()
        events_service.insert.return_value.execute.return_value = {
            "summary": "Planning",
            "htmlLink": "https://example.com/event",
        }
        service = _make_service(events_service)

        with patch.dict("sys.modules", _make_google_mocks(service)):
            tool = GoogleCalendarTool()
            result = await tool.run(
                action="create_event",
                summary="Planning",
                start="2026-03-24T10:00:00+05:30",
                end="2026-03-24T11:00:00+05:30",
            )

        assert not result.is_error
        assert "Created event: Planning" in result.output
        assert "https://example.com/event" in result.output
        body = events_service.insert.call_args.kwargs["body"]
        assert body["summary"] == "Planning"
        assert body["start"]["dateTime"] == "2026-03-24T10:00:00+05:30"

    @pytest.mark.asyncio
    async def test_create_event_with_description_and_timezone(self):
        events_service = MagicMock()
        events_service.insert.return_value.execute.return_value = {
            "summary": "Meeting",
            "htmlLink": "https://example.com/evt",
        }
        service = _make_service(events_service)

        with patch.dict("sys.modules", _make_google_mocks(service)):
            tool = GoogleCalendarTool()
            result = await tool.run(
                action="create_event",
                summary="Meeting",
                description="Weekly sync",
                start="2026-03-24T14:00:00Z",
                end="2026-03-24T15:00:00Z",
                timezone="America/New_York",
            )

        assert not result.is_error
        body = events_service.insert.call_args.kwargs["body"]
        assert body["description"] == "Weekly sync"
        assert body["start"]["timeZone"] == "America/New_York"

    @pytest.mark.asyncio
    async def test_create_event_missing_summary(self):
        tool = GoogleCalendarTool()
        result = await tool.run(
            action="create_event",
            start="2026-03-24T10:00:00Z",
            end="2026-03-24T11:00:00Z",
        )

        assert result.is_error
        assert "summary" in result.error

    @pytest.mark.asyncio
    async def test_create_event_missing_start(self):
        tool = GoogleCalendarTool()
        result = await tool.run(
            action="create_event",
            summary="Test",
            end="2026-03-24T11:00:00Z",
        )

        assert result.is_error
        assert "start" in result.error

    @pytest.mark.asyncio
    async def test_create_event_missing_end(self):
        tool = GoogleCalendarTool()
        result = await tool.run(
            action="create_event",
            summary="Test",
            start="2026-03-24T10:00:00Z",
        )

        assert result.is_error
        assert "end" in result.error

    # ------------------------------------------------------------------
    # delete_event
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_delete_event(self):
        events_service = MagicMock()
        events_service.delete.return_value.execute.return_value = {}
        service = _make_service(events_service)

        with patch.dict("sys.modules", _make_google_mocks(service)):
            tool = GoogleCalendarTool()
            result = await tool.run(action="delete_event", event_id="evt-123")

        assert not result.is_error
        assert "Deleted event evt-123" in result.output
        events_service.delete.assert_called_once_with(calendarId="primary", eventId="evt-123")

    @pytest.mark.asyncio
    async def test_delete_event_missing_id(self):
        tool = GoogleCalendarTool()
        result = await tool.run(action="delete_event")

        assert result.is_error
        assert "event_id" in result.error

    @pytest.mark.asyncio
    async def test_delete_event_custom_calendar(self):
        events_service = MagicMock()
        events_service.delete.return_value.execute.return_value = {}
        service = _make_service(events_service)

        with patch.dict("sys.modules", _make_google_mocks(service)):
            tool = GoogleCalendarTool()
            result = await tool.run(
                action="delete_event",
                event_id="evt-456",
                calendar_id="work@group.calendar.google.com",
            )

        assert not result.is_error
        events_service.delete.assert_called_once_with(
            calendarId="work@group.calendar.google.com", eventId="evt-456"
        )

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_no_action_returns_error(self):
        tool = GoogleCalendarTool()
        result = await tool.run()

        assert result.is_error
        assert "No action" in result.error

    @pytest.mark.asyncio
    async def test_unknown_action_returns_error(self):
        tool = GoogleCalendarTool()
        result = await tool.run(action="update_event")

        assert result.is_error
        assert "Unknown action" in result.error
        assert "update_event" in result.error

    # ------------------------------------------------------------------
    # Schema and metadata
    # ------------------------------------------------------------------

    def test_tool_name_and_description(self):
        tool = GoogleCalendarTool()
        assert tool.name == "google_calendar"
        assert "Calendar" in tool.description

    def test_schema_returns_valid_openai_format(self):
        tool = GoogleCalendarTool()
        schema = tool.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "google_calendar"
        assert "action" in schema["function"]["parameters"]["properties"]
        assert "action" in schema["function"]["parameters"]["required"]

    def test_anthropic_schema_returns_valid_format(self):
        tool = GoogleCalendarTool()
        schema = tool.anthropic_schema()
        assert schema["name"] == "google_calendar"
        assert "action" in schema["input_schema"]["properties"]
