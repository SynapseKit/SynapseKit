from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..base import BaseTool, ToolResult


class GoogleCalendarTool(BaseTool):
    """Create, list, and delete Google Calendar events."""

    name = "google_calendar"
    description = "Interact with Google Calendar. Actions: list_events, create_event, delete_event."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action to perform",
                "enum": ["list_events", "create_event", "delete_event"],
            },
            "calendar_id": {
                "type": "string",
                "description": "Calendar ID (default: primary)",
                "default": "primary",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of events to return for list_events",
                "default": 10,
            },
            "time_min": {
                "type": "string",
                "description": "RFC3339 lower bound for list_events",
                "default": "",
            },
            "time_max": {
                "type": "string",
                "description": "RFC3339 upper bound for list_events",
                "default": "",
            },
            "event_id": {
                "type": "string",
                "description": "Event ID to delete",
                "default": "",
            },
            "summary": {
                "type": "string",
                "description": "Event summary for create_event",
                "default": "",
            },
            "description": {
                "type": "string",
                "description": "Event description for create_event",
                "default": "",
            },
            "start": {
                "type": "string",
                "description": "RFC3339 start datetime for create_event",
                "default": "",
            },
            "end": {
                "type": "string",
                "description": "RFC3339 end datetime for create_event",
                "default": "",
            },
            "timezone": {
                "type": "string",
                "description": "Timezone for create_event (default: UTC)",
                "default": "UTC",
            },
        },
        "required": ["action"],
    }

    async def run(self, action: str = "", **kwargs: Any) -> ToolResult:
        if not action:
            return ToolResult(output="", error="No action specified for Google Calendar.")

        if action == "list_events":
            return await self._list_events(**kwargs)
        if action == "create_event":
            return await self._create_event(**kwargs)
        if action == "delete_event":
            return await self._delete_event(**kwargs)

        return ToolResult(
            output="",
            error=f"Unknown action: {action}. Must be one of: list_events, create_event, delete_event",
        )

    def _get_service(self) -> Any:
        try:
            import google.auth
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "google-api-python-client required for GoogleCalendarTool: pip install synapsekit[gcal-tool]"
            ) from None

        scopes = ["https://www.googleapis.com/auth/calendar"]
        credentials, _ = google.auth.default(scopes=scopes)
        return build("calendar", "v3", credentials=credentials, cache_discovery=False)

    async def _run_blocking(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    async def _list_events(
        self,
        calendar_id: str = "primary",
        max_results: int = 10,
        time_min: str = "",
        time_max: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        service = self._get_service()
        list_kwargs: dict[str, Any] = {
            "calendarId": calendar_id,
            "maxResults": max_results,
            "singleEvents": True,
            "orderBy": "startTime",
        }
        if time_min:
            list_kwargs["timeMin"] = time_min
        if time_max:
            list_kwargs["timeMax"] = time_max

        request = service.events().list(**list_kwargs)
        data: dict[str, Any] = await self._run_blocking(request.execute)
        events = data.get("items", [])
        if not events:
            return ToolResult(output="No events found.")

        lines = []
        for event in events:
            start = event.get("start", {}).get("dateTime") or event.get("start", {}).get("date", "")
            end = event.get("end", {}).get("dateTime") or event.get("end", {}).get("date", "")
            lines.append(
                f"- {event.get('summary', 'Untitled')} | {start} -> {end} | {event.get('id', '')}"
            )
        return ToolResult(output="\n".join(lines))

    async def _create_event(
        self,
        calendar_id: str = "primary",
        summary: str = "",
        description: str = "",
        start: str = "",
        end: str = "",
        timezone: str = "UTC",
        **kwargs: Any,
    ) -> ToolResult:
        if not summary or not start or not end:
            return ToolResult(
                output="",
                error="summary, start, and end are required for create_event.",
            )

        service = self._get_service()
        event = {
            "summary": summary,
            "description": description,
            "start": {"dateTime": start, "timeZone": timezone},
            "end": {"dateTime": end, "timeZone": timezone},
        }
        created: dict[str, Any] = await self._run_blocking(
            service.events().insert(calendarId=calendar_id, body=event).execute
        )
        return ToolResult(
            output=f"Created event: {created.get('summary', summary)} | {created.get('htmlLink', '')}"
        )

    async def _delete_event(
        self,
        calendar_id: str = "primary",
        event_id: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        if not event_id:
            return ToolResult(output="", error="event_id is required for delete_event.")

        service = self._get_service()
        await self._run_blocking(
            service.events().delete(calendarId=calendar_id, eventId=event_id).execute
        )
        return ToolResult(output=f"Deleted event {event_id} from calendar {calendar_id}.")
