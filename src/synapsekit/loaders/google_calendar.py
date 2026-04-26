from __future__ import annotations

import asyncio
from typing import Any

from .base import Document


class GoogleCalendarLoader:
    """Load Google Calendar events.

    Example::

        loader = GoogleCalendarLoader(
            credentials_path="/path/to/credentials.json",
            calendar_id="primary",
            max_results=50,
        )
        docs = loader.load()

    pip install synapsekit[google-calendar]
    (requires google-api-python-client>=2.0, google-auth>=2.0)
    """

    _SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

    def __init__(
        self,
        credentials_path: str,
        calendar_id: str = "primary",
        max_results: int = 50,
        time_min: str | None = None,
        time_max: str | None = None,
    ) -> None:
        if not credentials_path:
            raise ValueError("credentials_path must be provided")

        self._credentials_path = credentials_path
        self._calendar_id = calendar_id
        self._max_results = max_results
        self._time_min = time_min
        self._time_max = time_max

    def load(self) -> list[Document]:
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "google-api-python-client and google-auth required: "
                "pip install synapsekit[google-calendar]"
            ) from None

        credentials = service_account.Credentials.from_service_account_file(
            self._credentials_path,
            scopes=self._SCOPES,
        )
        service = build("calendar", "v3", credentials=credentials, cache_discovery=False)

        params: dict[str, Any] = {
            "calendarId": self._calendar_id,
            "maxResults": self._max_results,
            "singleEvents": True,
            "orderBy": "startTime",
        }
        if self._time_min:
            params["timeMin"] = self._time_min
        if self._time_max:
            params["timeMax"] = self._time_max

        result = service.events().list(**params).execute()
        events: list[dict[str, Any]] = result.get("items", [])

        docs: list[Document] = []
        for event in events:
            summary = event.get("summary", "")
            description = event.get("description", "")
            text = summary + ("\n" + description if description else "")

            start: dict[str, Any] = event.get("start", {})
            end: dict[str, Any] = event.get("end", {})
            organizer: dict[str, Any] = event.get("organizer", {})

            metadata: dict[str, Any] = {
                "source": "google_calendar",
                "event_id": event.get("id"),
                "start": start.get("dateTime") or start.get("date"),
                "end": end.get("dateTime") or end.get("date"),
                "location": event.get("location"),
                "html_link": event.get("htmlLink"),
                "organizer": organizer.get("email"),
            }
            docs.append(Document(text=text, metadata=metadata))

        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)
