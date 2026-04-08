from __future__ import annotations

import os
from typing import Any

from .base import Document


class GoogleSheetsLoader:
    """Load data from a Google Sheets document."""

    def __init__(
        self,
        spreadsheet_id: str,
        sheet_name: str | None = None,
        credentials_path: str = "credentials.json",
    ) -> None:
        self._spreadsheet_id = spreadsheet_id
        self._sheet_name = sheet_name
        self._credentials_path = credentials_path

    def _get_credentials(self) -> Any:
        from google.oauth2.service_account import Credentials

        scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        if os.path.exists(self._credentials_path):
            return Credentials.from_service_account_file(self._credentials_path, scopes=scopes)
        raise FileNotFoundError(f"Credentials file not found at {self._credentials_path}")

    def load(self) -> list[Document]:
        try:
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "google-api-python-client required: pip install synapsekit[gsheets]"
            ) from None

        try:
            import google.auth  # noqa: F401
        except ImportError:
            raise ImportError("google-auth required: pip install synapsekit[gsheets]") from None

        creds = self._get_credentials()
        service = build("sheets", "v4", credentials=creds, cache_discovery=False)
        sheets_api = service.spreadsheets()

        if self._sheet_name is None:
            # Fetch the first sheet if not specified
            sheet_metadata = sheets_api.get(spreadsheetId=self._spreadsheet_id).execute()
            sheets = sheet_metadata.get("sheets", [])
            if not sheets:
                return []
            self._sheet_name = sheets[0].get("properties", {}).get("title", "")

        range_name = f"{self._sheet_name}"
        result = (
            sheets_api.values().get(spreadsheetId=self._spreadsheet_id, range=range_name).execute()
        )
        values = result.get("values", [])

        if not values:
            return []

        docs: list[Document] = []
        headers = values[0]
        for row_idx, row in enumerate(values[1:], start=2):
            row_dict = {}
            for col_idx, header in enumerate(headers):
                val = row[col_idx] if col_idx < len(row) else ""
                row_dict[str(header)] = str(val)

            text = ", ".join(f"{k}: {v}" for k, v in row_dict.items())
            metadata = {
                "source": f"https://docs.google.com/spreadsheets/d/{self._spreadsheet_id}",
                "sheet": self._sheet_name,
                "row": row_idx,
            }
            docs.append(Document(text=text, metadata=metadata))

        return docs
