from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from .base import Document


class SnowflakeLoader:
    """Load documents from a Snowflake query."""

    def __init__(
        self,
        account: str,
        user: str,
        password: str,
        query: str,
        database: str | None = None,
        schema: str | None = None,
        warehouse: str | None = None,
        role: str | None = None,
        text_fields: list[str] | None = None,
        limit: int | None = None,
    ) -> None:
        if not account:
            raise ValueError("account must be provided")
        if not user:
            raise ValueError("user must be provided")
        if not password:
            raise ValueError("password must be provided")
        if not query:
            raise ValueError("query must be provided")

        self._account = account
        self._user = user
        self._password = password
        self._database = database
        self._schema = schema
        self._warehouse = warehouse
        self._role = role
        self._query = query
        self._text_fields = text_fields
        self._limit = limit

    def _effective_query(self) -> str:
        """Append LIMIT to the query when set, so Snowflake enforces it server-side."""
        if self._limit is None:
            return self._query
        q = self._query.rstrip().rstrip(";")
        return f"{q} LIMIT {int(self._limit)}"

    def load(self) -> list[Document]:
        try:
            import snowflake.connector
        except ImportError:
            raise ImportError(
                "snowflake-connector-python required: pip install synapsekit[snowflake]"
            ) from None

        conn = None
        cur = None
        try:
            conn = snowflake.connector.connect(
                account=self._account,
                user=self._user,
                password=self._password,
                database=self._database,
                schema=self._schema,
                warehouse=self._warehouse,
                role=self._role,
            )
            cur = conn.cursor()
            cur.execute(self._effective_query())
            rows = cur.fetchall()
            columns = [col[0] for col in (cur.description or [])]
        except Exception as e:
            raise RuntimeError(f"Snowflake query failed: {e}") from e
        finally:
            if cur is not None:
                with suppress(Exception):
                    cur.close()
            if conn is not None:
                with suppress(Exception):
                    conn.close()

        docs: list[Document] = []
        for row in rows:
            record = dict(zip(columns, row, strict=False))

            if self._text_fields is not None:
                text = " ".join(str(record.get(f, "")) for f in self._text_fields if record.get(f))
            else:
                text = " ".join(str(v) for v in record.values() if v not in (None, ""))

            if not text:
                continue

            # NOTE: includes all columns; may be large for wide tables
            metadata: dict[str, Any] = {
                "source": "snowflake",
                "database": self._database,
                "schema": self._schema,
                "query": self._query,
                **record,
            }

            docs.append(Document(text=text, metadata=metadata))

        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)
