"""SQLLoader — load rows from SQL databases as documents."""

from __future__ import annotations

import asyncio
import logging

from .base import Document

logger = logging.getLogger(__name__)


class SQLLoader:
    """Load rows from SQL databases as documents.

    This loader uses SQLAlchemy to execute queries and convert each row into a
    Document. Supports both synchronous and asynchronous loading.

    Prerequisites:
        - SQLAlchemy 2.0+
        - Appropriate database driver (e.g., psycopg2, pymysql, etc.)

    Example::

        loader = SQLLoader(
            connection_string="sqlite:///example.db",
            query="SELECT * FROM documents WHERE category = 'news'",
            text_columns=["title", "content"],
        )
        docs = loader.load()  # synchronous
        # or
        docs = await loader.aload()  # asynchronous
    """

    def __init__(
        self,
        connection_string: str,
        query: str,
        text_columns: list[str] | None = None,
        metadata_columns: list[str] | None = None,
    ) -> None:
        """Initialize SQL loader.

        Args:
            connection_string: SQLAlchemy database URL
            query: SQL query to execute
            text_columns: Columns to concatenate as document text (if None, all columns)
            metadata_columns: Columns to include in metadata (if None, all columns)
        """
        if not connection_string:
            raise ValueError("connection_string must be provided")
        if not query:
            raise ValueError("query must be provided")

        self.connection_string = connection_string
        self.query = query
        self.text_columns = text_columns
        self.metadata_columns = metadata_columns

    def load(self) -> list[Document]:
        """Synchronously execute query and return Documents."""
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.aload())
        finally:
            loop.close()

    async def aload(self) -> list[Document]:
        """Asynchronously execute query and return Documents."""
        try:
            from sqlalchemy import create_engine, text
        except ImportError:
            raise ImportError("SQLAlchemy required: pip install synapsekit[sql]") from None

        loop = asyncio.get_running_loop()

        # Create engine and execute query
        engine = await loop.run_in_executor(None, lambda: create_engine(self.connection_string))

        documents = []

        def _execute_and_fetch():
            with engine.connect() as connection:
                result = connection.execute(text(self.query))
                rows = result.fetchall()
                keys = result.keys()
                return [dict(zip(keys, row, strict=False)) for row in rows]

        try:
            rows = await loop.run_in_executor(None, _execute_and_fetch)

            for idx, row in enumerate(rows):
                # Determine text content
                if self.text_columns:
                    text_parts = [str(row.get(col, "")) for col in self.text_columns]
                    text = " ".join(text_parts)
                else:
                    # Use all columns
                    text = " ".join(str(v) for v in row.values())

                # Determine metadata
                if self.metadata_columns:
                    metadata = {col: row[col] for col in self.metadata_columns if col in row}
                else:
                    metadata = dict(row)

                metadata["source"] = "sql"
                metadata["row_index"] = idx

                documents.append(Document(text=text, metadata=metadata))

        finally:
            await loop.run_in_executor(None, engine.dispose)

        return documents
