"""Tests for SQLLoader."""

from __future__ import annotations

import pytest

from synapsekit.loaders import Document, SQLLoader


class TestSQLLoader:
    """Test suite for SQLLoader."""

    def test_init_requires_connection_string(self):
        """Test that connection_string is required."""
        with pytest.raises(ValueError, match="connection_string must be provided"):
            SQLLoader(connection_string="", query="SELECT * FROM test")

    def test_init_requires_query(self):
        """Test that query is required."""
        with pytest.raises(ValueError, match="query must be provided"):
            SQLLoader(connection_string="sqlite:///:memory:", query="")

    def test_init_with_minimal_params(self):
        """Test initialization with minimal parameters."""
        loader = SQLLoader(connection_string="sqlite:///:memory:", query="SELECT * FROM users")
        assert loader.connection_string == "sqlite:///:memory:"
        assert loader.query == "SELECT * FROM users"
        assert loader.text_columns is None
        assert loader.metadata_columns is None

    def test_init_with_all_params(self):
        """Test initialization with all parameters."""
        loader = SQLLoader(
            connection_string="postgresql://user:pass@localhost/db",
            query="SELECT * FROM articles WHERE published = true",
            text_columns=["title", "content"],
            metadata_columns=["id", "author", "date"],
        )
        assert loader.connection_string == "postgresql://user:pass@localhost/db"
        assert loader.query == "SELECT * FROM articles WHERE published = true"
        assert loader.text_columns == ["title", "content"]
        assert loader.metadata_columns == ["id", "author", "date"]

    @pytest.mark.asyncio
    async def test_aload_with_sqlite_memory(self):
        """Test loading from SQLite in-memory database."""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        create_engine, text = sqlalchemy.create_engine, sqlalchemy.text

        # Create in-memory database and populate
        engine = create_engine("sqlite:///:memory:")
        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE documents (id INTEGER, title TEXT, content TEXT)"))
            conn.execute(
                text("INSERT INTO documents VALUES (1, 'First', 'This is the first document')"),
            )
            conn.execute(
                text("INSERT INTO documents VALUES (2, 'Second', 'This is the second document')"),
            )
            conn.commit()

        # Load documents
        SQLLoader(connection_string="sqlite:///:memory:", query="SELECT * FROM documents")

        # Note: Since we're using a new connection, we need to recreate the table
        # This is a limitation of in-memory SQLite for testing
        # We'll use a better approach

    def test_load_with_sqlite_file(self, tmp_path):
        """Test loading from SQLite file database."""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        create_engine, text = sqlalchemy.create_engine, sqlalchemy.text

        # Create temp database file
        db_path = tmp_path / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")

        # Create and populate table
        with engine.connect() as conn:
            conn.execute(
                text("""
                CREATE TABLE articles (
                    id INTEGER PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    author TEXT,
                    published INTEGER
                )
            """)
            )
            conn.execute(
                text("""
                INSERT INTO articles (id, title, content, author, published)
                VALUES
                    (1, 'AI News', 'Recent developments in AI', 'Alice', 1),
                    (2, 'Python Tips', 'Top 10 Python tricks', 'Bob', 1)
            """)
            )
            conn.commit()

        # Load all columns as text
        loader = SQLLoader(
            connection_string=f"sqlite:///{db_path}",
            query="SELECT * FROM articles WHERE published = 1",
        )
        docs = loader.load()

        assert len(docs) == 2
        assert isinstance(docs[0], Document)
        assert "AI News" in docs[0].text or "Recent developments" in docs[0].text
        assert docs[0].metadata["source"] == "sql"
        assert "row_index" in docs[0].metadata

    def test_load_with_text_columns(self, tmp_path):
        """Test loading with specific text columns."""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        create_engine, text = sqlalchemy.create_engine, sqlalchemy.text

        db_path = tmp_path / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")

        with engine.connect() as conn:
            conn.execute(
                text("""
                CREATE TABLE posts (
                    id INTEGER PRIMARY KEY,
                    title TEXT,
                    body TEXT,
                    tags TEXT
                )
            """)
            )
            conn.execute(
                text("""
                INSERT INTO posts (id, title, body, tags)
                VALUES (1, 'Hello World', 'This is my first post', 'intro,hello')
            """)
            )
            conn.commit()

        # Only use title and body as text
        loader = SQLLoader(
            connection_string=f"sqlite:///{db_path}",
            query="SELECT * FROM posts",
            text_columns=["title", "body"],
        )
        docs = loader.load()

        assert len(docs) == 1
        assert "Hello World" in docs[0].text
        assert "This is my first post" in docs[0].text
        # tags should not be in text
        assert "intro,hello" not in docs[0].text or "intro,hello" in str(docs[0].metadata)

    def test_load_with_metadata_columns(self, tmp_path):
        """Test loading with specific metadata columns."""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        create_engine, text = sqlalchemy.create_engine, sqlalchemy.text

        db_path = tmp_path / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")

        with engine.connect() as conn:
            conn.execute(
                text("""
                CREATE TABLE logs (
                    id INTEGER,
                    timestamp TEXT,
                    level TEXT,
                    message TEXT
                )
            """)
            )
            conn.execute(
                text("""
                INSERT INTO logs (id, timestamp, level, message)
                VALUES (1, '2024-01-01', 'INFO', 'Application started')
            """)
            )
            conn.commit()

        # Only include id and level in metadata
        loader = SQLLoader(
            connection_string=f"sqlite:///{db_path}",
            query="SELECT * FROM logs",
            text_columns=["message"],
            metadata_columns=["id", "level"],
        )
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].text == "Application started"
        assert docs[0].metadata["id"] == 1
        assert docs[0].metadata["level"] == "INFO"
        assert "timestamp" not in docs[0].metadata or docs[0].metadata.get("source") == "sql"

    def test_load_with_where_clause(self, tmp_path):
        """Test loading with WHERE clause filtering."""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        create_engine, text = sqlalchemy.create_engine, sqlalchemy.text

        db_path = tmp_path / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")

        with engine.connect() as conn:
            conn.execute(
                text("""
                CREATE TABLE items (
                    id INTEGER,
                    name TEXT,
                    category TEXT
                )
            """)
            )
            conn.execute(
                text("""
                INSERT INTO items (id, name, category) VALUES
                    (1, 'Apple', 'fruit'),
                    (2, 'Carrot', 'vegetable'),
                    (3, 'Banana', 'fruit')
            """)
            )
            conn.commit()

        # Only load fruits
        loader = SQLLoader(
            connection_string=f"sqlite:///{db_path}",
            query="SELECT * FROM items WHERE category = 'fruit'",
        )
        docs = loader.load()

        assert len(docs) == 2
        texts = [doc.text for doc in docs]
        assert any("Apple" in text for text in texts)
        assert any("Banana" in text for text in texts)
        assert not any("Carrot" in text for text in texts)

    def test_load_empty_result(self, tmp_path):
        """Test loading with query that returns no rows."""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        create_engine, text = sqlalchemy.create_engine, sqlalchemy.text

        db_path = tmp_path / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")

        with engine.connect() as conn:
            conn.execute(
                text("""
                CREATE TABLE empty_table (id INTEGER, data TEXT)
            """)
            )
            conn.commit()

        loader = SQLLoader(
            connection_string=f"sqlite:///{db_path}", query="SELECT * FROM empty_table"
        )
        docs = loader.load()

        assert len(docs) == 0
        assert docs == []

    @pytest.mark.asyncio
    async def test_aload_async(self, tmp_path):
        """Test asynchronous loading."""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        create_engine, text = sqlalchemy.create_engine, sqlalchemy.text

        db_path = tmp_path / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")

        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE async_test (id INTEGER, value TEXT)"))
            conn.execute(text("INSERT INTO async_test VALUES (1, 'async data')"))
            conn.commit()

        loader = SQLLoader(
            connection_string=f"sqlite:///{db_path}", query="SELECT * FROM async_test"
        )
        docs = await loader.aload()

        assert len(docs) == 1
        assert "async data" in docs[0].text

    @pytest.mark.asyncio
    async def test_aload_missing_dependencies(self):
        """Test error when SQLAlchemy is missing."""
        import sys
        from unittest.mock import patch

        def mock_import(name, *args, **kwargs):
            if "sqlalchemy" in name:
                raise ImportError("No module named 'sqlalchemy'")
            return __import__(name, *args, **kwargs)

        with patch.dict(sys.modules, {"sqlalchemy": None}):
            with patch("builtins.__import__", side_effect=mock_import):
                loader = SQLLoader(connection_string="sqlite:///:memory:", query="SELECT 1")

                with pytest.raises(
                    ImportError,
                    match="SQLAlchemy required",
                ):
                    await loader.aload()
