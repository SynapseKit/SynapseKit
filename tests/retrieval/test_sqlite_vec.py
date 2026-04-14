"""Tests for SQLiteVecStore (sqlite-vec backend) with mocked sqlite_vec loader."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class FakeCursor:
    def __init__(self, one=None, many=None):
        self._one = one
        self._many = many or []

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._many


class FakeConnection:
    def __init__(self, path: str):
        self.path = path
        self.row_factory = None
        self._closed = False
        self._tables: set[str] = set()
        self._meta_dim: int | None = None
        self._rows: list[dict] = []
        self._next_rowid = 1

    def enable_load_extension(self, _enabled: bool) -> None:
        return None

    def _extract_table_name(self, sql: str, marker: str) -> str:
        frag = sql.lower().split(marker, 1)[1].strip().split()[0]
        return frag.strip('"')

    def execute(self, sql: str, params=()):
        norm = " ".join(sql.strip().split()).lower()

        if norm.startswith("create table if not exists"):
            table = self._extract_table_name(norm, "create table if not exists")
            self._tables.add(table)
            return FakeCursor()

        if "from sqlite_master" in norm:
            wanted = params[0]
            return FakeCursor((1,) if wanted in self._tables else None)

        if norm.startswith("select dim from"):
            if self._meta_dim is None:
                return FakeCursor(None)
            return FakeCursor((self._meta_dim,))

        if norm.startswith("insert into") and "on conflict" in norm:
            self._meta_dim = int(params[0])
            return FakeCursor()

        if norm.startswith("create virtual table"):
            table = self._extract_table_name(norm, "create virtual table")
            self._tables.add(table)
            return FakeCursor()

        if norm.startswith("insert into") and "values (?, ?, ?)" in norm:
            vec, text, metadata = params
            arr = np.frombuffer(vec, dtype=np.float32)
            self._rows.append(
                {
                    "rowid": self._next_rowid,
                    "embedding": arr,
                    "text": text,
                    "metadata": metadata,
                }
            )
            self._next_rowid += 1
            return FakeCursor()

        if norm.startswith("select rowid, text, metadata, distance"):
            query_vec = np.frombuffer(params[0], dtype=np.float32)
            limit = int(params[1])
            scored = []
            for row in self._rows:
                dist = float(np.linalg.norm(row["embedding"] - query_vec))
                scored.append((row["rowid"], row["text"], row["metadata"], dist))
            scored.sort(key=lambda x: x[3])
            return FakeCursor(many=scored[:limit])

        if norm.startswith("select count(*)"):
            return FakeCursor((len(self._rows),))

        raise AssertionError(f"Unexpected SQL: {sql}")

    def commit(self) -> None:
        return None

    def backup(self, dest: FakeConnection) -> None:
        dest._tables = set(self._tables)
        dest._meta_dim = self._meta_dim
        dest._rows = [
            {
                "rowid": r["rowid"],
                "embedding": np.array(r["embedding"], dtype=np.float32),
                "text": r["text"],
                "metadata": r["metadata"],
            }
            for r in self._rows
        ]
        dest._next_rowid = self._next_rowid

    def close(self) -> None:
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.commit()
        return False


class FakeSQLiteFactory:
    def __init__(self):
        self._connections: dict[str, FakeConnection] = {}

    def connect(self, path: str):
        key = str(Path(path))
        if key not in self._connections:
            self._connections[key] = FakeConnection(key)
        if path != ":memory:":
            p = Path(key)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch(exist_ok=True)
        return self._connections[key]


def make_mock_embeddings(dim: int = 4):
    mock = MagicMock()

    def _vec_for(text: str) -> np.ndarray:
        v = np.zeros(dim, dtype=np.float32)
        idx = sum(ord(c) for c in text) % dim
        v[idx] = 1.0
        return v

    async def embed(texts):
        return np.array([_vec_for(t) for t in texts], dtype=np.float32)

    async def embed_one(text):
        return _vec_for(text)

    mock.embed = embed
    mock.embed_one = embed_one
    return mock


class TestSQLiteVecStore:
    def test_import_error_without_sqlite_vec(self):
        with patch.dict("sys.modules", {"sqlite_vec": None}):
            import synapsekit.retrieval.sqlite_vec as sqlite_vec_mod

            importlib.reload(sqlite_vec_mod)
            with pytest.raises(ImportError, match="sqlite-vec"):
                sqlite_vec_mod.SQLiteVecStore(make_mock_embeddings())

    @pytest.mark.asyncio
    async def test_add_search_and_metadata_filter(self):
        fake_sqlite = FakeSQLiteFactory()
        fake_sqlite_vec = MagicMock()

        with patch("sqlite3.connect", side_effect=fake_sqlite.connect), patch.dict(
            "sys.modules", {"sqlite_vec": fake_sqlite_vec}
        ):
            import synapsekit.retrieval.sqlite_vec as sqlite_vec_mod

            importlib.reload(sqlite_vec_mod)
            store = sqlite_vec_mod.SQLiteVecStore(make_mock_embeddings())

            await store.add(
                ["alpha", "beta", "gamma"],
                metadata=[{"src": "a"}, {"src": "b"}, {"src": "a"}],
            )

            results = await store.search("alpha", top_k=2)
            assert len(results) == 2
            assert all("text" in r and "score" in r and "metadata" in r for r in results)

            filtered = await store.search("alpha", top_k=5, metadata_filter={"src": "b"})
            assert len(filtered) == 1
            assert filtered[0]["metadata"]["src"] == "b"

            assert len(store) == 3

    @pytest.mark.asyncio
    async def test_save_and_load_roundtrip(self, tmp_path):
        fake_sqlite = FakeSQLiteFactory()
        fake_sqlite_vec = MagicMock()

        with patch("sqlite3.connect", side_effect=fake_sqlite.connect), patch.dict(
            "sys.modules", {"sqlite_vec": fake_sqlite_vec}
        ):
            import synapsekit.retrieval.sqlite_vec as sqlite_vec_mod

            importlib.reload(sqlite_vec_mod)

            store = sqlite_vec_mod.SQLiteVecStore(make_mock_embeddings())
            await store.add(["persist me"], metadata=[{"id": 1}])

            save_path = str(tmp_path / "sqlite_vec_store.db")
            store.save(save_path)

            loaded = sqlite_vec_mod.SQLiteVecStore(make_mock_embeddings())
            loaded.load(save_path)

            results = await loaded.search("persist me", top_k=1)
            assert len(results) == 1
            assert results[0]["text"] == "persist me"
            assert results[0]["metadata"] == {"id": 1}

    def test_save_raises_when_empty(self, tmp_path):
        fake_sqlite = FakeSQLiteFactory()
        fake_sqlite_vec = MagicMock()

        with patch("sqlite3.connect", side_effect=fake_sqlite.connect), patch.dict(
            "sys.modules", {"sqlite_vec": fake_sqlite_vec}
        ):
            import synapsekit.retrieval.sqlite_vec as sqlite_vec_mod

            importlib.reload(sqlite_vec_mod)
            store = sqlite_vec_mod.SQLiteVecStore(make_mock_embeddings())

            with pytest.raises(ValueError, match="empty"):
                store.save(str(tmp_path / "empty.db"))
