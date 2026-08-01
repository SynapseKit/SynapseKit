"""Personal knowledge mesh over local markdown and git repositories."""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import threading
import time
from collections import defaultdict
from collections.abc import AsyncGenerator, Iterable
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from .._compat import run_sync
from ..llm.base import BaseLLM, LLMConfig
from ..loaders.base import Document
from ..retrieval.vectorstore import InMemoryVectorStore
from ..retrieval.world_model import (
    GraphBackend,
    HeuristicWorldModelExtractor,
    InMemoryWorldGraphBackend,
    KuzuWorldGraphBackend,
    QueryStrategy,
    WorldModelExtractor,
    WorldModelRAG,
)
from .embeddings import HashingEmbeddings
from .loaders import (
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MESH_INCLUDES,
    GitRepoLoader,
    LocalMdLoader,
)
from .privacy import DEFAULT_MESH_IGNORE, MeshPrivacyFilter
from .resolution import CrossProjectEntityResolver, DuplicationDetector, DuplicationMatch

VectorBackend = Literal["auto", "memory", "sqlite_vec"]
GraphStoreBackend = Literal["auto", "memory", "kuzu"]
_QUERY_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class MeshConfig:
    """Configuration for a personal knowledge mesh."""

    roots: list[str | Path] = field(default_factory=lambda: [Path.cwd()])
    include: list[str] = field(default_factory=lambda: list(DEFAULT_MESH_INCLUDES))
    ignore_file: str | Path | None = DEFAULT_MESH_IGNORE
    state_dir: str | Path = field(default_factory=lambda: Path.home() / ".synapsekit" / "mesh")
    db_path: str | Path | None = None
    graph_path: str | Path | None = None
    vector_backend: VectorBackend = "auto"
    graph_backend: GraphStoreBackend = "memory"
    use_git: bool = True
    include_git_history: bool = True
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    chunk_chars: int = 4_000
    embedding_dimensions: int = 384
    retrieval_top_k: int = 5

    def expanded_roots(self) -> list[Path]:
        """Return existing roots as expanded ``Path`` objects."""

        return [Path(root).expanduser() for root in self.roots]

    def expanded_state_dir(self) -> Path:
        """Return the mesh state directory."""

        return Path(self.state_dir).expanduser()

    def expanded_db_path(self) -> Path:
        """Return the vector database path."""

        if self.db_path is not None:
            return Path(self.db_path).expanduser()
        return self.expanded_state_dir() / "vectors.sqlite3"

    def expanded_graph_path(self) -> Path:
        """Return the optional graph database path."""

        if self.graph_path is not None:
            return Path(self.graph_path).expanduser()
        return self.expanded_state_dir() / "graph.kuzu"


@dataclass(frozen=True)
class MeshHit:
    """A ranked mesh retrieval hit with source citation metadata."""

    text: str
    score: float
    path: str
    line_start: int | None = None
    line_end: int | None = None
    headings: tuple[str, ...] = ()
    repo_root: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MeshQueryResult:
    """Result returned by ``KnowledgeMesh.query``."""

    query: str
    answer: str
    hits: list[MeshHit]
    strategy: QueryStrategy
    graph_entities: tuple[str, ...] = ()


@dataclass(frozen=True)
class MeshIndexSummary:
    """Summary returned after a mesh reindex."""

    discovered_files: int
    discovered_chunks: int
    changed_files: int
    changed_chunks: int
    skipped_chunks: int
    ingested_chunks: int
    duration_seconds: float


@dataclass(frozen=True)
class MeshStatus:
    """Current mesh index status."""

    state: str
    roots: tuple[str, ...]
    indexed_files: int
    indexed_chunks: int
    active_chunks: int
    vector_backend: str
    db_path: str
    graph_path: str
    offline_default: bool
    last_indexed_at: str | None = None
    pid: int | None = None


class LocalMeshLLM(BaseLLM):
    """Offline response generator used when callers do not pass an LLM."""

    def __init__(self) -> None:
        super().__init__(LLMConfig(model="mesh-local", api_key="", provider="local"))

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:  # type: ignore[override]
        yield "Mesh query completed locally. See ranked hits for file citations."


class MeshIndexStore:
    """SQLite metadata store for incremental mesh indexing."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # ``check_same_thread=False`` lets us drive the connection from
        # ``asyncio.to_thread`` worker threads (see ``KnowledgeMesh.reindex``
        # and ``KnowledgeMesh.query``), while a lock serialises access so the
        # single shared connection is never touched concurrently.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._ensure_schema()

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        with self._lock:
            self._conn.close()

    def is_file_current(self, path: str, content_hash: str, mtime_ns: int | None) -> bool:
        """Return whether a file has already been indexed at this fingerprint."""

        with self._lock:
            row = self._conn.execute(
                "SELECT content_hash, mtime_ns FROM mesh_files WHERE path = ?",
                (path,),
            ).fetchone()
        if row is None:
            return False
        if str(row["content_hash"]) != content_hash:
            return False
        if mtime_ns is not None and row["mtime_ns"] is not None:
            return int(row["mtime_ns"]) == int(mtime_ns)
        return True

    def mark_file_chunks(self, path: str, docs: list[Document]) -> None:
        """Mark ``docs`` as active chunks for ``path``."""

        now = datetime.now(UTC).isoformat()
        first_meta = docs[0].metadata if docs else {}
        with self._lock:
            self._conn.execute("UPDATE mesh_chunks SET active = 0 WHERE path = ?", (path,))
            self._conn.execute(
                """
                INSERT INTO mesh_files(path, content_hash, mtime_ns, size_bytes, repo_root, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    mtime_ns = excluded.mtime_ns,
                    size_bytes = excluded.size_bytes,
                    repo_root = excluded.repo_root,
                    updated_at = excluded.updated_at
                """,
                (
                    path,
                    first_meta.get("content_hash"),
                    first_meta.get("mtime_ns"),
                    first_meta.get("size_bytes"),
                    first_meta.get("repo_root"),
                    now,
                ),
            )
            for doc in docs:
                meta = dict(doc.metadata)
                chunk_id = str(meta["chunk_id"])
                self._conn.execute(
                    """
                    INSERT INTO mesh_chunks(
                        chunk_id, path, content_hash, line_start, line_end, text, metadata, active, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        path = excluded.path,
                        content_hash = excluded.content_hash,
                        line_start = excluded.line_start,
                        line_end = excluded.line_end,
                        text = excluded.text,
                        metadata = excluded.metadata,
                        active = 1,
                        updated_at = excluded.updated_at
                    """,
                    (
                        chunk_id,
                        path,
                        meta.get("content_hash"),
                        meta.get("line_start"),
                        meta.get("line_end"),
                        doc.text,
                        json.dumps(meta, sort_keys=True),
                        now,
                    ),
                )
            self._conn.commit()

    def update_status(self, **values: Any) -> None:
        """Persist status key-value pairs."""

        with self._lock, self._conn:
            for key, value in values.items():
                self._conn.execute(
                    """
                    INSERT INTO mesh_status(key, value)
                    VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (key, json.dumps(value)),
                )

    def status_value(self, key: str, default: Any = None) -> Any:
        """Return one status value."""

        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM mesh_status WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return default
        return json.loads(str(row["value"]))

    def active_chunk_ids(self) -> set[str]:
        """Return active chunk IDs."""

        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_id FROM mesh_chunks WHERE active = 1"
            ).fetchall()
        return {str(row["chunk_id"]) for row in rows}

    def filter_active_chunk_ids(self, candidate_ids: Iterable[str]) -> set[str]:
        """Return the subset of ``candidate_ids`` that are currently active.

        Only the handful of candidate chunk IDs surfaced by a vector query are
        checked against the index, avoiding materialising the entire active set
        just to filter ~20 hits. The ``idx_chunks_path_active`` index keeps the
        ``active = 1`` predicate off a full table scan.
        """

        candidates = [str(chunk_id) for chunk_id in candidate_ids]
        if not candidates:
            return set()
        placeholders = ",".join("?" for _ in candidates)
        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_id FROM mesh_chunks "
                f"WHERE chunk_id IN ({placeholders}) AND active = 1",
                candidates,
            ).fetchall()
        return {str(row["chunk_id"]) for row in rows}

    def active_chunk_ids_for_path(self, path: str) -> set[str]:
        """Return currently-active chunk IDs indexed for ``path``.

        Used before reindexing a changed file to find stale chunk IDs whose
        vectors must be removed from the vector store once the new chunks
        are ingested.
        """

        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_id FROM mesh_chunks WHERE path = ? AND active = 1", (path,)
            ).fetchall()
        return {str(row["chunk_id"]) for row in rows}

    def active_documents(self) -> list[Document]:
        """Return active indexed chunks as documents."""

        with self._lock:
            rows = self._conn.execute(
                "SELECT text, metadata FROM mesh_chunks WHERE active = 1 ORDER BY path, line_start"
            ).fetchall()
        return [
            Document(text=str(row["text"]), metadata=json.loads(str(row["metadata"])))
            for row in rows
        ]

    def counts(self) -> tuple[int, int, int]:
        """Return indexed file, total chunk, and active chunk counts."""

        with self._lock:
            files = self._conn.execute("SELECT COUNT(*) FROM mesh_files").fetchone()[0]
            chunks = self._conn.execute("SELECT COUNT(*) FROM mesh_chunks").fetchone()[0]
            active = self._conn.execute(
                "SELECT COUNT(*) FROM mesh_chunks WHERE active = 1"
            ).fetchone()[0]
        return int(files), int(chunks), int(active)

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS mesh_files (
                    path TEXT PRIMARY KEY,
                    content_hash TEXT,
                    mtime_ns INTEGER,
                    size_bytes INTEGER,
                    repo_root TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mesh_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    content_hash TEXT,
                    line_start INTEGER,
                    line_end INTEGER,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mesh_status (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                -- Serves ``WHERE path = ? AND active = 1`` and ``WHERE active = 1``
                -- lookups on every reindex/query without full table scans.
                CREATE INDEX IF NOT EXISTS idx_chunks_path_active
                    ON mesh_chunks(path, active);
                """
            )
            self._conn.commit()


class KnowledgeMesh:
    """Federated local knowledge mesh backed by ``WorldModelRAG``."""

    def __init__(
        self,
        config: MeshConfig | None = None,
        *,
        rag: WorldModelRAG | None = None,
        llm: BaseLLM | None = None,
        extractor: WorldModelExtractor | None = None,
        graph_backend: GraphBackend | None = None,
        vector_store: Any | None = None,
        privacy_filter: MeshPrivacyFilter | None = None,
    ) -> None:
        self.config = config or MeshConfig()
        self.state_dir = self.config.expanded_state_dir()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.privacy_filter = privacy_filter or MeshPrivacyFilter(self.config.ignore_file)
        self.store = MeshIndexStore(self.state_dir / "index.sqlite3")
        self.duplication_detector = DuplicationDetector()
        self._vector_backend = "custom"
        self._memory_vectors_path = self.state_dir / "vectors.npz"

        self.rag = rag or self._build_rag(
            llm=llm,
            extractor=extractor,
            graph_backend=graph_backend,
            vector_store=vector_store,
        )

    async def reindex(self, *, force: bool = False) -> MeshIndexSummary:
        """Incrementally index changed mesh documents."""

        started = time.perf_counter()
        # Reading the whole tree (git + markdown loaders) is blocking I/O; run
        # it in a worker thread so the event loop stays responsive on a large
        # mesh.
        docs = await asyncio.to_thread(self._load_documents)
        grouped = _group_by_path(docs)

        def _select_changed() -> tuple[list[Document], int]:
            selected: list[Document] = []
            skipped = 0
            for path, file_docs in grouped.items():
                meta = file_docs[0].metadata
                current = self.store.is_file_current(
                    path,
                    str(meta.get("content_hash", "")),
                    _optional_int(meta.get("mtime_ns")),
                )
                if current and not force:
                    skipped += len(file_docs)
                    continue
                selected.extend(file_docs)
            return selected, skipped

        # Each ``is_file_current`` call is a synchronous SQLite read; batch them
        # into a single worker-thread hop rather than blocking the loop per file.
        changed_docs, skipped_chunks = await asyncio.to_thread(_select_changed)

        if changed_docs:
            changed_by_path = _group_by_path(changed_docs)

            def _collect_stale() -> set[str]:
                stale: set[str] = set()
                for path, file_docs in changed_by_path.items():
                    new_chunk_ids = {str(doc.metadata.get("chunk_id")) for doc in file_docs}
                    old_chunk_ids = self.store.active_chunk_ids_for_path(path)
                    stale.update(old_chunk_ids - new_chunk_ids)
                return stale

            stale_chunk_ids = await asyncio.to_thread(_collect_stale)

            await self.rag.ingest(changed_docs)

            def _mark_all() -> None:
                for path, file_docs in changed_by_path.items():
                    self.store.mark_file_chunks(path, file_docs)

            await asyncio.to_thread(_mark_all)
            if stale_chunk_ids:
                self.rag.delete_by_metadata("chunk_id", stale_chunk_ids)
            self._save_vector_store()

        summary = MeshIndexSummary(
            discovered_files=len(grouped),
            discovered_chunks=len(docs),
            changed_files=len(_group_by_path(changed_docs)),
            changed_chunks=len(changed_docs),
            skipped_chunks=skipped_chunks,
            ingested_chunks=len(changed_docs),
            duration_seconds=time.perf_counter() - started,
        )
        await asyncio.to_thread(
            self.store.update_status,
            state="ready",
            last_indexed_at=datetime.now(UTC).isoformat(),
            last_summary=asdict(summary),
            roots=[str(root) for root in self.config.expanded_roots()],
            vector_backend=self._vector_backend,
        )
        return summary

    def reindex_sync(self, *, force: bool = False) -> MeshIndexSummary:
        """Sync wrapper for ``reindex``."""

        return run_sync(self.reindex(force=force))

    async def ingest_okf(
        self, path: str | Path, *, extract_body: bool = False, **loader_kwargs: Any
    ) -> int:
        """Ingest an Open Knowledge Format bundle end-to-end (#825).

        Loads the bundle via :class:`~synapsekit.loaders.okf.OpenKnowledgeFormatLoader`,
        vector-indexes each concept body, and builds the *explicit* cross-link
        graph on the mesh's world model via
        :func:`~synapsekit.retrieval.okf_graph.okf_to_world_model` — bypassing
        the lossy ``HeuristicWorldModelExtractor`` for the structure OKF already
        encodes. Pass ``extract_body=True`` to *also* run extraction over the
        freeform Markdown bodies. Query the result through ``self.rag``
        (``WorldModelRAG``) with ``graph_first`` / ``vector_first`` / ``hybrid``.

        Returns the number of concepts ingested. Link resolution is forced on
        (the graph needs it), so any ``resolve_links`` kwarg is ignored.
        """
        from ..loaders.okf import OpenKnowledgeFormatLoader
        from ..retrieval.okf_graph import okf_to_world_model

        loader_kwargs.pop("resolve_links", None)
        docs = await OpenKnowledgeFormatLoader(path, resolve_links=True, **loader_kwargs).aload()

        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for doc in docs:
            if not doc.text.strip():
                continue
            doc_id = doc.metadata.get("concept_path")
            # The vector store indexes metadata into a hashable inverted index,
            # so pass only scalar fields — the rich frontmatter/link structure
            # lives on the graph nodes built below.
            scalar: dict[str, Any] = {"source": doc_id, "world_model_doc_id": doc_id}
            for key in ("concept_path", "okf_type", "title", "resource", "timestamp"):
                value = doc.metadata.get(key)
                if isinstance(value, str | int | float | bool):
                    scalar[key] = value
            texts.append(doc.text)
            metadatas.append(scalar)
        if texts:
            await self.rag.vector_retriever.add(texts, metadatas)

        okf_to_world_model(docs, self.rag.graph_backend)
        if extract_body and docs:
            await self.rag.ingest(docs)
        return len(docs)

    def ingest_okf_sync(
        self, path: str | Path, *, extract_body: bool = False, **loader_kwargs: Any
    ) -> int:
        """Sync wrapper for ``ingest_okf``."""

        return run_sync(self.ingest_okf(path, extract_body=extract_body, **loader_kwargs))

    async def query(
        self,
        query: str,
        *,
        top_k: int | None = None,
        strategy: QueryStrategy = "hybrid",
    ) -> MeshQueryResult:
        """Query the mesh and return ranked hits with file citations."""

        if not query.strip():
            raise ValueError("query must not be empty")

        # ``top_k is not None`` (rather than ``top_k or ...``) so an explicit
        # ``top_k=0`` requests zero hits instead of silently falling back to the
        # configured default.
        limit = top_k if top_k is not None else self.config.retrieval_top_k
        wm_result = await self.rag.query(query, top_k=max(limit * 4, limit), strategy=strategy)
        # Only check the handful of candidate chunk IDs the vector query
        # surfaced against the active set, instead of materialising every active
        # chunk ID in the mesh. The SQLite read is blocking, so run it in a
        # worker thread to keep the event loop free.
        candidate_ids = [
            str(cid)
            for item in wm_result.embeddings
            if (cid := (item.get("metadata") or {}).get("chunk_id")) is not None
        ]
        active = await asyncio.to_thread(self.store.filter_active_chunk_ids, candidate_ids)
        hits = _rerank_hits(query, _hits_from_embeddings(wm_result.embeddings, active, limit * 4))[
            :limit
        ]
        answer = self._answer_from_hits(query, hits, wm_result.answer)
        entities = tuple(node.name for node in wm_result.subgraph.nodes)
        return MeshQueryResult(
            query=query,
            answer=answer,
            hits=hits,
            strategy=wm_result.strategy,
            graph_entities=entities,
        )

    def query_sync(
        self,
        query: str,
        *,
        top_k: int | None = None,
        strategy: QueryStrategy = "hybrid",
    ) -> MeshQueryResult:
        """Sync wrapper for ``query``."""

        return run_sync(self.query(query, top_k=top_k, strategy=strategy))

    def duplicates(self, *, limit: int = 20) -> list[DuplicationMatch]:
        """Return likely duplicate mesh chunks."""

        return self.duplication_detector.find(self.store.active_documents(), limit=limit)

    def status(self) -> MeshStatus:
        """Return current mesh status."""

        indexed_files, indexed_chunks, active_chunks = self.store.counts()
        state = str(self.store.status_value("state", "empty" if active_chunks == 0 else "ready"))
        return MeshStatus(
            state=state,
            roots=tuple(str(root) for root in self.config.expanded_roots()),
            indexed_files=indexed_files,
            indexed_chunks=indexed_chunks,
            active_chunks=active_chunks,
            vector_backend=str(self.store.status_value("vector_backend", self._vector_backend)),
            db_path=str(self.config.expanded_db_path()),
            graph_path=str(self.config.expanded_graph_path()),
            offline_default=isinstance(getattr(self.rag, "llm", None), LocalMeshLLM),
            last_indexed_at=self.store.status_value("last_indexed_at"),
            pid=self.store.status_value("pid"),
        )

    def as_mcp_tools(self) -> list[Any]:
        """Return MCP-compatible tools for this mesh."""

        from .mcp import build_mesh_tools

        return build_mesh_tools(self)

    def _build_rag(
        self,
        *,
        llm: BaseLLM | None,
        extractor: WorldModelExtractor | None,
        graph_backend: GraphBackend | None,
        vector_store: Any | None,
    ) -> WorldModelRAG:
        resolver = CrossProjectEntityResolver()
        graph = graph_backend or self._make_graph_backend(resolver)
        vector = vector_store or self._make_vector_store()
        return WorldModelRAG(
            llm=llm or LocalMeshLLM(),
            vector_store=vector,
            extractor=extractor or HeuristicWorldModelExtractor(),
            graph_backend=graph,
            retrieval_top_k=self.config.retrieval_top_k,
            trace=False,
        )

    def _make_graph_backend(self, resolver: CrossProjectEntityResolver) -> GraphBackend:
        if self.config.graph_backend in {"auto", "kuzu"}:
            try:
                return KuzuWorldGraphBackend(self.config.expanded_graph_path(), resolver=resolver)
            except ImportError:
                if self.config.graph_backend == "kuzu":
                    raise
        return InMemoryWorldGraphBackend(resolver)

    def _make_vector_store(self) -> Any:
        embeddings = HashingEmbeddings(self.config.embedding_dimensions)
        if self.config.vector_backend in {"auto", "sqlite_vec"}:
            try:
                from ..retrieval.sqlite_vec import SQLiteVecStore

                self._vector_backend = "sqlite_vec"
                return SQLiteVecStore(
                    cast(Any, embeddings),
                    db_path=str(self.config.expanded_db_path()),
                )
            except ImportError:
                if self.config.vector_backend == "sqlite_vec":
                    raise

        store = InMemoryVectorStore(embeddings)  # type: ignore[arg-type]
        self._vector_backend = "in_memory"
        if self._memory_vectors_path.exists():
            with suppress(Exception):
                store.load(str(self._memory_vectors_path))
        return store

    def _save_vector_store(self) -> None:
        if self._vector_backend != "in_memory":
            return
        save = getattr(getattr(self.rag, "vector_store", None), "save", None)
        if callable(save) and len(getattr(self.rag.vector_store, "_texts", [])) > 0:
            save(str(self._memory_vectors_path))

    def _load_documents(self) -> list[Document]:
        docs: list[Document] = []
        for root in self.config.expanded_roots():
            if self.config.use_git:
                docs.extend(
                    GitRepoLoader(
                        root,
                        include=self.config.include,
                        privacy_filter=self.privacy_filter,
                        max_file_bytes=self.config.max_file_bytes,
                        include_history=self.config.include_git_history,
                    ).load()
                )
            docs.extend(
                LocalMdLoader(
                    root,
                    include=self.config.include,
                    privacy_filter=self.privacy_filter,
                    max_file_bytes=self.config.max_file_bytes,
                    chunk_chars=self.config.chunk_chars,
                ).load()
            )
        return _dedupe_docs(docs)

    @staticmethod
    def _answer_from_hits(query: str, hits: list[MeshHit], fallback: str) -> str:
        if not hits:
            return fallback
        lines = [f"Found {len(hits)} ranked mesh hit(s) for: {query}"]
        for index, hit in enumerate(hits, start=1):
            citation = hit.path
            if hit.line_start is not None:
                citation += f":{hit.line_start}"
                if hit.line_end is not None and hit.line_end != hit.line_start:
                    citation += f"-{hit.line_end}"
            snippet = " ".join(hit.text.split())[:240]
            lines.append(f"{index}. {citation} - {snippet}")
        return "\n".join(lines)


def _group_by_path(docs: Iterable[Document]) -> dict[str, list[Document]]:
    grouped: dict[str, list[Document]] = defaultdict(list)
    for doc in docs:
        path = str(doc.metadata.get("path") or doc.metadata.get("source") or "")
        if path:
            grouped[path].append(doc)
    return dict(grouped)


def _dedupe_docs(docs: list[Document]) -> list[Document]:
    seen: set[str] = set()
    deduped: list[Document] = []
    for doc in docs:
        key = str(doc.metadata.get("chunk_id") or (doc.metadata.get("source"), doc.text))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def _hits_from_embeddings(
    embeddings: list[dict[str, Any]],
    active_chunk_ids: set[str] | None,
    limit: int,
) -> list[MeshHit]:
    hits: list[MeshHit] = []
    for item in embeddings:
        metadata = dict(item.get("metadata") or {})
        chunk_id = metadata.get("chunk_id")
        if active_chunk_ids is not None and chunk_id not in active_chunk_ids:
            continue
        path = metadata.get("path") or metadata.get("source")
        if not path:
            continue
        hits.append(
            MeshHit(
                text=str(item.get("text", "")),
                score=float(item.get("score", 0.0)),
                path=str(path),
                line_start=_optional_int(metadata.get("line_start")),
                line_end=_optional_int(metadata.get("line_end")),
                headings=tuple(str(heading) for heading in metadata.get("headings", [])),
                repo_root=str(metadata["repo_root"]) if metadata.get("repo_root") else None,
                metadata=metadata,
            )
        )
        if len(hits) >= limit:
            break
    return hits


def _rerank_hits(query: str, hits: list[MeshHit]) -> list[MeshHit]:
    terms = set(_QUERY_TOKEN_RE.findall(query.casefold()))
    if not terms:
        return hits

    def rank(hit: MeshHit) -> tuple[int, float]:
        text_terms = set(_QUERY_TOKEN_RE.findall(hit.text.casefold()))
        path_terms = set(_QUERY_TOKEN_RE.findall(hit.path.casefold()))
        overlap = len(terms & (text_terms | path_terms))
        return overlap, hit.score

    return sorted(hits, key=rank, reverse=True)


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
