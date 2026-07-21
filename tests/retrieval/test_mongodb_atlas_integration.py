"""Real MongoDBAtlasVectorStore integration tests (testcontainers).

Uses the ``mongodb/mongodb-atlas-local`` image, which provides real Atlas Vector
Search (``$vectorSearch``) — the plain ``mongo`` image cannot. Creates the vector
search index out of band (as an Atlas deployment requires), waits for it to
become queryable, and — because Atlas search is *eventually consistent* — polls
until inserted docs are actually searchable before asserting. Part of #829.
"""

from __future__ import annotations

import inspect
import time

import numpy as np
import pytest

pytest.importorskip("pymongo")
_container_mod = pytest.importorskip("testcontainers.core.container")

from synapsekit.retrieval.mongodb_atlas import MongoDBAtlasVectorStore  # noqa: E402

_ATLAS_IMAGE = "mongodb/mongodb-atlas-local:8.0"
_DB = "synapsekit"
_DIM = 9
_VOCAB = ["apple", "banana", "cherry", "fruit", "red", "yellow", "car", "vehicle", "fast"]


class KeywordEmbeddings:
    def __init__(self, vocab: list[str]) -> None:
        self._index = {word: i for i, word in enumerate(vocab)}
        self.dimension = len(vocab)

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self._index), dtype=np.float32)
        for token in text.lower().split():
            if token in self._index:
                vec[self._index[token]] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self._index)), dtype=np.float32)
        return np.vstack([self._one(t) for t in texts]).astype(np.float32)

    async def embed_one(self, text: str) -> np.ndarray:
        return self._one(text)


@pytest.fixture(scope="module")
def atlas_uri():
    from pymongo import MongoClient

    container = _container_mod.DockerContainer(_ATLAS_IMAGE).with_exposed_ports(27017)
    with container as c:
        host = c.get_container_host_ip()
        port = c.get_exposed_port(27017)
        uri = f"mongodb://{host}:{port}/?directConnection=true"
        last_err: Exception | None = None
        for _ in range(120):
            try:
                client = MongoClient(uri, serverSelectionTimeoutMS=1000)
                client.admin.command("ping")
                client.close()
                break
            except Exception as exc:
                last_err = exc
                time.sleep(1)
        else:
            raise RuntimeError(f"atlas-local did not become ready: {last_err}")
        # Warm up mongot once here so its cold-start replica-set step-down is
        # absorbed before any test runs, not mid-test. _prepare_collection is
        # fully transient-tolerant, so this settles the deployment.
        _prepare_collection(uri, "_warmup")
        yield uri


# Transient replica-set / search-service error codes atlas-local throws while
# mongot initializes on cold start: NotPrimary, InterruptedAtShutdown, and
# "Error connecting to Search Index Management service" (125).
_TRANSIENT_CODES = {125, 11600, 91, 189, 10107, 13435, 13436}


def _is_transient(exc: Exception) -> bool:
    from pymongo.errors import AutoReconnect, NotPrimaryError, OperationFailure

    if isinstance(exc, (NotPrimaryError, AutoReconnect)):
        return True
    return isinstance(exc, OperationFailure) and getattr(exc, "code", None) in _TRANSIENT_CODES


def _retry_sync(fn, *, tries: int = 180, desc: str = "op"):
    last: Exception | None = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as exc:
            if not _is_transient(exc):
                raise
            last = exc
            time.sleep(1)
    raise RuntimeError(f"{desc} kept failing on a transient error: {last}")


def _prepare_collection(uri: str, collection: str) -> None:
    """Create the vector search index on a fresh collection and wait until queryable.

    Every server op is transient-tolerant: on cold start (esp. in CI) atlas-local's
    single-node replica set steps down and its search service reconnects while
    mongot initializes, so any op here may briefly hit NotPrimary / code 125.
    """
    from pymongo import MongoClient
    from pymongo.operations import SearchIndexModel

    client = MongoClient(uri)
    coll = client[_DB][collection]
    _retry_sync(lambda: coll.insert_one({"_seed": True}), desc="seed insert")
    _retry_sync(lambda: coll.delete_many({"_seed": True}), desc="seed cleanup")
    model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": _DIM,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "metadata.kind"},
            ]
        },
        name="vector_index",
        type="vectorSearch",
    )
    _retry_sync(lambda: coll.create_search_index(model=model), desc="create search index")
    for _ in range(180):
        try:
            info = list(coll.list_search_indexes("vector_index"))
        except Exception as exc:
            if not _is_transient(exc):
                raise
            time.sleep(1)
            continue
        if info and info[0].get("queryable"):
            break
        time.sleep(1)
    else:
        raise RuntimeError("vector search index did not become queryable")
    client.close()


def _store(uri: str, collection: str) -> MongoDBAtlasVectorStore:
    return MongoDBAtlasVectorStore(
        embedding_backend=KeywordEmbeddings(_VOCAB),
        uri=uri,
        database_name=_DB,
        collection_name=collection,
    )


async def _resilient_add(store, texts, metadata) -> None:
    """Insert, retrying transient replica-set errors.

    atlas-local runs a single-node replica set that briefly steps down /
    restarts when its search service (mongot) first initializes, surfacing as
    NotPrimaryError / "interrupted at shutdown". That's a startup artifact of the
    local image (real Atlas doesn't restart mid-use), so the test tolerates it.
    """
    from pymongo.errors import PyMongoError

    last: Exception | None = None
    for _ in range(30):
        try:
            await store.add(texts, metadata)
            return
        except PyMongoError as exc:
            last = exc
            time.sleep(1)
    raise RuntimeError(f"insert kept failing on a transient error: {last}")


async def _search_until(store, query, expected_text, *, top_k=5, metadata_filter=None):
    """Poll until Atlas's eventually-consistent index surfaces the expected doc,
    tolerating the same transient replica-set errors as _resilient_add."""
    from pymongo.errors import PyMongoError

    for _ in range(60):
        try:
            results = await store.search(query, top_k=top_k, metadata_filter=metadata_filter)
        except PyMongoError:
            time.sleep(1)
            continue
        if any(r["text"] == expected_text for r in results):
            return results
        time.sleep(1)
    return await store.search(query, top_k=top_k, metadata_filter=metadata_filter)


@pytest.mark.asyncio
async def test_add_and_search_returns_true_nearest_neighbor(atlas_uri):
    _prepare_collection(atlas_uri, "rank")
    store = _store(atlas_uri, "rank")
    await _resilient_add(
        store,
        ["apple fruit red", "banana fruit yellow", "car vehicle fast"],
        [{"kind": "fruit"}, {"kind": "fruit"}, {"kind": "auto"}],
    )
    results = await _search_until(store, "apple", "apple fruit red", top_k=3)
    assert results[0]["text"] == "apple fruit red"
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0]["metadata"] == {"kind": "fruit"}


@pytest.mark.asyncio
async def test_metadata_filter(atlas_uri):
    _prepare_collection(atlas_uri, "filter")
    store = _store(atlas_uri, "filter")
    await _resilient_add(
        store, ["apple fruit", "car vehicle"], [{"kind": "fruit"}, {"kind": "auto"}]
    )
    results = await _search_until(
        store, "apple car", "car vehicle", top_k=5, metadata_filter={"kind": "auto"}
    )
    assert [r["text"] for r in results] == ["car vehicle"]


@pytest.mark.asyncio
async def test_persistence_across_reconnect(atlas_uri):
    _prepare_collection(atlas_uri, "persist")
    writer = _store(atlas_uri, "persist")
    await _resilient_add(writer, ["apple fruit red"], [{"kind": "fruit"}])
    reader = _store(atlas_uri, "persist")
    results = await _search_until(reader, "apple", "apple fruit red", top_k=1)
    assert results[0]["text"] == "apple fruit red"


@pytest.mark.asyncio
async def test_add_empty_is_noop(atlas_uri):
    store = _store(atlas_uri, "noop")
    await store.add([])


@pytest.mark.asyncio
async def test_metadata_length_mismatch_raises(atlas_uri):
    store = _store(atlas_uri, "mismatch")
    with pytest.raises(ValueError):
        await store.add(["a", "b"], [{"only": "one"}])


def test_async_api_is_coroutine():
    # SynapseKit is async-first: the public IO surface must stay coroutines.
    store = MongoDBAtlasVectorStore(
        embedding_backend=KeywordEmbeddings(_VOCAB), uri="mongodb://localhost:27017"
    )
    assert inspect.iscoroutinefunction(store.add)
    assert inspect.iscoroutinefunction(store.search)
