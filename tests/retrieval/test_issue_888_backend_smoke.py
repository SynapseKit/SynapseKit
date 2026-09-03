"""Provider-independent smoke coverage for issue #888 adapters."""

from __future__ import annotations

import sys
import types

import pytest


class Embeddings:
    async def embed(self, texts):
        return [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts]

    async def embed_one(self, text):
        return [1.0, 0.0] if "alpha" in text else [0.0, 1.0]


@pytest.mark.asyncio
async def test_turbopuffer_add_search_and_filter():
    from synapsekit.retrieval.turbopuffer import TurbopufferVectorStore

    class Namespace:
        def write(self, **kwargs):
            self.rows = kwargs["upsert_rows"]

        def query(self, **_kwargs):
            return {
                "rows": [
                    {
                        "distance": 0.9,
                        "attributes": {
                            "text": "alpha",
                            "metadata": '{"kind": "a"}',
                        },
                    }
                ]
            }

    class Client:
        def __init__(self):
            self.ns = Namespace()

        def namespace(self, _name):
            return self.ns

    store = TurbopufferVectorStore(Embeddings(), client=Client())
    await store.add(["alpha"], [{"kind": "a"}])
    results = await store.search("alpha", metadata_filter={"kind": "a"})
    assert results == [{"text": "alpha", "score": 0.9, "metadata": {"kind": "a"}}]


@pytest.mark.asyncio
async def test_vertex_add_and_search_resolves_datapoint_ids(tmp_path):
    from synapsekit.retrieval.vertex_ai_vector import VertexAIVectorStore

    class Index:
        def upsert_datapoints(self, datapoints):
            self.datapoints = datapoints

    class Endpoint:
        def find_neighbors(self, **_kwargs):
            return [[{"id": self.datapoint_id, "distance": 0.8}]]

    index = Index()
    endpoint = Endpoint()
    document_store_path = tmp_path / "vertex-documents.json"
    store = VertexAIVectorStore(
        Embeddings(),
        index_endpoint=endpoint,
        index=index,
        deployed_index_id="deployed",
        document_store_path=str(document_store_path),
    )
    await store.add(["alpha"], [{"kind": "a"}])
    endpoint.datapoint_id = index.datapoints[0]["datapoint_id"]
    assert (await store.search("alpha"))[0]["text"] == "alpha"

    restored = VertexAIVectorStore(
        Embeddings(),
        index_endpoint=endpoint,
        index=Index(),
        deployed_index_id="deployed",
        document_store_path=str(document_store_path),
    )
    assert (await restored.search("alpha"))[0]["metadata"] == {"kind": "a"}


@pytest.mark.asyncio
async def test_azure_ai_search_add_and_search(monkeypatch):
    from synapsekit.retrieval.azure_ai_search import AzureAISearchVectorStore

    class VectorizedQuery:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    azure = types.ModuleType("azure")
    azure.__path__ = []
    search = types.ModuleType("azure.search")
    search.__path__ = []
    documents = types.ModuleType("azure.search.documents")
    documents.__path__ = []
    models = types.ModuleType("azure.search.documents.models")
    models.VectorizedQuery = VectorizedQuery
    monkeypatch.setitem(sys.modules, "azure", azure)
    monkeypatch.setitem(sys.modules, "azure.search", search)
    monkeypatch.setitem(sys.modules, "azure.search.documents", documents)
    monkeypatch.setitem(sys.modules, "azure.search.documents.models", models)

    class IndexClient:
        def get_index(self, _name):
            return object()

    class SearchClient:
        def upload_documents(self, documents):
            self.documents = documents

        def search(self, **_kwargs):
            return [{"text": "alpha", "metadata": '{"kind": "a"}', "@search.score": 0.9}]

    search_client = SearchClient()
    store = AzureAISearchVectorStore(
        Embeddings(),
        search_client=search_client,
        index_client=IndexClient(),
    )
    await store.add(["alpha"], [{"kind": "a"}])
    assert (await store.search("alpha"))[0]["metadata"] == {"kind": "a"}


class _Cursor:
    def __init__(self, database):
        self.database = database
        self.description = None

    def execute(self, statement, _params):
        self.database.statements.append(statement)
        if statement.lstrip().upper().startswith("SELECT"):
            self.description = [("text",), ("metadata",), ("score",)]

    def fetchall(self):
        return self.database.rows

    def close(self):
        pass


class _Database:
    def __init__(self, rows):
        self.rows = rows
        self.statements = []

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        pass


@pytest.mark.asyncio
async def test_sql_backends_parameterize_limit_and_round_trip_metadata():
    from synapsekit.retrieval.singlestore_vector import SingleStoreVectorStore
    from synapsekit.retrieval.tidb_vector import TiDBVectorStore

    for backend in (SingleStoreVectorStore, TiDBVectorStore):
        database = _Database([("alpha", '{"kind": "a"}', 0.9)])
        store = backend(Embeddings(), connection=database)
        await store.add(["alpha"], [{"kind": "a"}])
        results = await store.search("alpha", metadata_filter={"kind": "a"})
        assert results[0]["metadata"] == {"kind": "a"}
        assert any("LIMIT %s" in statement for statement in database.statements)


@pytest.mark.asyncio
async def test_surrealdb_add_and_search():
    from synapsekit.retrieval.surrealdb_vector import SurrealDBVectorStore

    class Client:
        def __init__(self):
            self.created = []

        def query(self, statement, _params=None):
            if statement.startswith("DEFINE"):
                return []
            return [{"result": [{"text": "alpha", "metadata": {"kind": "a"}, "score": 0.9}]}]

        def create(self, _table, data):
            self.created.append(data)

    client = Client()
    store = SurrealDBVectorStore(Embeddings(), client=client)
    await store.add(["alpha"], [{"kind": "a"}])
    assert (await store.search("alpha"))[0]["text"] == "alpha"
    assert client.created[0]["metadata"] == {"kind": "a"}


@pytest.mark.asyncio
async def test_couchbase_add_and_search():
    from synapsekit.retrieval.couchbase_vector import CouchbaseVectorStore

    search_module = types.ModuleType("couchbase.search")
    vector_module = types.ModuleType("couchbase.vector_search")
    options_module = types.ModuleType("couchbase.options")

    class SearchOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    options_module.SearchOptions = SearchOptions

    class VectorQuery:
        @classmethod
        def create(cls, *_args, **_kwargs):
            return cls()

    class VectorSearch:
        def __init__(self, query):
            self.query = query

    class SearchRequest:
        @classmethod
        def create(cls, query):
            instance = cls()
            instance.query = query
            return instance

    search_module.SearchRequest = SearchRequest
    vector_module.VectorQuery = VectorQuery
    vector_module.VectorSearch = VectorSearch
    couchbase_module = types.ModuleType("couchbase")
    couchbase_module.__path__ = []
    search_module.__package__ = "couchbase"
    vector_module.__package__ = "couchbase"

    class Row:
        # Real couchbase 4.x SearchRow exposes ``fields`` as a dict attribute,
        # populated only when SearchOptions(fields=...) is requested.
        score = 0.9
        fields = {"text": "alpha", "metadata": {"kind": "a"}}

    class Scope:
        def __init__(self):
            self.options = None

        def search(self, _index, _request, options=None):
            self.options = options

            class Result:
                def rows(self):
                    return [Row()]

            return Result()

    class Collection:
        def upsert(self, _key, _value):
            pass

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "couchbase", couchbase_module)
    monkeypatch.setitem(sys.modules, "couchbase.search", search_module)
    monkeypatch.setitem(sys.modules, "couchbase.vector_search", vector_module)
    monkeypatch.setitem(sys.modules, "couchbase.options", options_module)
    try:
        scope = Scope()
        store = CouchbaseVectorStore(
            Embeddings(), cluster=object(), collection=Collection(), search_scope=scope
        )
        await store.add(["alpha"], [{"kind": "a"}])
        result = (await store.search("alpha"))[0]
        assert result["text"] == "alpha"
        assert result["metadata"] == {"kind": "a"}
        # Regression: fields must be requested or the real SDK returns no
        # document text/metadata (only id + score).
        assert scope.options is not None
        assert scope.options.kwargs.get("fields") == ["*"]
    finally:
        monkeypatch.undo()


@pytest.mark.asyncio
async def test_deeplake_and_myscale_smoke():
    from synapsekit.retrieval.deeplake import DeepLakeVectorStore
    from synapsekit.retrieval.myscale_vector import MyScaleVectorStore

    class Dataset:
        def append(self, _rows):
            pass

        def commit(self):
            pass

        def search(self, **_kwargs):
            return {"text": ["alpha"], "metadata": [{"kind": "a"}], "score": [0.9]}

    deep_lake = DeepLakeVectorStore(Embeddings(), dataset=Dataset())
    await deep_lake.add(["alpha"], [{"kind": "a"}])
    assert (await deep_lake.search("alpha"))[0]["text"] == "alpha"

    class Result:
        result_rows = [("alpha", '{"kind": "a"}', 0.9)]

    class Client:
        def command(self, statement):
            return 1 if statement.startswith("EXISTS") else None

        def insert(self, *_args, **_kwargs):
            pass

        def query(self, *_args, **_kwargs):
            return Result()

    myscale = MyScaleVectorStore(Embeddings(), client=Client())
    await myscale.add(["alpha"], [{"kind": "a"}])
    assert (await myscale.search("alpha"))[0]["metadata"] == {"kind": "a"}
