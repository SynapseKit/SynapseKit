"""Tests for the 11 new vector store backends (all SDK/HTTP calls are mocked)."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embeddings(dim: int = 4):
    """Return a fake SynapsekitEmbeddings that yields deterministic vectors."""
    emb = MagicMock()
    emb.embed = AsyncMock(
        side_effect=lambda texts: np.ones((len(texts), dim), dtype=np.float32)
    )
    emb.embed_one = AsyncMock(
        return_value=np.ones(dim, dtype=np.float32)
    )
    return emb


def _run(coro):
    return asyncio.run(coro)


# ===========================================================================
# 1. VespaVectorStore
# ===========================================================================

class TestVespaVectorStore:
    def _make_store(self):
        mock_requests = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "root": {
                "children": [
                    {
                        "relevance": 0.9,
                        "fields": {"text": "hello world", "metadata": "{}"},
                    }
                ]
            }
        }
        mock_requests.post.return_value = mock_resp

        with patch.dict(sys.modules, {"requests": mock_requests}):
            sys.modules.pop("synapsekit.retrieval.vespa", None)
            from synapsekit.retrieval.vespa import VespaVectorStore

            store = VespaVectorStore(
                embedding_backend=_make_embeddings(),
                url="http://localhost:8080",
                application="app",
                schema="doc",
            )
            store._get_requests = lambda: mock_requests
            return store, mock_requests

    def test_add_and_search(self):
        store, mock_requests = self._make_store()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "root": {
                "children": [
                    {"relevance": 0.9, "fields": {"text": "hello", "metadata": "{}"}}
                ]
            }
        }
        mock_requests.post.return_value = mock_resp

        _run(store.add(["hello"], [{"src": "test"}]))
        results = _run(store.search("hello", top_k=1))

        assert mock_requests.post.called
        assert isinstance(results, list)
        assert results[0]["text"] == "hello"
        assert results[0]["score"] == pytest.approx(0.9)

    def test_add_empty(self):
        store, _ = self._make_store()
        _run(store.add([]))  # must not raise

    def test_metadata_mismatch_raises(self):
        store, _ = self._make_store()
        with pytest.raises(ValueError):
            _run(store.add(["a", "b"], [{"x": 1}]))

    def test_import_error(self):
        with patch.dict(sys.modules, {"requests": None}):
            # Remove cached module so __init__ triggers the ImportError
            sys.modules.pop("synapsekit.retrieval.vespa", None)
            import importlib as _il

            with pytest.raises(ImportError, match="pip install synapsekit\\[vespa\\]"):
                mod = _il.import_module("synapsekit.retrieval.vespa")
                mod.VespaVectorStore(embedding_backend=_make_embeddings())

    def test_metadata_filter(self):
        store, mock_requests = self._make_store()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "root": {
                "children": [
                    {"relevance": 0.8, "fields": {"text": "doc1", "metadata": '{"cat":"a"}'}},
                    {"relevance": 0.7, "fields": {"text": "doc2", "metadata": '{"cat":"b"}'}},
                ]
            }
        }
        mock_requests.post.return_value = mock_resp
        results = _run(store.search("q", top_k=5, metadata_filter={"cat": "a"}))
        assert all(r["metadata"].get("cat") == "a" for r in results)


# ===========================================================================
# 2. RedisVectorStore
# ===========================================================================

class TestRedisVectorStore:
    def _make_store(self):
        mock_redis_mod = MagicMock()
        mock_client = MagicMock()
        mock_redis_mod.from_url.return_value = mock_client
        # FT.INFO raises to simulate missing index
        mock_client.execute_command.side_effect = [
            Exception("no index"),  # FT.INFO call
            None,  # FT.CREATE
        ]
        mock_client.pipeline.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client.pipeline.return_value.__exit__ = MagicMock(return_value=False)
        pipe = MagicMock()
        pipe.execute.return_value = []
        mock_client.pipeline.return_value = pipe

        with patch.dict(sys.modules, {"redis": mock_redis_mod}):
            sys.modules.pop("synapsekit.retrieval.redis_vector", None)
            from synapsekit.retrieval.redis_vector import RedisVectorStore

            store = RedisVectorStore(
                embedding_backend=_make_embeddings(),
                url="redis://localhost:6379",
                index_name="test_idx",
            )
            store._client = mock_client
            return store, mock_client

    def test_add_creates_index_and_stores(self):
        store, mock_client = self._make_store()
        # Reset side_effect so FT.INFO raises (no index) then FT.CREATE succeeds
        mock_client.execute_command.side_effect = [
            Exception("no index"),
            None,
        ]
        pipe = MagicMock()
        pipe.execute.return_value = [1]
        mock_client.pipeline.return_value = pipe

        _run(store.add(["hello vespa"], [{"k": "v"}]))
        assert mock_client.execute_command.called

    def test_search_empty_when_no_dim(self):
        store, _ = self._make_store()
        store._dim = None
        results = _run(store.search("anything"))
        assert results == []

    def test_import_error(self):
        with patch.dict(sys.modules, {"redis": None}):
            sys.modules.pop("synapsekit.retrieval.redis_vector", None)
            with pytest.raises(ImportError, match="pip install synapsekit\\[redis-vector\\]"):
                import importlib as _il

                mod = _il.import_module("synapsekit.retrieval.redis_vector")
                mod.RedisVectorStore(embedding_backend=_make_embeddings())

    def test_add_empty(self):
        store, _ = self._make_store()
        _run(store.add([]))

    def test_metadata_mismatch_raises(self):
        store, _ = self._make_store()
        with pytest.raises(ValueError):
            _run(store.add(["a", "b"], [{}]))


# ===========================================================================
# 3. ElasticsearchVectorStore
# ===========================================================================

class TestElasticsearchVectorStore:
    def _make_store(self):
        mock_es_mod = MagicMock()
        mock_es_client = MagicMock()
        mock_es_mod.Elasticsearch.return_value = mock_es_client
        mock_es_client.indices.exists.return_value = False
        mock_es_client.indices.create.return_value = {}
        mock_es_client.index.return_value = {"result": "created"}
        mock_es_client.indices.refresh.return_value = {}
        mock_es_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {"text": "doc1", "metadata": {"k": "v"}},
                        "_score": 0.95,
                    }
                ]
            }
        }

        with patch.dict(sys.modules, {"elasticsearch": mock_es_mod}):
            sys.modules.pop("synapsekit.retrieval.elasticsearch_vector", None)
            from synapsekit.retrieval.elasticsearch_vector import ElasticsearchVectorStore

            store = ElasticsearchVectorStore(
                embedding_backend=_make_embeddings(),
                url="http://localhost:9200",
                index_name="test_idx",
            )
            store._es = mock_es_client
            return store, mock_es_client

    def test_add_and_search(self):
        store, mock_es = self._make_store()
        _run(store.add(["doc1"], [{"k": "v"}]))
        assert mock_es.indices.create.called or mock_es.indices.exists.called

        results = _run(store.search("query"))
        assert results[0]["text"] == "doc1"
        assert results[0]["score"] == pytest.approx(0.95)

    def test_search_returns_empty_before_add(self):
        store, _ = self._make_store()
        store._index_created = False
        store._dims = None
        results = _run(store.search("q"))
        assert results == []

    def test_import_error(self):
        with patch.dict(sys.modules, {"elasticsearch": None}):
            sys.modules.pop("synapsekit.retrieval.elasticsearch_vector", None)
            with pytest.raises(
                ImportError, match="pip install synapsekit\\[elasticsearch-vector\\]"
            ):
                import importlib as _il

                mod = _il.import_module("synapsekit.retrieval.elasticsearch_vector")
                mod.ElasticsearchVectorStore(embedding_backend=_make_embeddings())

    def test_add_empty(self):
        store, _ = self._make_store()
        _run(store.add([]))

    def test_metadata_filter(self):
        store, mock_es = self._make_store()
        store._index_created = True
        store._dims = 4
        mock_es.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"text": "a", "metadata": {"cat": "x"}}, "_score": 0.9},
                    {"_source": {"text": "b", "metadata": {"cat": "y"}}, "_score": 0.8},
                ]
            }
        }
        results = _run(store.search("q", metadata_filter={"cat": "x"}))
        assert len(results) == 1
        assert results[0]["metadata"]["cat"] == "x"


# ===========================================================================
# 4. OpenSearchVectorStore
# ===========================================================================

class TestOpenSearchVectorStore:
    def _make_store(self):
        mock_os_mod = MagicMock()
        mock_os_client = MagicMock()
        mock_os_mod.OpenSearch.return_value = mock_os_client
        mock_os_client.indices.exists.return_value = False
        mock_os_client.indices.create.return_value = {}
        mock_os_client.index.return_value = {"result": "created"}
        mock_os_client.indices.refresh.return_value = {}
        mock_os_client.search.return_value = {
            "hits": {"hits": [{"_source": {"text": "t", "metadata": {}}, "_score": 0.7}]}
        }

        with patch.dict(sys.modules, {"opensearchpy": mock_os_mod}):
            sys.modules.pop("synapsekit.retrieval.opensearch_vector", None)
            from synapsekit.retrieval.opensearch_vector import OpenSearchVectorStore

            store = OpenSearchVectorStore(
                embedding_backend=_make_embeddings(),
                url="http://localhost:9200",
                index_name="test_idx",
            )
            store._os = mock_os_client
            return store, mock_os_client

    def test_add_and_search(self):
        store, mock_os = self._make_store()
        _run(store.add(["t"]))
        results = _run(store.search("q"))
        assert results[0]["text"] == "t"

    def test_search_empty_before_add(self):
        store, _ = self._make_store()
        store._index_created = False
        store._dims = None
        assert _run(store.search("q")) == []

    def test_import_error(self):
        with patch.dict(sys.modules, {"opensearchpy": None}):
            sys.modules.pop("synapsekit.retrieval.opensearch_vector", None)
            with pytest.raises(ImportError, match="pip install synapsekit\\[opensearch\\]"):
                import importlib as _il

                mod = _il.import_module("synapsekit.retrieval.opensearch_vector")
                mod.OpenSearchVectorStore(embedding_backend=_make_embeddings())

    def test_add_empty(self):
        store, _ = self._make_store()
        _run(store.add([]))

    def test_metadata_mismatch(self):
        store, _ = self._make_store()
        with pytest.raises(ValueError):
            _run(store.add(["a", "b"], [{}]))


# ===========================================================================
# 5. SupabaseVectorStore
# ===========================================================================

class TestSupabaseVectorStore:
    def _make_store(self):
        mock_supabase_mod = MagicMock()
        mock_sb_client = MagicMock()
        mock_supabase_mod.create_client.return_value = mock_sb_client

        insert_chain = MagicMock()
        insert_chain.execute.return_value = MagicMock(data=[])
        mock_sb_client.table.return_value.insert.return_value = insert_chain

        rpc_chain = MagicMock()
        rpc_chain.execute.return_value = MagicMock(
            data=[{"content": "doc", "metadata": {}, "similarity": 0.85}]
        )
        mock_sb_client.rpc.return_value = rpc_chain

        with patch.dict(sys.modules, {"supabase": mock_supabase_mod}):
            sys.modules.pop("synapsekit.retrieval.supabase_vector", None)
            from synapsekit.retrieval.supabase_vector import SupabaseVectorStore

            store = SupabaseVectorStore(
                embedding_backend=_make_embeddings(),
                url="https://proj.supabase.co",
                key="anon-key",
                table_name="docs",
            )
            store._client = mock_sb_client
            return store, mock_sb_client

    def test_add_and_search(self):
        store, mock_sb = self._make_store()
        _run(store.add(["doc"], [{}]))
        assert mock_sb.table.called

        results = _run(store.search("q"))
        assert results[0]["text"] == "doc"
        assert results[0]["score"] == pytest.approx(0.85)

    def test_import_error(self):
        with patch.dict(sys.modules, {"supabase": None}):
            sys.modules.pop("synapsekit.retrieval.supabase_vector", None)
            with pytest.raises(ImportError, match="pip install synapsekit\\[supabase-vector\\]"):
                import importlib as _il

                mod = _il.import_module("synapsekit.retrieval.supabase_vector")
                mod.SupabaseVectorStore(
                    embedding_backend=_make_embeddings(),
                    url="https://x.supabase.co",
                    key="k",
                )

    def test_add_empty(self):
        store, _ = self._make_store()
        _run(store.add([]))

    def test_metadata_mismatch(self):
        store, _ = self._make_store()
        with pytest.raises(ValueError):
            _run(store.add(["a", "b"], [{}]))

    def test_metadata_filter(self):
        store, mock_sb = self._make_store()
        rpc_chain = MagicMock()
        rpc_chain.execute.return_value = MagicMock(
            data=[
                {"content": "a", "metadata": {"cat": "x"}, "similarity": 0.9},
                {"content": "b", "metadata": {"cat": "y"}, "similarity": 0.8},
            ]
        )
        mock_sb.rpc.return_value = rpc_chain
        results = _run(store.search("q", metadata_filter={"cat": "x"}))
        assert len(results) == 1


# ===========================================================================
# 6. TypesenseVectorStore
# ===========================================================================

class TestTypesenseVectorStore:
    def _make_store(self):
        mock_ts_mod = MagicMock()
        mock_ts_client = MagicMock()
        mock_ts_mod.Client.return_value = mock_ts_client

        # collection retrieve raises (doesn't exist) then create succeeds
        mock_ts_client.collections.__getitem__.return_value.retrieve.side_effect = Exception(
            "not found"
        )
        mock_ts_client.collections.create.return_value = {}
        mock_ts_client.collections.__getitem__.return_value.documents.import_.return_value = []

        mock_ts_client.multi_search.perform.return_value = {
            "results": [
                {
                    "hits": [
                        {
                            "document": {"text": "hi", "metadata": "{}"},
                            "vector_distance": 0.1,
                        }
                    ]
                }
            ]
        }

        with patch.dict(sys.modules, {"typesense": mock_ts_mod}):
            sys.modules.pop("synapsekit.retrieval.typesense_vector", None)
            from synapsekit.retrieval.typesense_vector import TypesenseVectorStore

            store = TypesenseVectorStore(
                embedding_backend=_make_embeddings(),
                host="localhost",
                port=8108,
                api_key="xyz",
                collection_name="test_col",
            )
            store._ts = mock_ts_client
            return store, mock_ts_client

    def test_add_and_search(self):
        store, mock_ts = self._make_store()
        # Force _ensure_collection to recreate
        store._collection_created = False
        mock_ts.collections.__getitem__.return_value.retrieve.side_effect = Exception("not found")
        mock_ts.collections.create.return_value = {}

        _run(store.add(["hi"], [{}]))
        assert mock_ts.collections.create.called or store._collection_created

    def test_search_empty_before_add(self):
        store, _ = self._make_store()
        store._dim = None
        assert _run(store.search("q")) == []

    def test_import_error(self):
        with patch.dict(sys.modules, {"typesense": None}):
            sys.modules.pop("synapsekit.retrieval.typesense_vector", None)
            with pytest.raises(ImportError, match="pip install synapsekit\\[typesense\\]"):
                import importlib as _il

                mod = _il.import_module("synapsekit.retrieval.typesense_vector")
                mod.TypesenseVectorStore(embedding_backend=_make_embeddings())

    def test_add_empty(self):
        store, _ = self._make_store()
        _run(store.add([]))

    def test_metadata_mismatch(self):
        store, _ = self._make_store()
        with pytest.raises(ValueError):
            _run(store.add(["a", "b"], [{}]))


# ===========================================================================
# 7. MarqoVectorStore
# ===========================================================================

class TestMarqoVectorStore:
    def _make_store(self):
        mock_marqo_mod = MagicMock()
        mock_mq = MagicMock()
        mock_marqo_mod.Client.return_value = mock_mq

        mock_mq.index.return_value.get_stats.side_effect = Exception("not found")
        mock_mq.create_index.return_value = {}
        mock_mq.index.return_value.add_documents.return_value = {}
        mock_mq.index.return_value.search.return_value = {
            "hits": [{"text": "hello", "_score": 0.9, "_id": "1"}]
        }

        with patch.dict(sys.modules, {"marqo": mock_marqo_mod}):
            sys.modules.pop("synapsekit.retrieval.marqo_vector", None)
            from synapsekit.retrieval.marqo_vector import MarqoVectorStore

            store = MarqoVectorStore(url="http://localhost:8882", index_name="idx")
            store._mq = mock_mq
            return store, mock_mq

    def test_add_and_search(self):
        store, mock_mq = self._make_store()
        mock_mq.index.return_value.get_stats.side_effect = Exception("not found")

        _run(store.add(["hello"]))
        results = _run(store.search("hello"))
        assert results[0]["text"] == "hello"
        assert results[0]["score"] == pytest.approx(0.9)

    def test_import_error(self):
        with patch.dict(sys.modules, {"marqo": None}):
            sys.modules.pop("synapsekit.retrieval.marqo_vector", None)
            with pytest.raises(ImportError, match="pip install synapsekit\\[marqo\\]"):
                import importlib as _il

                mod = _il.import_module("synapsekit.retrieval.marqo_vector")
                mod.MarqoVectorStore()

    def test_add_empty(self):
        store, _ = self._make_store()
        _run(store.add([]))

    def test_metadata_mismatch(self):
        store, _ = self._make_store()
        with pytest.raises(ValueError):
            _run(store.add(["a", "b"], [{}]))

    def test_metadata_filter(self):
        store, mock_mq = self._make_store()
        store._index_created = True
        mock_mq.index.return_value.search.return_value = {
            "hits": [
                {"text": "a", "_score": 0.9, "_id": "1", "cat": "x"},
                {"text": "b", "_score": 0.8, "_id": "2", "cat": "y"},
            ]
        }
        results = _run(store.search("q", metadata_filter={"cat": "x"}))
        assert len(results) == 1
        assert results[0]["metadata"]["cat"] == "x"


# ===========================================================================
# 8. ZillizVectorStore
# ===========================================================================

class TestZillizVectorStore:
    def _make_store(self):
        mock_pymilvus_mod = MagicMock()
        mock_client = MagicMock()
        mock_pymilvus_mod.MilvusClient.return_value = mock_client
        mock_client.has_collection.return_value = False
        mock_client.create_collection.return_value = {}
        mock_client.insert.return_value = {}
        mock_client.search.return_value = [
            [{"entity": {"text": "doc", "metadata": "{}"}, "distance": 0.95}]
        ]

        with patch.dict(sys.modules, {"pymilvus": mock_pymilvus_mod}):
            sys.modules.pop("synapsekit.retrieval.zilliz_vector", None)
            from synapsekit.retrieval.zilliz_vector import ZillizVectorStore

            store = ZillizVectorStore(
                embedding_backend=_make_embeddings(),
                uri="https://my.zilliz.com",
                token="tok",
                collection_name="col",
            )
            store._client = mock_client
            return store, mock_client

    def test_add_and_search(self):
        store, mock_client = self._make_store()
        _run(store.add(["doc"], [{}]))
        assert mock_client.insert.called

        results = _run(store.search("q"))
        assert results[0]["text"] == "doc"

    def test_search_empty_before_add(self):
        store, _ = self._make_store()
        store._collection_created = False
        store._dim = None
        assert _run(store.search("q")) == []

    def test_import_error(self):
        with patch.dict(sys.modules, {"pymilvus": None}):
            sys.modules.pop("synapsekit.retrieval.zilliz_vector", None)
            with pytest.raises(ImportError, match="pip install synapsekit\\[zilliz\\]"):
                import importlib as _il

                mod = _il.import_module("synapsekit.retrieval.zilliz_vector")
                mod.ZillizVectorStore(
                    embedding_backend=_make_embeddings(), uri="u", token="t"
                )

    def test_add_empty(self):
        store, _ = self._make_store()
        _run(store.add([]))

    def test_metadata_mismatch(self):
        store, _ = self._make_store()
        with pytest.raises(ValueError):
            _run(store.add(["a", "b"], [{}]))


# ===========================================================================
# 9. DuckDBVectorStore
# ===========================================================================

class TestDuckDBVectorStore:
    def _make_store(self):
        mock_duckdb_mod = MagicMock()
        mock_conn = MagicMock()
        mock_duckdb_mod.connect.return_value = mock_conn
        # VSS install should silently pass
        mock_conn.execute.return_value = mock_conn

        with patch.dict(sys.modules, {"duckdb": mock_duckdb_mod}):
            sys.modules.pop("synapsekit.retrieval.duckdb_vector", None)
            from synapsekit.retrieval.duckdb_vector import DuckDBVectorStore

            store = DuckDBVectorStore(
                embedding_backend=_make_embeddings(),
                db_path=":memory:",
                table_name="vecs",
            )
            store._conn = mock_conn
            return store, mock_conn

    def test_add_and_search(self):
        store, mock_conn = self._make_store()
        mock_conn.execute.return_value = mock_conn
        _run(store.add(["doc1"], [{"k": "v"}]))
        assert mock_conn.execute.called

        mock_conn.execute.return_value.fetchall.return_value = [
            ("doc1", '{"k":"v"}', 0.99)
        ]
        store._table_created = True
        results = _run(store.search("q"))
        assert results[0]["text"] == "doc1"
        assert results[0]["score"] == pytest.approx(0.99)

    def test_search_empty_before_add(self):
        store, _ = self._make_store()
        store._table_created = False
        assert _run(store.search("q")) == []

    def test_import_error(self):
        with patch.dict(sys.modules, {"duckdb": None}):
            sys.modules.pop("synapsekit.retrieval.duckdb_vector", None)
            with pytest.raises(ImportError, match="pip install synapsekit\\[duckdb-vector\\]"):
                import importlib as _il

                mod = _il.import_module("synapsekit.retrieval.duckdb_vector")
                mod.DuckDBVectorStore(embedding_backend=_make_embeddings())

    def test_add_empty(self):
        store, _ = self._make_store()
        _run(store.add([]))

    def test_metadata_filter(self):
        store, mock_conn = self._make_store()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("a", '{"cat":"x"}', 0.9),
            ("b", '{"cat":"y"}', 0.8),
        ]
        store._table_created = True
        results = _run(store.search("q", metadata_filter={"cat": "x"}))
        assert len(results) == 1
        assert results[0]["metadata"]["cat"] == "x"


# ===========================================================================
# 10. ClickHouseVectorStore
# ===========================================================================

class TestClickHouseVectorStore:
    def _make_store(self):
        mock_ch_mod = MagicMock()
        mock_ch_client = MagicMock()
        mock_ch_mod.get_client.return_value = mock_ch_client
        mock_ch_client.command.return_value = None
        mock_ch_client.insert.return_value = None
        mock_result = MagicMock()
        mock_result.result_rows = [("doc1", '{"k":"v"}', 0.5)]
        mock_ch_client.query.return_value = mock_result

        with patch.dict(sys.modules, {"clickhouse_connect": mock_ch_mod}):
            sys.modules.pop("synapsekit.retrieval.clickhouse_vector", None)
            from synapsekit.retrieval.clickhouse_vector import ClickHouseVectorStore

            store = ClickHouseVectorStore(
                embedding_backend=_make_embeddings(),
                host="localhost",
                port=8123,
                database="default",
                table_name="vecs",
            )
            store._client = mock_ch_client
            return store, mock_ch_client

    def test_add_and_search(self):
        store, mock_ch = self._make_store()
        _run(store.add(["doc1"], [{"k": "v"}]))
        assert mock_ch.insert.called

        results = _run(store.search("q"))
        assert results[0]["text"] == "doc1"
        assert results[0]["score"] == pytest.approx(0.5)

    def test_search_empty_before_add(self):
        store, _ = self._make_store()
        store._table_created = False
        assert _run(store.search("q")) == []

    def test_import_error(self):
        with patch.dict(sys.modules, {"clickhouse_connect": None}):
            sys.modules.pop("synapsekit.retrieval.clickhouse_vector", None)
            with pytest.raises(ImportError, match="pip install synapsekit\\[clickhouse\\]"):
                import importlib as _il

                mod = _il.import_module("synapsekit.retrieval.clickhouse_vector")
                mod.ClickHouseVectorStore(embedding_backend=_make_embeddings())

    def test_add_empty(self):
        store, _ = self._make_store()
        _run(store.add([]))

    def test_metadata_mismatch(self):
        store, _ = self._make_store()
        with pytest.raises(ValueError):
            _run(store.add(["a", "b"], [{}]))

    def test_metadata_filter(self):
        store, mock_ch = self._make_store()
        mock_result = MagicMock()
        mock_result.result_rows = [
            ("a", '{"cat":"x"}', 0.2),
            ("b", '{"cat":"y"}', 0.4),
        ]
        mock_ch.query.return_value = mock_result
        store._table_created = True
        results = _run(store.search("q", metadata_filter={"cat": "x"}))
        assert len(results) == 1


# ===========================================================================
# 11. CassandraVectorStore (cassandra-driver mode)
# ===========================================================================

class TestCassandraVectorStore:
    def _make_store(self):
        mock_cassandra_mod = MagicMock()
        mock_session = MagicMock()

        # simulate prepare + execute
        mock_stmt = MagicMock()
        mock_session.prepare.return_value = mock_stmt
        mock_session.execute.return_value = []

        # Patch cassandra.cluster.Cluster
        mock_cluster_mod = MagicMock()
        mock_cluster = MagicMock()
        mock_cluster.connect.return_value = mock_session
        mock_cluster_mod.Cluster.return_value = mock_cluster

        with patch.dict(
            sys.modules,
            {
                "cassandra": mock_cassandra_mod,
                "cassandra.cluster": mock_cluster_mod,
            },
        ):
            sys.modules.pop("synapsekit.retrieval.cassandra_vector", None)
            from synapsekit.retrieval.cassandra_vector import CassandraVectorStore

            store = CassandraVectorStore(
                embedding_backend=_make_embeddings(),
                keyspace="ks",
                table_name="vecs",
                session=mock_session,
            )
            return store, mock_session

    def test_add_stores_docs(self):
        store, mock_session = self._make_store()
        _run(store.add(["hello"], [{"x": 1}]))
        assert mock_session.execute.called

    def test_search_returns_empty_before_add(self):
        store, _ = self._make_store()
        store._table_created = False
        store._dim = None
        assert _run(store.search("q")) == []

    def test_add_empty(self):
        store, _ = self._make_store()
        _run(store.add([]))

    def test_metadata_mismatch(self):
        store, _ = self._make_store()
        with pytest.raises(ValueError):
            _run(store.add(["a", "b"], [{}]))

    def test_import_error_cassandra(self):
        with patch.dict(sys.modules, {"cassandra": None, "cassandra.cluster": None}):
            sys.modules.pop("synapsekit.retrieval.cassandra_vector", None)
            with pytest.raises(ImportError, match="pip install synapsekit\\[cassandra\\]"):
                import importlib as _il

                mod = _il.import_module("synapsekit.retrieval.cassandra_vector")
                mod.CassandraVectorStore(
                    embedding_backend=_make_embeddings(), keyspace="ks"
                )

    def test_search_with_results(self):
        store, mock_session = self._make_store()
        row = MagicMock()
        row.text = "doc"
        row.metadata = '{"cat":"z"}'
        mock_session.execute.return_value = [row]
        store._table_created = True
        store._dim = 4

        results = _run(store.search("q"))
        assert results[0]["text"] == "doc"
        assert results[0]["metadata"]["cat"] == "z"


# ===========================================================================
# __init__.py exports
# ===========================================================================

class TestInitExports:
    def test_all_new_stores_in_all(self):
        import synapsekit.retrieval as r

        expected = [
            "VespaVectorStore",
            "RedisVectorStore",
            "ElasticsearchVectorStore",
            "OpenSearchVectorStore",
            "SupabaseVectorStore",
            "TypesenseVectorStore",
            "MarqoVectorStore",
            "ZillizVectorStore",
            "DuckDBVectorStore",
            "ClickHouseVectorStore",
            "CassandraVectorStore",
        ]
        for name in expected:
            assert name in r.__all__, f"{name} missing from __all__"

    def test_backends_map_present(self):
        import synapsekit.retrieval as r

        expected_keys = [
            "VespaVectorStore",
            "RedisVectorStore",
            "ElasticsearchVectorStore",
            "OpenSearchVectorStore",
            "SupabaseVectorStore",
            "TypesenseVectorStore",
            "MarqoVectorStore",
            "ZillizVectorStore",
            "DuckDBVectorStore",
            "ClickHouseVectorStore",
            "CassandraVectorStore",
        ]
        for key in expected_keys:
            assert key in r._BACKENDS, f"{key} missing from _BACKENDS"
