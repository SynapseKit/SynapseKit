"""Tests for v0.6.5 features: 3 tools, 3 retrieval strategies, 1 cache backend, 1 memory type."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ------------------------------------------------------------------ #
# Version
# ------------------------------------------------------------------ #


def test_version():
    import synapsekit

    assert synapsekit.__version__ == "0.6.6"


# ------------------------------------------------------------------ #
# Feature 1: TokenBufferMemory
# ------------------------------------------------------------------ #


class TestTokenBufferMemory:
    def test_add_and_get(self):
        from synapsekit.memory.token_buffer import TokenBufferMemory

        mem = TokenBufferMemory(max_tokens=4000)
        mem.add("user", "Hello")
        mem.add("assistant", "Hi there!")
        msgs = mem.get_messages()
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "Hello"}
        assert msgs[1] == {"role": "assistant", "content": "Hi there!"}

    def test_trimming(self):
        from synapsekit.memory.token_buffer import TokenBufferMemory

        # chars_per_token=1 means 1 char = 1 token
        mem = TokenBufferMemory(max_tokens=10, chars_per_token=1)
        mem.add("user", "aaaaaaaaaa")  # 10 tokens, exactly at limit
        assert len(mem) == 1
        mem.add("user", "bbb")  # total would be 13, should trim oldest
        assert len(mem) == 1
        assert mem.get_messages()[0]["content"] == "bbb"

    def test_format_context(self):
        from synapsekit.memory.token_buffer import TokenBufferMemory

        mem = TokenBufferMemory()
        mem.add("user", "Hello")
        mem.add("assistant", "Hi")
        ctx = mem.format_context()
        assert "User: Hello" in ctx
        assert "Assistant: Hi" in ctx

    def test_clear(self):
        from synapsekit.memory.token_buffer import TokenBufferMemory

        mem = TokenBufferMemory()
        mem.add("user", "Hello")
        mem.clear()
        assert len(mem) == 0
        assert mem.get_messages() == []

    def test_len(self):
        from synapsekit.memory.token_buffer import TokenBufferMemory

        mem = TokenBufferMemory()
        assert len(mem) == 0
        mem.add("user", "Hello")
        assert len(mem) == 1

    def test_invalid_max_tokens(self):
        from synapsekit.memory.token_buffer import TokenBufferMemory

        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            TokenBufferMemory(max_tokens=0)

    def test_invalid_chars_per_token(self):
        from synapsekit.memory.token_buffer import TokenBufferMemory

        with pytest.raises(ValueError, match="chars_per_token must be >= 1"):
            TokenBufferMemory(chars_per_token=0)

    def test_estimate_tokens(self):
        from synapsekit.memory.token_buffer import TokenBufferMemory

        mem = TokenBufferMemory(chars_per_token=4)
        assert mem._estimate_tokens("abcdefgh") == 2  # 8 // 4

    def test_import_from_top_level(self):
        from synapsekit import TokenBufferMemory

        assert TokenBufferMemory is not None


# ------------------------------------------------------------------ #
# Feature 2: DuckDuckGoSearchTool
# ------------------------------------------------------------------ #


def _mock_ddgs_module(ddgs_instance):
    """Create a mock duckduckgo_search module with a DDGS class."""
    mod = MagicMock()
    mod.DDGS.return_value = ddgs_instance
    return mod


class TestDuckDuckGoSearchTool:
    @pytest.fixture
    def tool(self):
        from synapsekit.agents.tools.duck_search import DuckDuckGoSearchTool

        return DuckDuckGoSearchTool()

    async def test_text_search(self, tool):
        mock_results = [
            {"title": "Result 1", "href": "https://example.com/1", "body": "Snippet 1"},
            {"title": "Result 2", "href": "https://example.com/2", "body": "Snippet 2"},
        ]

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = mock_results

        with patch.dict("sys.modules", {"duckduckgo_search": _mock_ddgs_module(mock_ddgs)}):
            result = await tool.run(query="test query")

        assert not result.is_error
        assert "Result 1" in result.output
        assert "Result 2" in result.output

    async def test_news_search(self, tool):
        mock_results = [
            {"title": "News 1", "url": "https://news.com/1", "excerpt": "Excerpt 1"},
        ]

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.news.return_value = mock_results

        with patch.dict("sys.modules", {"duckduckgo_search": _mock_ddgs_module(mock_ddgs)}):
            result = await tool.run(query="test", search_type="news")

        assert not result.is_error
        assert "News 1" in result.output

    async def test_no_query_error(self, tool):
        result = await tool.run()
        assert result.is_error
        assert "No search query" in result.error

    async def test_empty_results(self, tool):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = []

        with patch.dict("sys.modules", {"duckduckgo_search": _mock_ddgs_module(mock_ddgs)}):
            result = await tool.run(query="obscure query")

        assert result.output == "No results found."

    async def test_import_error(self, tool):
        with patch.dict("sys.modules", {"duckduckgo_search": None}):
            with pytest.raises(ImportError, match="duckduckgo-search required"):
                await tool.run(query="test")

    def test_import_from_top_level(self):
        from synapsekit import DuckDuckGoSearchTool

        assert DuckDuckGoSearchTool is not None


# ------------------------------------------------------------------ #
# Feature 3: PDFReaderTool
# ------------------------------------------------------------------ #


def _mock_pypdf_module(reader_instance):
    """Create a mock pypdf module with a PdfReader class."""
    mod = MagicMock()
    mod.PdfReader.return_value = reader_instance
    return mod


class TestPDFReaderTool:
    @pytest.fixture
    def tool(self):
        from synapsekit.agents.tools.pdf_reader import PDFReaderTool

        return PDFReaderTool()

    async def test_read_all_pages(self, tool):
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2]

        with (
            patch("os.path.isfile", return_value=True),
            patch.dict("sys.modules", {"pypdf": _mock_pypdf_module(mock_reader)}),
        ):
            result = await tool.run(file_path="/tmp/test.pdf")

        assert not result.is_error
        assert "Page 1 content" in result.output
        assert "Page 2 content" in result.output
        assert "--- Page 1 ---" in result.output

    async def test_specific_pages(self, tool):
        mock_pages = [MagicMock() for _ in range(5)]
        for i, page in enumerate(mock_pages):
            page.extract_text.return_value = f"Content {i + 1}"

        mock_reader = MagicMock()
        mock_reader.pages = mock_pages

        with (
            patch("os.path.isfile", return_value=True),
            patch.dict("sys.modules", {"pypdf": _mock_pypdf_module(mock_reader)}),
        ):
            result = await tool.run(file_path="/tmp/test.pdf", page_numbers="1,3")

        assert not result.is_error
        assert "Content 1" in result.output
        assert "Content 3" in result.output
        assert "Content 2" not in result.output

    async def test_file_not_found(self, tool):
        mock_pypdf = _mock_pypdf_module(MagicMock())
        with (
            patch("os.path.isfile", return_value=False),
            patch.dict("sys.modules", {"pypdf": mock_pypdf}),
        ):
            result = await tool.run(file_path="/nonexistent.pdf")

        assert result.is_error
        assert "File not found" in result.error

    async def test_invalid_page_numbers(self, tool):
        mock_reader = MagicMock()
        mock_reader.pages = [MagicMock()]

        with (
            patch("os.path.isfile", return_value=True),
            patch.dict("sys.modules", {"pypdf": _mock_pypdf_module(mock_reader)}),
        ):
            result = await tool.run(file_path="/tmp/test.pdf", page_numbers="abc")

        assert result.is_error
        assert "Invalid page numbers" in result.error

    async def test_page_out_of_range(self, tool):
        mock_reader = MagicMock()
        mock_reader.pages = [MagicMock()]

        with (
            patch("os.path.isfile", return_value=True),
            patch.dict("sys.modules", {"pypdf": _mock_pypdf_module(mock_reader)}),
        ):
            result = await tool.run(file_path="/tmp/test.pdf", page_numbers="5")

        assert result.is_error
        assert "out of range" in result.error

    async def test_no_path_error(self, tool):
        result = await tool.run()
        assert result.is_error
        assert "No file path" in result.error

    def test_import_from_top_level(self):
        from synapsekit import PDFReaderTool

        assert PDFReaderTool is not None


# ------------------------------------------------------------------ #
# Feature 4: GraphQLTool
# ------------------------------------------------------------------ #


def _mock_aiohttp_module(mock_session):
    """Create a mock aiohttp module."""
    mod = MagicMock()
    mod.ClientSession.return_value = mock_session
    mod.ClientTimeout.return_value = MagicMock()
    return mod


class TestGraphQLTool:
    @pytest.fixture
    def tool(self):
        from synapsekit.agents.tools.graphql import GraphQLTool

        return GraphQLTool()

    def _make_session(self, response):
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        return mock_session

    def _make_response(self, status=200, json_data=None, text=""):
        mock_response = MagicMock()
        mock_response.status = status
        mock_response.json = AsyncMock(return_value=json_data or {})
        mock_response.text = AsyncMock(return_value=text)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        return mock_response

    async def test_basic_query(self, tool):
        resp = self._make_response(json_data={"data": {"user": {"name": "Alice"}}})
        session = self._make_session(resp)

        with patch.dict("sys.modules", {"aiohttp": _mock_aiohttp_module(session)}):
            result = await tool.run(
                url="https://api.example.com/graphql",
                query="{ user { name } }",
            )

        assert not result.is_error
        data = json.loads(result.output)
        assert data["data"]["user"]["name"] == "Alice"

    async def test_with_variables(self, tool):
        resp = self._make_response(json_data={"data": {"user": {"id": "1"}}})
        session = self._make_session(resp)

        with patch.dict("sys.modules", {"aiohttp": _mock_aiohttp_module(session)}):
            result = await tool.run(
                url="https://api.example.com/graphql",
                query="query($id: ID!) { user(id: $id) { id } }",
                variables='{"id": "1"}',
            )

        assert not result.is_error

    async def test_with_headers(self, tool):
        resp = self._make_response(json_data={"data": {}})
        session = self._make_session(resp)

        with patch.dict("sys.modules", {"aiohttp": _mock_aiohttp_module(session)}):
            result = await tool.run(
                url="https://api.example.com/graphql",
                query="{ users { id } }",
                headers='{"Authorization": "Bearer token123"}',
            )

        assert not result.is_error

    async def test_invalid_json_variables(self, tool):
        mock_aiohttp = _mock_aiohttp_module(MagicMock())
        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await tool.run(
                url="https://api.example.com/graphql",
                query="{ user { id } }",
                variables="not-json",
            )
        assert result.is_error
        assert "Invalid JSON in variables" in result.error

    async def test_invalid_json_headers(self, tool):
        mock_aiohttp = _mock_aiohttp_module(MagicMock())
        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await tool.run(
                url="https://api.example.com/graphql",
                query="{ user { id } }",
                headers="not-json",
            )
        assert result.is_error
        assert "Invalid JSON in headers" in result.error

    async def test_no_url_error(self, tool):
        result = await tool.run(query="{ user { id } }")
        assert result.is_error
        assert "No URL" in result.error

    async def test_no_query_error(self, tool):
        result = await tool.run(url="https://api.example.com/graphql")
        assert result.is_error
        assert "No GraphQL query" in result.error

    async def test_http_error(self, tool):
        resp = self._make_response(status=500, text="Internal Server Error")
        session = self._make_session(resp)

        with patch.dict("sys.modules", {"aiohttp": _mock_aiohttp_module(session)}):
            result = await tool.run(
                url="https://api.example.com/graphql",
                query="{ user { id } }",
            )

        assert result.is_error
        assert "HTTP 500" in result.error

    def test_import_from_top_level(self):
        from synapsekit import GraphQLTool

        assert GraphQLTool is not None


# ------------------------------------------------------------------ #
# Feature 5: RedisLLMCache
# ------------------------------------------------------------------ #


class TestRedisLLMCache:
    def _make_cache_obj(self):
        """Create a RedisLLMCache with a mocked client."""
        from synapsekit.llm._redis_cache import RedisLLMCache

        cache = RedisLLMCache.__new__(RedisLLMCache)
        cache._client = MagicMock()
        cache._prefix = "synapsekit:llm:"
        cache._ttl = None
        cache.hits = 0
        cache.misses = 0
        return cache

    def test_put_and_get_hit(self):
        cache = self._make_cache_obj()
        cache._ttl = 60
        cache._client.get.return_value = json.dumps("cached_value")

        cache.put("key1", "value1")
        cache._client.set.assert_called_once()

        result = cache.get("key1")
        assert result == "cached_value"
        assert cache.hits == 1

    def test_get_miss(self):
        cache = self._make_cache_obj()
        cache._client.get.return_value = None

        result = cache.get("nonexistent")
        assert result is None
        assert cache.misses == 1

    def test_clear(self):
        cache = self._make_cache_obj()
        cache._client.scan.side_effect = [
            (0, ["synapsekit:llm:k1", "synapsekit:llm:k2"]),
        ]

        cache.clear()
        cache._client.delete.assert_called_once()

    def test_ttl(self):
        cache = self._make_cache_obj()
        cache._ttl = 300

        cache.put("key1", "value1")
        cache._client.expire.assert_called_once_with("synapsekit:llm:key1", 300)

    def test_make_key(self):
        from synapsekit.llm._redis_cache import RedisLLMCache

        key = RedisLLMCache.make_key("gpt-4", "hello", 0.7, 100)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hex

    def test_redis_cache_backend_in_basellm(self):
        """Ensure BaseLLM config supports the redis backend string."""
        from synapsekit.llm.base import LLMConfig

        config = LLMConfig(
            model="test",
            api_key="key",
            provider="openai",
            cache=True,
            cache_backend="redis",
            cache_db_path="redis://localhost:6379",
        )
        assert config.cache_backend == "redis"


# ------------------------------------------------------------------ #
# Feature 6: CohereReranker
# ------------------------------------------------------------------ #


class TestCohereReranker:
    async def test_basic_retrieve(self):
        from synapsekit.retrieval.cohere_reranker import CohereReranker

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = ["doc1", "doc2", "doc3"]

        mock_rerank_result = MagicMock()
        r1 = MagicMock()
        r1.index = 2
        r1.relevance_score = 0.9
        r2 = MagicMock()
        r2.index = 0
        r2.relevance_score = 0.7
        mock_rerank_result.results = [r1, r2]

        mock_client = MagicMock()
        mock_client.rerank.return_value = mock_rerank_result

        reranker = CohereReranker(retriever=mock_retriever)
        reranker._client = mock_client

        results = await reranker.retrieve("test query", top_k=2)
        assert results == ["doc3", "doc1"]

    async def test_retrieve_with_scores(self):
        from synapsekit.retrieval.cohere_reranker import CohereReranker

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = ["doc1", "doc2"]

        r1 = MagicMock()
        r1.index = 1
        r1.relevance_score = 0.95
        mock_rerank_result = MagicMock()
        mock_rerank_result.results = [r1]

        mock_client = MagicMock()
        mock_client.rerank.return_value = mock_rerank_result

        reranker = CohereReranker(retriever=mock_retriever)
        reranker._client = mock_client

        results = await reranker.retrieve_with_scores("test", top_k=1)
        assert len(results) == 1
        assert results[0]["text"] == "doc2"
        assert results[0]["relevance_score"] == 0.95

    async def test_empty_results(self):
        from synapsekit.retrieval.cohere_reranker import CohereReranker

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = []

        reranker = CohereReranker(retriever=mock_retriever)
        results = await reranker.retrieve("test")
        assert results == []

    def test_env_var_fallback(self):
        from synapsekit.retrieval.cohere_reranker import CohereReranker

        mock_retriever = MagicMock()
        reranker = CohereReranker(retriever=mock_retriever)
        # No api_key passed, should fall back to CO_API_KEY env var
        assert reranker._api_key is None

    def test_import_from_top_level(self):
        from synapsekit import CohereReranker

        assert CohereReranker is not None


# ------------------------------------------------------------------ #
# Feature 7: StepBackRetriever
# ------------------------------------------------------------------ #


class TestStepBackRetriever:
    async def test_basic_retrieve(self):
        from synapsekit.retrieval.step_back import StepBackRetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.side_effect = [
            ["doc1", "doc2"],  # original query results
            ["doc2", "doc3"],  # step-back query results
        ]

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "What are the general principles?"

        sb = StepBackRetriever(retriever=mock_retriever, llm=mock_llm)
        results = await sb.retrieve("What is X?", top_k=5)

        # Should be deduplicated: doc1, doc2, doc3
        assert results == ["doc1", "doc2", "doc3"]

    async def test_deduplication(self):
        from synapsekit.retrieval.step_back import StepBackRetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.side_effect = [
            ["doc1", "doc2"],
            ["doc1", "doc2"],  # all duplicates
        ]

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "Abstract question"

        sb = StepBackRetriever(retriever=mock_retriever, llm=mock_llm)
        results = await sb.retrieve("query", top_k=5)
        assert results == ["doc1", "doc2"]

    async def test_custom_prompt(self):
        from synapsekit.retrieval.step_back import StepBackRetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.side_effect = [["doc1"], ["doc2"]]

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "step back"

        custom = "Generalize: {query}"
        sb = StepBackRetriever(retriever=mock_retriever, llm=mock_llm, prompt_template=custom)
        await sb.retrieve("specific q")

        mock_llm.generate.assert_called_once_with("Generalize: specific q")

    async def test_empty_results(self):
        from synapsekit.retrieval.step_back import StepBackRetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.side_effect = [[], []]

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "abstract"

        sb = StepBackRetriever(retriever=mock_retriever, llm=mock_llm)
        results = await sb.retrieve("query")
        assert results == []

    def test_import_from_top_level(self):
        from synapsekit import StepBackRetriever

        assert StepBackRetriever is not None


# ------------------------------------------------------------------ #
# Feature 8: FLARERetriever
# ------------------------------------------------------------------ #


class TestFLARERetriever:
    async def test_no_markers_early_exit(self):
        from synapsekit.retrieval.flare import FLARERetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = ["doc1", "doc2"]

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "Here is a plain answer with no markers."

        flare = FLARERetriever(retriever=mock_retriever, llm=mock_llm)
        results = await flare.retrieve("What is X?")

        assert results == ["doc1", "doc2"]
        # Only 1 LLM call (initial generate)
        assert mock_llm.generate.call_count == 1

    async def test_with_markers(self):
        from synapsekit.retrieval.flare import FLARERetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.side_effect = [
            ["doc1"],  # initial retrieval
            ["doc2"],  # marker retrieval
        ]

        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            "Some answer [SEARCH: more about topic]",  # first gen with marker
            "Final answer without markers.",  # second gen, no markers
        ]

        flare = FLARERetriever(retriever=mock_retriever, llm=mock_llm)
        results = await flare.retrieve("What is X?", top_k=5)

        assert "doc1" in results
        assert "doc2" in results

    async def test_max_iterations_cap(self):
        from synapsekit.retrieval.flare import FLARERetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = ["doc1"]

        mock_llm = AsyncMock()
        # Always returns markers — should still stop at max_iterations
        mock_llm.generate.return_value = "Answer [SEARCH: more info]"

        flare = FLARERetriever(retriever=mock_retriever, llm=mock_llm, max_iterations=2)
        await flare.retrieve("query")

        # Should stop after 2 iterations
        assert mock_llm.generate.call_count == 2

    async def test_empty_retrieval(self):
        from synapsekit.retrieval.flare import FLARERetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = []

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "No info available."

        flare = FLARERetriever(retriever=mock_retriever, llm=mock_llm)
        results = await flare.retrieve("query")
        assert results == []

    def test_parse_search_markers(self):
        from synapsekit.retrieval.flare import FLARERetriever

        flare = FLARERetriever(retriever=MagicMock(), llm=MagicMock())
        text = "Some text [SEARCH: query one] more text [SEARCH: query two] end"
        markers = flare._parse_search_markers(text)
        assert markers == ["query one", "query two"]

    def test_parse_no_markers(self):
        from synapsekit.retrieval.flare import FLARERetriever

        flare = FLARERetriever(retriever=MagicMock(), llm=MagicMock())
        markers = flare._parse_search_markers("Just plain text.")
        assert markers == []

    def test_import_from_top_level(self):
        from synapsekit import FLARERetriever

        assert FLARERetriever is not None
