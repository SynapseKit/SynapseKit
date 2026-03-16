"""Tests for v0.6.6 features: 2 tools, 4 retrieval strategies, 2 LLM providers, 2 memory backends."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ------------------------------------------------------------------ #
# Version
# ------------------------------------------------------------------ #


def test_version():
    import synapsekit

    assert synapsekit.__version__ == "0.6.6"


# ------------------------------------------------------------------ #
# Feature 1: BufferMemory (#133)
# ------------------------------------------------------------------ #


class TestBufferMemory:
    def test_add_and_get(self):
        from synapsekit.memory.buffer import BufferMemory

        mem = BufferMemory()
        mem.add("user", "Hello")
        mem.add("assistant", "Hi there!")
        msgs = mem.get_messages()
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "Hello"}
        assert msgs[1] == {"role": "assistant", "content": "Hi there!"}

    def test_unbounded(self):
        from synapsekit.memory.buffer import BufferMemory

        mem = BufferMemory()
        for i in range(100):
            mem.add("user", f"msg {i}")
        assert len(mem) == 100

    def test_format_context(self):
        from synapsekit.memory.buffer import BufferMemory

        mem = BufferMemory()
        mem.add("user", "Hello")
        mem.add("assistant", "Hi")
        ctx = mem.format_context()
        assert "User: Hello" in ctx
        assert "Assistant: Hi" in ctx

    def test_format_context_empty(self):
        from synapsekit.memory.buffer import BufferMemory

        mem = BufferMemory()
        assert mem.format_context() == ""

    def test_clear(self):
        from synapsekit.memory.buffer import BufferMemory

        mem = BufferMemory()
        mem.add("user", "Hello")
        mem.clear()
        assert len(mem) == 0
        assert mem.get_messages() == []

    def test_len(self):
        from synapsekit.memory.buffer import BufferMemory

        mem = BufferMemory()
        assert len(mem) == 0
        mem.add("user", "Hello")
        assert len(mem) == 1

    def test_get_messages_returns_copy(self):
        from synapsekit.memory.buffer import BufferMemory

        mem = BufferMemory()
        mem.add("user", "Hello")
        msgs = mem.get_messages()
        msgs.clear()
        assert len(mem) == 1

    def test_import_from_top_level(self):
        from synapsekit import BufferMemory

        assert BufferMemory is not None


# ------------------------------------------------------------------ #
# Feature 2: ArxivSearchTool (#230)
# ------------------------------------------------------------------ #

_ARXIV_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Attention Is All You Need</title>
    <summary>We propose a new network architecture, the Transformer.</summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <link href="http://arxiv.org/abs/1706.03762v7" type="text/html"/>
  </entry>
</feed>"""

_ARXIV_EMPTY_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"""


class TestArxivSearchTool:
    @pytest.fixture
    def tool(self):
        from synapsekit.agents.tools.arxiv_search import ArxivSearchTool

        return ArxivSearchTool()

    async def test_search_success(self, tool):
        mock_resp = MagicMock()
        mock_resp.read.return_value = _ARXIV_XML.encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = await tool.run(query="attention transformer")

        assert not result.is_error
        assert "Attention Is All You Need" in result.output
        assert "Ashish Vaswani" in result.output

    async def test_empty_results(self, tool):
        mock_resp = MagicMock()
        mock_resp.read.return_value = _ARXIV_EMPTY_XML.encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = await tool.run(query="xyznonexistentquery")

        assert result.output == "No results found."

    async def test_no_query_error(self, tool):
        result = await tool.run()
        assert result.is_error
        assert "No search query" in result.error

    def test_import_from_top_level(self):
        from synapsekit import ArxivSearchTool

        assert ArxivSearchTool is not None


# ------------------------------------------------------------------ #
# Feature 3: TavilySearchTool (#200)
# ------------------------------------------------------------------ #


def _mock_tavily_module(mock_client):
    """Create a mock tavily module with a TavilyClient class."""
    mod = MagicMock()
    mod.TavilyClient.return_value = mock_client
    return mod


class TestTavilySearchTool:
    @pytest.fixture
    def tool(self):
        from synapsekit.agents.tools.tavily_search import TavilySearchTool

        return TavilySearchTool(api_key="test-key")

    async def test_search_success(self, tool):
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {"title": "Result 1", "url": "https://example.com/1", "content": "Snippet 1"},
                {"title": "Result 2", "url": "https://example.com/2", "content": "Snippet 2"},
            ]
        }

        with patch.dict("sys.modules", {"tavily": _mock_tavily_module(mock_client)}):
            result = await tool.run(query="AI breakthroughs")

        assert not result.is_error
        assert "Result 1" in result.output
        assert "Result 2" in result.output

    async def test_empty_results(self, tool):
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}

        with patch.dict("sys.modules", {"tavily": _mock_tavily_module(mock_client)}):
            result = await tool.run(query="obscure query")

        assert result.output == "No results found."

    async def test_no_query_error(self, tool):
        mock_tavily = _mock_tavily_module(MagicMock())
        with patch.dict("sys.modules", {"tavily": mock_tavily}):
            result = await tool.run()
        assert result.is_error
        assert "No search query" in result.error

    async def test_missing_dep(self):
        from synapsekit.agents.tools.tavily_search import TavilySearchTool

        tool = TavilySearchTool(api_key="test-key")
        with patch.dict("sys.modules", {"tavily": None}):
            with pytest.raises(ImportError, match="tavily-python"):
                await tool.run(query="test")

    async def test_no_api_key(self):
        from synapsekit.agents.tools.tavily_search import TavilySearchTool

        tool = TavilySearchTool()
        mock_tavily = _mock_tavily_module(MagicMock())
        with (
            patch.dict("sys.modules", {"tavily": mock_tavily}),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = await tool.run(query="test")
        assert result.is_error
        assert "API key" in result.error

    def test_import_from_top_level(self):
        from synapsekit import TavilySearchTool

        assert TavilySearchTool is not None


# ------------------------------------------------------------------ #
# Feature 4: PerplexityLLM (#169)
# ------------------------------------------------------------------ #


def _make_config(provider="perplexity", model="sonar"):
    from synapsekit.llm.base import LLMConfig

    return LLMConfig(
        model=model,
        api_key="test-key",
        provider=provider,
        system_prompt="You are helpful.",
        temperature=0.2,
        max_tokens=100,
    )


class TestPerplexityLLM:
    def test_import(self):
        from synapsekit.llm.perplexity import PerplexityLLM

        assert PerplexityLLM is not None

    def test_default_base_url(self):
        from synapsekit.llm.perplexity import PerplexityLLM

        llm = PerplexityLLM(config=_make_config())
        assert llm._base_url == "https://api.perplexity.ai"

    def test_custom_base_url(self):
        from synapsekit.llm.perplexity import PerplexityLLM

        llm = PerplexityLLM(config=_make_config(), base_url="https://custom.api/v1")
        assert llm._base_url == "https://custom.api/v1"

    def test_import_from_top_level(self):
        from synapsekit import PerplexityLLM

        assert PerplexityLLM is not None


# ------------------------------------------------------------------ #
# Feature 5: CerebrasLLM (#171)
# ------------------------------------------------------------------ #


class TestCerebrasLLM:
    def test_import(self):
        from synapsekit.llm.cerebras import CerebrasLLM

        assert CerebrasLLM is not None

    def test_default_base_url(self):
        from synapsekit.llm.cerebras import CerebrasLLM

        llm = CerebrasLLM(config=_make_config(provider="cerebras"))
        assert llm._base_url == "https://api.cerebras.ai/v1"

    def test_custom_base_url(self):
        from synapsekit.llm.cerebras import CerebrasLLM

        llm = CerebrasLLM(config=_make_config(provider="cerebras"), base_url="https://custom/v1")
        assert llm._base_url == "https://custom/v1"

    def test_import_from_top_level(self):
        from synapsekit import CerebrasLLM

        assert CerebrasLLM is not None


# ------------------------------------------------------------------ #
# Feature 6: HybridSearchRetriever (#143)
# ------------------------------------------------------------------ #


class TestHybridSearchRetriever:
    async def test_basic_retrieve(self):
        from synapsekit.retrieval.hybrid_search import HybridSearchRetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = ["machine learning is great", "deep learning rocks"]

        hybrid = HybridSearchRetriever(retriever=mock_retriever)
        hybrid.add_documents(
            [
                "machine learning is great",
                "deep learning rocks",
                "cooking recipes are fun",
            ]
        )

        results = await hybrid.retrieve("machine learning", top_k=2)
        assert len(results) <= 2
        assert isinstance(results, list)

    async def test_weight_tuning(self):
        from synapsekit.retrieval.hybrid_search import HybridSearchRetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = ["doc1"]

        # All weight to BM25
        hybrid = HybridSearchRetriever(retriever=mock_retriever, bm25_weight=1.0, vector_weight=0.0)
        hybrid.add_documents(["doc1", "doc2"])
        results = await hybrid.retrieve("doc1", top_k=2)
        assert isinstance(results, list)

    async def test_empty_bm25_fallback(self):
        from synapsekit.retrieval.hybrid_search import HybridSearchRetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = ["doc1", "doc2"]

        hybrid = HybridSearchRetriever(retriever=mock_retriever)
        # No add_documents called — BM25 index is None
        results = await hybrid.retrieve("query", top_k=5)
        # Should still return vector results
        assert results == ["doc1", "doc2"]

    async def test_deduplication(self):
        from synapsekit.retrieval.hybrid_search import HybridSearchRetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = ["doc1", "doc2"]

        hybrid = HybridSearchRetriever(retriever=mock_retriever)
        hybrid.add_documents(["doc1", "doc2", "doc3"])
        results = await hybrid.retrieve("doc1", top_k=5)
        # No duplicates
        assert len(results) == len(set(results))

    def test_import_from_top_level(self):
        from synapsekit import HybridSearchRetriever

        assert HybridSearchRetriever is not None


# ------------------------------------------------------------------ #
# Feature 7: EntityMemory (#136)
# ------------------------------------------------------------------ #


class TestEntityMemory:
    async def test_entity_extraction(self):
        from synapsekit.memory.entity import EntityMemory

        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            "Alice, Acme Corp",  # extract
            "Alice is a software engineer.",  # summarize Alice
            "Acme Corp is a tech company.",  # summarize Acme Corp
        ]

        mem = EntityMemory(llm=mock_llm)
        await mem.add("user", "Alice works at Acme Corp.")

        entities = mem.get_entities()
        assert "Alice" in entities
        assert "Acme Corp" in entities
        assert len(mem) == 1

    async def test_entity_update(self):
        from synapsekit.memory.entity import EntityMemory

        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            "Alice",  # extract first message
            "Alice is a person.",  # summarize
            "Alice",  # extract second message
            "Alice is a software engineer at Acme.",  # updated summary
        ]

        mem = EntityMemory(llm=mock_llm)
        await mem.add("user", "I met Alice.")
        await mem.add("user", "Alice works at Acme.")

        entities = mem.get_entities()
        assert "Alice" in entities
        assert "Acme" in entities["Alice"]

    async def test_eviction(self):
        from synapsekit.memory.entity import EntityMemory

        mock_llm = AsyncMock()
        # For each add: extract returns entity name, summarize returns desc
        responses = []
        for i in range(5):
            responses.append(f"Entity{i}")
            responses.append(f"Description for Entity{i}")
        mock_llm.generate.side_effect = responses

        mem = EntityMemory(llm=mock_llm, max_entities=3)
        for i in range(5):
            await mem.add("user", f"About Entity{i}")

        entities = mem.get_entities()
        assert len(entities) <= 3
        # Oldest should be evicted
        assert "Entity0" not in entities
        assert "Entity1" not in entities

    async def test_no_entities(self):
        from synapsekit.memory.entity import EntityMemory

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "NONE"

        mem = EntityMemory(llm=mock_llm)
        await mem.add("user", "Hello!")

        assert mem.get_entities() == {}
        assert len(mem) == 1

    async def test_format_context_includes_entities(self):
        from synapsekit.memory.entity import EntityMemory

        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            "Alice",
            "Alice is a friend.",
        ]

        mem = EntityMemory(llm=mock_llm)
        await mem.add("user", "I know Alice.")

        ctx = mem.format_context()
        assert "Known entities:" in ctx
        assert "Alice" in ctx
        assert "User: I know Alice." in ctx

    async def test_clear(self):
        from synapsekit.memory.entity import EntityMemory

        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = ["Alice", "Alice is a person."]

        mem = EntityMemory(llm=mock_llm)
        await mem.add("user", "Alice is here.")
        mem.clear()

        assert len(mem) == 0
        assert mem.get_entities() == {}
        assert mem.get_messages() == []

    def test_import_from_top_level(self):
        from synapsekit import EntityMemory

        assert EntityMemory is not None


# ------------------------------------------------------------------ #
# Feature 8: SelfRAGRetriever (#154)
# ------------------------------------------------------------------ #


class TestSelfRAGRetriever:
    async def test_all_relevant_fully_supported(self):
        from synapsekit.retrieval.self_rag import SelfRAGRetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = ["doc1", "doc2"]

        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            "relevant",  # grade doc1
            "relevant",  # grade doc2
            "Answer based on docs.",  # generate
            "fully",  # support check
        ]

        sr = SelfRAGRetriever(retriever=mock_retriever, llm=mock_llm)
        results = await sr.retrieve("What is X?", top_k=5)
        assert results == ["doc1", "doc2"]

    async def test_not_supported_triggers_retry(self):
        from synapsekit.retrieval.self_rag import SelfRAGRetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.side_effect = [
            ["doc1"],  # first attempt
            ["doc2"],  # second attempt
        ]

        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            "relevant",  # grade doc1
            "Some answer.",  # generate
            "not",  # support check — not supported
            "Rewritten question",  # critique
            "relevant",  # grade doc2
            "Better answer.",  # generate
            "fully",  # support check — fully
        ]

        sr = SelfRAGRetriever(retriever=mock_retriever, llm=mock_llm, max_iterations=2)
        results = await sr.retrieve("What is X?", top_k=5)
        assert results == ["doc2"]

    async def test_empty_retrieval(self):
        from synapsekit.retrieval.self_rag import SelfRAGRetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = []

        mock_llm = AsyncMock()

        sr = SelfRAGRetriever(retriever=mock_retriever, llm=mock_llm)
        results = await sr.retrieve("What is X?")
        assert results == []

    async def test_max_iterations_cap(self):
        from synapsekit.retrieval.self_rag import SelfRAGRetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = ["doc1"]

        mock_llm = AsyncMock()
        # Always "not" supported
        mock_llm.generate.side_effect = [
            "relevant",  # grade
            "Answer.",  # generate
            "not",  # support
            "Rewritten",  # critique
            "relevant",  # grade
            "Answer.",  # generate
            "not",  # support — hits max_iterations
        ]

        sr = SelfRAGRetriever(retriever=mock_retriever, llm=mock_llm, max_iterations=2)
        results = await sr.retrieve("What is X?")
        assert results == ["doc1"]

    async def test_retrieve_with_reflection_metadata(self):
        from synapsekit.retrieval.self_rag import SelfRAGRetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = ["doc1"]

        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            "relevant",
            "Answer.",
            "fully",
        ]

        sr = SelfRAGRetriever(retriever=mock_retriever, llm=mock_llm)
        docs, meta = await sr.retrieve_with_reflection("What is X?")
        assert docs == ["doc1"]
        assert meta["support_level"] == "fully"
        assert meta["iterations"] == 1

    def test_import_from_top_level(self):
        from synapsekit import SelfRAGRetriever

        assert SelfRAGRetriever is not None


# ------------------------------------------------------------------ #
# Feature 9: AdaptiveRAGRetriever (#153)
# ------------------------------------------------------------------ #


class TestAdaptiveRAGRetriever:
    async def test_routes_to_simple(self):
        from synapsekit.retrieval.adaptive import AdaptiveRAGRetriever

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "simple"

        simple = AsyncMock()
        simple.retrieve.return_value = ["simple_doc"]
        moderate = AsyncMock()
        complex_ = AsyncMock()

        ar = AdaptiveRAGRetriever(
            llm=mock_llm,
            simple_retriever=simple,
            moderate_retriever=moderate,
            complex_retriever=complex_,
        )
        results = await ar.retrieve("What is 2+2?")
        assert results == ["simple_doc"]
        simple.retrieve.assert_called_once()

    async def test_routes_to_complex(self):
        from synapsekit.retrieval.adaptive import AdaptiveRAGRetriever

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "complex"

        simple = AsyncMock()
        complex_ = AsyncMock()
        complex_.retrieve.return_value = ["complex_doc"]

        ar = AdaptiveRAGRetriever(llm=mock_llm, simple_retriever=simple, complex_retriever=complex_)
        results = await ar.retrieve("Explain the implications of quantum computing on cryptography")
        assert results == ["complex_doc"]

    async def test_fallback_when_not_all_provided(self):
        from synapsekit.retrieval.adaptive import AdaptiveRAGRetriever

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "complex"

        simple = AsyncMock()
        simple.retrieve.return_value = ["fallback_doc"]

        ar = AdaptiveRAGRetriever(llm=mock_llm, simple_retriever=simple)
        # No moderate or complex provided, should fall back
        results = await ar.retrieve("complex question")
        assert results == ["fallback_doc"]

    async def test_retrieve_with_classification(self):
        from synapsekit.retrieval.adaptive import AdaptiveRAGRetriever

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "moderate"

        moderate = AsyncMock()
        moderate.retrieve.return_value = ["mod_doc"]

        ar = AdaptiveRAGRetriever(
            llm=mock_llm, simple_retriever=AsyncMock(), moderate_retriever=moderate
        )
        results, classification = await ar.retrieve_with_classification("question")
        assert results == ["mod_doc"]
        assert classification == "moderate"

    def test_import_from_top_level(self):
        from synapsekit import AdaptiveRAGRetriever

        assert AdaptiveRAGRetriever is not None


# ------------------------------------------------------------------ #
# Feature 10: MultiStepRetriever (#155)
# ------------------------------------------------------------------ #


class TestMultiStepRetriever:
    async def test_gaps_found_and_filled(self):
        from synapsekit.retrieval.multi_step import MultiStepRetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.side_effect = [
            ["doc1", "doc2"],  # initial
            ["doc3"],  # gap fill
        ]

        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            "Partial answer.",  # generate
            "What about topic B?",  # gap (single query)
            "Full answer.",  # generate after gap fill
            "COMPLETE",  # no more gaps
        ]

        ms = MultiStepRetriever(retriever=mock_retriever, llm=mock_llm)
        results = await ms.retrieve("What is A and B?", top_k=5)
        assert "doc1" in results
        assert "doc2" in results
        assert "doc3" in results

    async def test_complete_exits_early(self):
        from synapsekit.retrieval.multi_step import MultiStepRetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = ["doc1"]

        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            "Complete answer.",
            "COMPLETE",
        ]

        ms = MultiStepRetriever(retriever=mock_retriever, llm=mock_llm, max_steps=3)
        results = await ms.retrieve("Simple question")
        assert results == ["doc1"]
        # Only 1 retrieval call (initial), no gap fill
        assert mock_retriever.retrieve.call_count == 1

    async def test_max_steps_cap(self):
        from synapsekit.retrieval.multi_step import MultiStepRetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = ["doc1"]

        mock_llm = AsyncMock()
        # Always finds gaps
        mock_llm.generate.side_effect = [
            "Partial.",
            "gap query 1",
            "Still partial.",
            "gap query 2",
        ]

        ms = MultiStepRetriever(retriever=mock_retriever, llm=mock_llm, max_steps=2)
        results = await ms.retrieve("question")
        assert "doc1" in results

    async def test_deduplication(self):
        from synapsekit.retrieval.multi_step import MultiStepRetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.side_effect = [
            ["doc1", "doc2"],  # initial
            ["doc1", "doc3"],  # gap fill — doc1 is duplicate
        ]

        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            "Partial.",
            "gap query",
            "Full.",
            "COMPLETE",
        ]

        ms = MultiStepRetriever(retriever=mock_retriever, llm=mock_llm)
        results = await ms.retrieve("question")
        assert results.count("doc1") == 1
        assert "doc3" in results

    async def test_retrieve_with_steps_trace(self):
        from synapsekit.retrieval.multi_step import MultiStepRetriever

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = ["doc1"]

        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            "Answer.",
            "COMPLETE",
        ]

        ms = MultiStepRetriever(retriever=mock_retriever, llm=mock_llm)
        docs, trace = await ms.retrieve_with_steps("question")
        assert docs == ["doc1"]
        assert len(trace) >= 2
        assert trace[0]["step"] == 0
        assert trace[1].get("complete") is True

    def test_import_from_top_level(self):
        from synapsekit import MultiStepRetriever

        assert MultiStepRetriever is not None
