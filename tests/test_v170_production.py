"""Production-grade tests for v1.7.0 readiness.

Covers:
  - FederatedRetriever (comprehensive — all fusion modes, remote HTTP, auth, dedup, edge cases)
  - CostQualityRouter.stream() (explore + exploit + stats update)
  - ReasoningLLM edge cases (DeepSeek stream, Google generate, provider detection, delegation)
  - Performance benchmarks (timing assertions for vector store, JSON, cache, splitters)
  - Integration scenarios (cross-feature wiring)
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockRetriever:
    """Retriever stub supporting both retrieve() and retrieve_with_scores()."""

    def __init__(self, results: list[tuple[str, float]]) -> None:
        self._results = results

    async def retrieve(self, query: str, top_k: int = 5, metadata_filter=None) -> list[str]:
        return [t for t, _ in self._results[:top_k]]

    async def retrieve_with_scores(
        self, query: str, top_k: int = 5, metadata_filter=None
    ) -> list[dict]:
        return [
            {"text": t, "score": s, "metadata": {"src": "mock"}} for t, s in self._results[:top_k]
        ]


class _ScoresOnlyRetriever:
    """Retriever that only exposes retrieve() (no scores)."""

    def __init__(self, results: list[str]) -> None:
        self._results = results

    async def retrieve(self, query: str, top_k: int = 5, metadata_filter=None) -> list[str]:
        return self._results[:top_k]


# ===========================================================================
# 1. FederatedRetriever — comprehensive
# ===========================================================================


class TestFederatedRetrieverInit:
    def test_empty_sources_raises(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        with pytest.raises(ValueError, match="At least one source"):
            FederatedRetriever(sources=[])

    def test_single_source_ok(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        r = _MockRetriever([("a", 0.9)])
        fr = FederatedRetriever(sources=[{"name": "x", "retriever": r}])
        assert fr._top_k == 10

    def test_custom_params_stored(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        r = _MockRetriever([])
        fr = FederatedRetriever(
            sources=[{"retriever": r}],
            fusion="score",
            top_k=5,
            timeout_ms=1000,
            rrf_k=30,
            dedup_threshold=0.85,
        )
        assert fr._fusion == "score"
        assert fr._top_k == 5
        assert fr._timeout_ms == 1000
        assert fr._rrf_k == 30
        assert fr._dedup_threshold == 0.85


class TestFederatedRetrieverRRF:
    @pytest.mark.asyncio
    async def test_rrf_scores_boost_repeated_results(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        a = _MockRetriever([("shared", 0.9), ("only_a", 0.8)])
        b = _MockRetriever([("shared", 0.95), ("only_b", 0.7)])

        fr = FederatedRetriever(
            sources=[{"name": "a", "retriever": a}, {"name": "b", "retriever": b}],
            fusion="rrf",
            top_k=3,
        )
        results = await fr.retrieve_with_scores("q")
        texts = [r["text"] for r in results]
        assert texts[0] == "shared"

    @pytest.mark.asyncio
    async def test_rrf_top_k_respected(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        a = _MockRetriever([("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)])
        fr = FederatedRetriever(sources=[{"retriever": a}], fusion="rrf", top_k=2)
        results = await fr.retrieve("q")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_rrf_top_k_override_per_call(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        a = _MockRetriever([("a", 0.9), ("b", 0.8), ("c", 0.7)])
        fr = FederatedRetriever(sources=[{"retriever": a}], fusion="rrf", top_k=10)
        results = await fr.retrieve("q", top_k=1)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_rrf_source_name_propagated(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        a = _MockRetriever([("doc", 0.9)])
        fr = FederatedRetriever(sources=[{"name": "my_source", "retriever": a}])
        results = await fr.retrieve_with_scores("q")
        assert results[0]["source"] == "my_source"

    @pytest.mark.asyncio
    async def test_rrf_no_name_source_is_none(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        a = _MockRetriever([("doc", 0.9)])
        fr = FederatedRetriever(sources=[{"retriever": a}])
        results = await fr.retrieve_with_scores("q")
        assert results[0]["source"] is None


class TestFederatedRetrieverScoreFusion:
    @pytest.mark.asyncio
    async def test_score_fusion_normalises_scores(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        a = _MockRetriever([("high", 1.0), ("low", 0.0)])
        fr = FederatedRetriever(sources=[{"retriever": a}], fusion="score", top_k=2)
        results = await fr.retrieve_with_scores("q")
        assert results[0]["text"] == "high"

    @pytest.mark.asyncio
    async def test_score_fusion_all_same_score(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        # When all scores are identical, denom=0 → all normalised to 1.0
        a = _MockRetriever([("a", 0.5), ("b", 0.5), ("c", 0.5)])
        fr = FederatedRetriever(sources=[{"retriever": a}], fusion="score", top_k=3)
        results = await fr.retrieve("q")
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_score_fusion_no_score_uses_rank_fallback(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        # Retriever with no scores — use _ScoresOnlyRetriever
        class _NoScore:
            async def retrieve_with_scores(self, query, top_k=5, metadata_filter=None):
                return [{"text": "doc1", "score": None, "metadata": {}}]

        fr = FederatedRetriever(sources=[{"retriever": _NoScore()}], fusion="score", top_k=1)
        results = await fr.retrieve("q")
        assert results == ["doc1"]

    @pytest.mark.asyncio
    async def test_score_fusion_cross_source_ranking(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        # A document appearing in BOTH sources accumulates normalised score from each
        # and should rank above documents seen in only one source.
        a = _MockRetriever([("shared_doc", 0.9), ("only_a", 0.1)])
        b = _MockRetriever([("shared_doc", 0.7), ("only_b", 0.2)])
        fr = FederatedRetriever(
            sources=[{"retriever": a}, {"retriever": b}], fusion="score", top_k=3
        )
        results = await fr.retrieve("q")
        # "shared_doc" gets normalised top score from both sources → should rank first
        assert results[0] == "shared_doc"


class TestFederatedRetrieverInterleave:
    @pytest.mark.asyncio
    async def test_interleave_round_robin_order(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        a = _MockRetriever([("a1", 0.9), ("a2", 0.8)])
        b = _MockRetriever([("b1", 0.7), ("b2", 0.6)])
        fr = FederatedRetriever(
            sources=[{"retriever": a}, {"retriever": b}], fusion="interleave", top_k=4
        )
        results = await fr.retrieve("q")
        assert results == ["a1", "b1", "a2", "b2"]

    @pytest.mark.asyncio
    async def test_interleave_skips_duplicates(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        a = _MockRetriever([("same", 0.9)])
        b = _MockRetriever([("same", 0.8), ("other", 0.7)])
        fr = FederatedRetriever(
            sources=[{"retriever": a}, {"retriever": b}], fusion="interleave", top_k=5
        )
        results = await fr.retrieve("q")
        assert results.count("same") == 1
        assert "other" in results

    @pytest.mark.asyncio
    async def test_interleave_unequal_source_lengths(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        a = _MockRetriever([("a1", 0.9), ("a2", 0.8), ("a3", 0.7)])
        b = _MockRetriever([("b1", 0.6)])
        fr = FederatedRetriever(
            sources=[{"retriever": a}, {"retriever": b}], fusion="interleave", top_k=5
        )
        results = await fr.retrieve("q")
        assert "a1" in results
        assert "b1" in results
        assert "a2" in results


class TestFederatedRetrieverDedup:
    @pytest.mark.asyncio
    async def test_exact_duplicate_removed(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        a = _MockRetriever([("same doc", 0.9)])
        b = _MockRetriever([("same doc", 0.8)])
        fr = FederatedRetriever(sources=[{"retriever": a}, {"retriever": b}], fusion="rrf", top_k=5)
        results = await fr.retrieve("q")
        assert results.count("same doc") == 1

    @pytest.mark.asyncio
    async def test_near_duplicate_removed_above_threshold(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        a = _MockRetriever([("The quick brown fox jumps over the lazy dog", 0.9)])
        b = _MockRetriever([("The quick brown fox jumps over the lazy dog.", 0.8)])
        fr = FederatedRetriever(
            sources=[{"retriever": a}, {"retriever": b}],
            fusion="rrf",
            dedup_threshold=0.90,
            top_k=5,
        )
        results = await fr.retrieve("q")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_low_threshold_allows_near_duplicates(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        # dedup_threshold=0.99 means only >99% similarity triggers dedup,
        # so "apple pie recipe" vs "apple pie recipes" (≈97% similar) are BOTH kept.
        a = _MockRetriever([("apple pie recipe", 0.9)])
        b = _MockRetriever([("apple pie recipes", 0.8)])
        fr = FederatedRetriever(
            sources=[{"retriever": a}, {"retriever": b}],
            fusion="rrf",
            dedup_threshold=0.99,  # very high threshold → near-dups kept
            top_k=5,
        )
        results = await fr.retrieve("q")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_choose_best_keeps_higher_score(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        a = _MockRetriever([("shared", 0.9)])
        b = _MockRetriever([("shared", 0.5)])
        fr = FederatedRetriever(
            sources=[{"retriever": a}, {"retriever": b}], fusion="score", top_k=1
        )
        results = await fr.retrieve_with_scores("q")
        assert results[0]["score"] == 0.9


class TestFederatedRetrieverFaultTolerance:
    @pytest.mark.asyncio
    async def test_failing_source_skipped_partial_result(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        class _Broken:
            async def retrieve_with_scores(self, query, top_k=5, metadata_filter=None):
                raise RuntimeError("provider down")

        good = _MockRetriever([("good doc", 0.9)])
        fr = FederatedRetriever(
            sources=[{"retriever": good}, {"retriever": _Broken()}], fusion="rrf", top_k=5
        )
        results = await fr.retrieve("q")
        assert results == ["good doc"]

    @pytest.mark.asyncio
    async def test_all_sources_fail_returns_empty(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        class _Broken:
            async def retrieve_with_scores(self, query, top_k=5, metadata_filter=None):
                raise RuntimeError("dead")

        fr = FederatedRetriever(
            sources=[{"retriever": _Broken()}, {"retriever": _Broken()}], fusion="rrf"
        )
        results = await fr.retrieve("q")
        assert results == []

    @pytest.mark.asyncio
    async def test_per_source_timeout_override(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        class _Slow:
            async def retrieve_with_scores(self, query, top_k=5, metadata_filter=None):
                await asyncio.sleep(0.1)
                return [{"text": "late", "score": 0.1, "metadata": {}}]

        fast = _MockRetriever([("fast", 0.9)])
        fr = FederatedRetriever(
            sources=[
                {"retriever": fast},
                {"retriever": _Slow(), "timeout_ms": 1},  # 1ms — will timeout
            ],
            timeout_ms=5000,
            fusion="rrf",
        )
        results = await fr.retrieve("q")
        assert "fast" in results
        assert "late" not in results

    @pytest.mark.asyncio
    async def test_invalid_source_config_returns_empty(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        # Source with neither 'retriever' nor 'url' raises ValueError inside
        # asyncio.gather(return_exceptions=True), which silently skips it.
        # The result is an empty list (all sources failed).
        fr = FederatedRetriever(
            sources=[{"name": "bad_source"}],  # neither 'retriever' nor 'url'
            fusion="rrf",
        )
        results = await fr.retrieve("q")
        assert results == []

    @pytest.mark.asyncio
    async def test_scores_only_retriever_no_scores_in_results(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        r = _ScoresOnlyRetriever(["doc_a", "doc_b"])
        fr = FederatedRetriever(sources=[{"retriever": r}], fusion="rrf", top_k=2)
        results = await fr.retrieve_with_scores("q")
        assert len(results) == 2
        assert all(r["score"] is None for r in results)

    @pytest.mark.asyncio
    async def test_metadata_filter_passed_to_local_retriever(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        received_filter = []

        class _FilterCapture:
            async def retrieve_with_scores(self, query, top_k=5, metadata_filter=None):
                received_filter.append(metadata_filter)
                return [{"text": "doc", "score": 0.9, "metadata": {}}]

        fr = FederatedRetriever(sources=[{"retriever": _FilterCapture()}])
        await fr.retrieve("q", metadata_filter={"tag": "news"})
        assert received_filter[0] == {"tag": "news"}


class TestFederatedRetrieverRemoteHTTP:
    @pytest.mark.asyncio
    async def test_remote_source_sends_correct_payload(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        captured = {}

        class _FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"results": [{"text": "remote doc", "score": 0.8, "metadata": {}}]}

        class _FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, json=None, headers=None):
                captured["url"] = url
                captured["json"] = json
                captured["headers"] = headers
                return _FakeResponse()

        with patch("httpx.AsyncClient", return_value=_FakeClient()):
            fr = FederatedRetriever(
                sources=[{"name": "remote", "url": "https://example.com/retrieve"}]
            )
            results = await fr.retrieve("my query", top_k=3)

        assert results == ["remote doc"]
        assert captured["json"]["query"] == "my query"
        assert captured["json"]["top_k"] == 3

    @pytest.mark.asyncio
    async def test_remote_source_sends_auth_header(self, monkeypatch):
        import socket

        from synapsekit.retrieval.federated import FederatedRetriever

        # The SSRF guard (fail-closed) now resolves the host before the request;
        # stub DNS so the placeholder host validates as a public address.
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *a, **k: [
                (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 0))
            ],
        )

        captured_headers = {}

        class _FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return [{"text": "doc", "score": 0.9}]

        class _FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, json=None, headers=None):
                captured_headers.update(headers or {})
                return _FakeResponse()

        with patch("httpx.AsyncClient", return_value=_FakeClient()):
            fr = FederatedRetriever(
                sources=[
                    {
                        "url": "https://api.example.com/retrieve",
                        "api_key": "secret-token-123",
                    }
                ]
            )
            await fr.retrieve("q")

        assert captured_headers.get("Authorization") == "Bearer secret-token-123"

    @pytest.mark.asyncio
    async def test_remote_source_skips_empty_text(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        class _FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "results": [
                        {"text": "", "score": 0.9},
                        {"text": "good doc", "score": 0.8},
                        {"score": 0.7},  # no text key
                    ]
                }

        class _FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, json=None, headers=None):
                return _FakeResponse()

        with patch("httpx.AsyncClient", return_value=_FakeClient()):
            fr = FederatedRetriever(sources=[{"url": "https://example.com/retrieve"}])
            results = await fr.retrieve("q")

        assert results == ["good doc"]

    @pytest.mark.asyncio
    async def test_remote_source_no_httpx_returns_empty(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        # When httpx is unavailable, _fetch_remote raises ImportError inside
        # asyncio.gather(return_exceptions=True), which silently skips it.
        # The result is an empty list.
        with patch.dict("sys.modules", {"httpx": None}):
            fr = FederatedRetriever(sources=[{"url": "https://example.com/retrieve"}])
            results = await fr.retrieve("q")
        assert results == []

    @pytest.mark.asyncio
    async def test_remote_source_results_key_parsed(self):
        from synapsekit.retrieval.federated import FederatedRetriever

        class _FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                # Response with "results" key and extra top-level fields
                return {"results": [{"text": "remote doc", "score": 0.5}], "count": 1}

        class _FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, json=None, headers=None):
                return _FakeResponse()

        with patch("httpx.AsyncClient", return_value=_FakeClient()):
            fr = FederatedRetriever(sources=[{"url": "https://example.com/retrieve"}])
            results = await fr.retrieve("q")

        assert results == ["remote doc"]


# ===========================================================================
# 2. CostQualityRouter — stream() comprehensive
# ===========================================================================


class TestCostQualityRouterStream:
    def _make_llm(self, name: str, tokens: list[str]) -> MagicMock:
        from synapsekit.llm.base import LLMConfig

        llm = MagicMock()
        llm.config = LLMConfig(model=name, api_key="", provider="openai")
        llm._input_tokens = 0
        llm._output_tokens = 10

        async def _stream(prompt, **kw):
            for t in tokens:
                yield t

        llm.stream = _stream
        return llm

    @pytest.mark.asyncio
    async def test_stream_explore_yields_tokens(self):
        from synapsekit.llm.cost_quality_router import CostQualityRouter

        llm = self._make_llm("model-a", ["hello", " world"])
        router = CostQualityRouter(candidates=[llm], explore_n=10)

        tokens = []
        async for tok in router.stream("prompt"):
            tokens.append(tok)

        assert tokens == ["hello", " world"]
        assert router._mode == "explore"

    @pytest.mark.asyncio
    async def test_stream_exploit_after_explore_n(self):
        from synapsekit.llm.cost_quality_router import CostQualityRouter

        llm = self._make_llm("model-x", ["ok"])
        router = CostQualityRouter(candidates=[llm], explore_n=0)

        tokens = []
        async for tok in router.stream("prompt"):
            tokens.append(tok)

        assert router._mode == "exploit"
        assert "ok" in tokens

    @pytest.mark.asyncio
    async def test_stream_updates_stats_after_completion(self):
        from synapsekit.llm.cost_quality_router import CostQualityRouter

        llm = self._make_llm("model-z", ["a", "b"])
        router = CostQualityRouter(candidates=[llm], explore_n=5)

        async for _ in router.stream("prompt"):
            pass

        assert router._stats["model-z"]["calls"] == 1

    @pytest.mark.asyncio
    async def test_stream_exception_raises_when_no_candidate_succeeds(self):
        from synapsekit.llm.base import LLMConfig
        from synapsekit.llm.cost_quality_router import CostQualityRouter

        llm = MagicMock()
        llm.config = LLMConfig(model="broken-model", api_key="", provider="openai")
        llm._input_tokens = 0
        llm._output_tokens = 0

        async def _broken_stream(prompt, **kw):
            raise RuntimeError("provider dead")
            yield  # make it a generator

        llm.stream = _broken_stream
        router = CostQualityRouter(candidates=[llm], explore_n=5)

        with pytest.raises(RuntimeError, match="provider dead"):
            async for _ in router.stream("prompt"):
                pass

    @pytest.mark.asyncio
    async def test_stream_uses_exploit_ordering(self):
        from synapsekit.llm.cost_quality_router import CostQualityRouter

        cheap = self._make_llm("cheap", ["cheap-tok"])
        expensive = self._make_llm("expensive", ["exp-tok"])

        router = CostQualityRouter(candidates=[cheap, expensive], explore_n=0)
        # Manually set stats so cheap qualifies and is cheaper
        router._stats["cheap"] = {
            "calls": 5,
            "avg_cost": 0.001,
            "avg_quality": 0.9,
            "_total_quality": 4.5,
            "_quality_calls": 5,
        }
        router._stats["expensive"] = {
            "calls": 5,
            "avg_cost": 0.01,
            "avg_quality": 0.95,
            "_total_quality": 4.75,
            "_quality_calls": 5,
        }

        tokens = []
        async for tok in router.stream("prompt"):
            tokens.append(tok)

        assert "cheap-tok" in tokens

    @pytest.mark.asyncio
    async def test_stream_increments_call_counter(self):
        from synapsekit.llm.cost_quality_router import CostQualityRouter

        llm = self._make_llm("m", ["x"])
        router = CostQualityRouter(candidates=[llm], explore_n=5)
        assert router._calls == 0

        async for _ in router.stream("p"):
            pass

        assert router._calls == 1


# ===========================================================================
# 3. ReasoningLLM — edge cases & delegation
# ===========================================================================


class TestReasoningLLMProviderDetection:
    def _make_llm(self, model: str):
        from synapsekit.llm.reasoning import ReasoningLLM

        with patch("synapsekit.llm.reasoning.ReasoningLLM._build_provider"):
            llm = object.__new__(ReasoningLLM)
            llm._model = model
            return llm

    def test_detect_openai_o1(self):
        from synapsekit.llm.reasoning import ReasoningLLM

        assert ReasoningLLM._detect_provider("o1-preview") == "openai"

    def test_detect_openai_o3(self):
        from synapsekit.llm.reasoning import ReasoningLLM

        assert ReasoningLLM._detect_provider("o3-mini") == "openai"

    def test_detect_anthropic_claude(self):
        from synapsekit.llm.reasoning import ReasoningLLM

        assert ReasoningLLM._detect_provider("claude-3-opus-20240229") == "anthropic"

    def test_detect_google_gemini(self):
        from synapsekit.llm.reasoning import ReasoningLLM

        assert ReasoningLLM._detect_provider("gemini-2.0-flash-thinking") == "google"

    def test_detect_deepseek(self):
        from synapsekit.llm.reasoning import ReasoningLLM

        assert ReasoningLLM._detect_provider("deepseek-r1") == "deepseek"

    def test_detect_qwq_as_qwen(self):
        from synapsekit.llm.reasoning import ReasoningLLM

        # qwq models route through the Qwen/DeepSeek-compatible provider
        assert ReasoningLLM._detect_provider("qwq-32b-preview") == "qwen"

    def test_unsupported_raises(self):
        from synapsekit.llm.reasoning import ReasoningLLM

        with pytest.raises(ValueError, match="Unsupported"):
            ReasoningLLM._detect_provider("some-random-llm")


class TestReasoningLLMDeepSeekStream:
    @pytest.mark.asyncio
    async def test_deepseek_stream_yields_thinking_then_answer(self):
        from synapsekit.llm.providers.deepseek_r1 import DeepSeekR1Reasoning

        class _FakeChoice:
            class delta:  # noqa: N801
                reasoning_content = "thinking"
                content = None

        class _FakeChunk:
            choices = [_FakeChoice()]

        class _FakeClient:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    async def create(*a, **kw):
                        class _Stream:
                            def __aiter__(self):
                                return self

                            _chunks = [_FakeChunk()]
                            _idx = 0

                            async def __anext__(self):
                                if self._idx >= len(self._chunks):
                                    raise StopAsyncIteration
                                chunk = self._chunks[self._idx]
                                self._idx += 1
                                return chunk

                        return _Stream()

        provider = DeepSeekR1Reasoning.__new__(DeepSeekR1Reasoning)
        provider.model = "deepseek-r1"
        provider.thinking = True
        provider.api_key = None
        provider.budget_tokens = None
        provider.provider = "deepseek"
        provider._client = _FakeClient()

        chunks = []
        async for chunk in provider.stream("hello"):
            chunks.append(chunk)

        assert any(c.is_thinking for c in chunks)

    @pytest.mark.asyncio
    async def test_deepseek_stream_wraps_errors(self):
        from synapsekit.llm.providers.deepseek_r1 import DeepSeekR1Reasoning

        class _FakeClient:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    async def create(*a, **kw):
                        raise RuntimeError("api error")

        provider = DeepSeekR1Reasoning.__new__(DeepSeekR1Reasoning)
        provider.model = "deepseek-r1"
        provider.thinking = True
        provider.api_key = None
        provider.budget_tokens = None
        provider.provider = "deepseek"
        provider._client = _FakeClient()

        with pytest.raises(RuntimeError):
            async for _ in provider.stream("prompt"):
                pass


class TestReasoningLLMGoogleProvider:
    @pytest.mark.asyncio
    async def test_google_generate_returns_response(self):
        from synapsekit.llm.providers.google_thinking import GoogleThinking
        from synapsekit.llm.reasoning import ReasoningResponse

        class _FakeModel:
            async def generate_content_async(self, contents, **kwargs):
                class _Part:
                    text = "The answer"

                    def __init__(self, thought=False):
                        self.thought = thought

                class _Candidate:
                    content = type("C", (), {"parts": [_Part(thought=False)]})()
                    usage_metadata = type(
                        "U",
                        (),
                        {"thoughts_token_count": 5, "candidates_token_count": 3},
                    )()

                class _Resp:
                    candidates = [_Candidate()]

                return _Resp()

        provider = GoogleThinking.__new__(GoogleThinking)
        provider.model = "gemini-2.0-flash-thinking"
        provider.budget_tokens = 1024
        provider.thinking = True
        provider.api_key = None
        provider.provider = "google"
        provider._model = _FakeModel()  # the genai model object (lazy-loaded)

        resp = await provider.generate("test question")
        assert isinstance(resp, ReasoningResponse)
        assert resp.answer == "The answer"
        assert resp.provider == "google"

    @pytest.mark.asyncio
    async def test_google_thinking_false_skips_config(self):
        from synapsekit.llm.providers.google_thinking import GoogleThinking

        configs_sent = []

        class _FakeModel:
            async def generate_content_async(self, contents, generation_config=None, **kwargs):
                configs_sent.append(generation_config)

                class _Part:
                    text = "answer"
                    thought = False

                class _Candidate:
                    content = type("C", (), {"parts": [_Part()]})()
                    usage_metadata = type(
                        "U",
                        (),
                        {"thoughts_token_count": 0, "candidates_token_count": 2},
                    )()

                class _Resp:
                    candidates = [_Candidate()]

                return _Resp()

        provider = GoogleThinking.__new__(GoogleThinking)
        provider.model = "gemini-2.0"
        provider.budget_tokens = 1024
        provider.thinking = False
        provider.api_key = None
        provider.provider = "google"
        provider._model = _FakeModel()  # the genai model object (lazy-loaded)

        await provider.generate("q")
        # When thinking=False, no thinkingConfig in generation config
        cfg = configs_sent[0]
        if cfg is not None and hasattr(cfg, "thinking_config"):
            assert cfg.thinking_config is None


# ===========================================================================
# 4. Performance benchmarks
# ===========================================================================


class TestVectorStorePerformance:
    @pytest.mark.asyncio
    async def test_1000_add_and_search_under_3s(self):
        import numpy as np

        from synapsekit.embeddings.backend import SynapsekitEmbeddings
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        class _Emb(SynapsekitEmbeddings):
            async def embed(self, texts):
                return np.random.rand(len(texts), 128).astype(np.float32)

            async def embed_one(self, text):
                return np.random.rand(128).astype(np.float32)

        store = InMemoryVectorStore(_Emb())
        texts = [f"document {i}" for i in range(1000)]
        metas = [{"id": i} for i in range(1000)]

        t0 = time.perf_counter()
        await store.add(texts, metadata=metas)
        results = await store.search("query text", top_k=5)
        elapsed = time.perf_counter() - t0

        assert len(results) == 5
        assert elapsed < 3.0, f"1000-doc add+search took {elapsed:.2f}s (limit 3s)"

    @pytest.mark.asyncio
    async def test_buffer_doubling_no_quadratic_growth(self):
        """Adding N docs should remain O(N), not O(N²)."""
        import numpy as np

        from synapsekit.embeddings.backend import SynapsekitEmbeddings
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        class _Emb(SynapsekitEmbeddings):
            async def embed(self, texts):
                return np.random.rand(len(texts), 64).astype(np.float32)

            async def embed_one(self, text):
                return np.random.rand(64).astype(np.float32)

        store = InMemoryVectorStore(_Emb())

        t0 = time.perf_counter()
        for i in range(0, 500, 50):
            await store.add([f"doc {j}" for j in range(i, i + 50)])
        elapsed = time.perf_counter() - t0

        assert elapsed < 2.0, f"Batched add took {elapsed:.2f}s, possible O(N²)"

    @pytest.mark.asyncio
    async def test_mmr_diversity_lambda_0(self):
        """lambda_mult=0 → pure diversity, results should be spread."""
        import numpy as np

        from synapsekit.embeddings.backend import SynapsekitEmbeddings
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        _vecs = {
            "query": np.array([1.0, 0.0], dtype=np.float32),
            "a": np.array([0.99, 0.01], dtype=np.float32),
            "b": np.array([0.01, 0.99], dtype=np.float32),
            "c": np.array([0.98, 0.02], dtype=np.float32),
        }

        class _Emb(SynapsekitEmbeddings):
            async def embed(self, texts):
                return np.stack([_vecs.get(t, np.random.rand(2).astype(np.float32)) for t in texts])

            async def embed_one(self, text):
                return _vecs.get(text, np.random.rand(2).astype(np.float32))

        store = InMemoryVectorStore(_Emb())
        await store.add(["a", "b", "c"])
        results = await store.search_mmr("query", top_k=2, lambda_mult=0.0)
        texts = [r["text"] for r in results]
        # With pure diversity, should include "b" (most different from query direction)
        assert "b" in texts

    @pytest.mark.asyncio
    async def test_save_load_roundtrip_1k_docs(self, tmp_path):
        import numpy as np

        from synapsekit.embeddings.backend import SynapsekitEmbeddings
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        class _Emb(SynapsekitEmbeddings):
            async def embed(self, texts):
                return np.random.rand(len(texts), 32).astype(np.float32)

            async def embed_one(self, text):
                return np.random.rand(32).astype(np.float32)

        store = InMemoryVectorStore(_Emb())
        await store.add([f"doc {i}" for i in range(1000)])
        path = str(tmp_path / "store.npz")

        t0 = time.perf_counter()
        store.save(path)
        store2 = InMemoryVectorStore(_Emb())
        store2.load(path)
        elapsed = time.perf_counter() - t0

        assert len(store2._texts) == 1000
        assert elapsed < 2.0, f"Save/load 1k docs took {elapsed:.2f}s"


class TestJSONPerformance:
    def test_dumps_10k_ops_under_1s(self):
        from synapsekit._json import dumps

        data = {"model": "gpt-4o", "prompt": "hello world", "tokens": list(range(50))}
        t0 = time.perf_counter()
        for _ in range(10_000):
            dumps(data)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"10k dumps took {elapsed:.2f}s"

    def test_loads_10k_ops_under_1s(self):
        from synapsekit._json import dumps, loads

        data = {"model": "gpt-4o", "prompt": "hello", "nums": list(range(20))}
        serialised = dumps(data)
        t0 = time.perf_counter()
        for _ in range(10_000):
            loads(serialised)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"10k loads took {elapsed:.2f}s"

    def test_dumps_bytes_10k_ops_under_1s(self):
        from synapsekit._json import dumps_bytes

        data = {"key": "value", "nums": list(range(30))}
        t0 = time.perf_counter()
        for _ in range(10_000):
            dumps_bytes(data)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"10k dumps_bytes took {elapsed:.2f}s"


class TestCacheKeyPerformance:
    def test_5k_cache_keys_under_1s(self):
        from synapsekit.llm._cache import AsyncLRUCache

        t0 = time.perf_counter()
        for i in range(5_000):
            AsyncLRUCache.make_key(
                model="gpt-4o",
                prompt_or_messages=f"prompt number {i}",
                temperature=0.7,
                max_tokens=256,
            )
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"5k cache keys took {elapsed:.2f}s"

    def test_cache_key_stable_across_equivalent_inputs(self):
        from synapsekit.llm._cache import AsyncLRUCache

        k1 = AsyncLRUCache.make_key("m", "hello", 0.0, 256)
        k2 = AsyncLRUCache.make_key("m", "hello", 0.0, 256)
        assert k1 == k2

    def test_cache_key_differs_for_different_prompts(self):
        from synapsekit.llm._cache import AsyncLRUCache

        k1 = AsyncLRUCache.make_key("m", "hello", 0.0, 256)
        k2 = AsyncLRUCache.make_key("m", "world", 0.0, 256)
        assert k1 != k2


class TestSplitterPerformance:
    def test_recursive_100k_chars_under_1s(self):
        from synapsekit.text_splitters.recursive import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text = "The quick brown fox jumps over the lazy dog. " * 2500  # ~112k chars
        t0 = time.perf_counter()
        chunks = splitter.split(text)
        elapsed = time.perf_counter() - t0
        assert len(chunks) > 10
        assert elapsed < 1.0, f"100k-char split took {elapsed:.2f}s"

    def test_character_1mb_under_2s(self):
        from synapsekit.text_splitters.character import CharacterTextSplitter

        splitter = CharacterTextSplitter(chunk_size=1000, separator="\n")
        text = "Line of text.\n" * 75_000  # ~1MB
        t0 = time.perf_counter()
        chunks = splitter.split(text)
        elapsed = time.perf_counter() - t0
        assert len(chunks) > 0
        assert elapsed < 2.0, f"1MB char-split took {elapsed:.2f}s"

    def test_splitter_slots_defined(self):
        from synapsekit.text_splitters.recursive import RecursiveCharacterTextSplitter

        # Verify that __slots__ is declared on the class (memory-efficient hot path)
        s = RecursiveCharacterTextSplitter(chunk_size=100)
        assert "__slots__" in type(s).__dict__, (
            "RecursiveCharacterTextSplitter should define __slots__"
        )


class TestAsyncLRUCachePerformance:
    def test_lru_10k_gets_under_1s(self):
        from synapsekit.llm._cache import AsyncLRUCache

        cache = AsyncLRUCache(maxsize=1024)
        cache.put("key", "value")

        t0 = time.perf_counter()
        for _ in range(10_000):
            cache.get("key")
        elapsed = time.perf_counter() - t0

        assert elapsed < 1.0, f"10k LRU gets took {elapsed:.2f}s"

    def test_lru_eviction_maintains_maxsize(self):
        from synapsekit.llm._cache import AsyncLRUCache

        cache = AsyncLRUCache(maxsize=10)
        for i in range(20):
            cache.put(f"k{i}", f"v{i}")

        assert len(cache) == 10


# ===========================================================================
# 5. Integration — cross-feature wiring
# ===========================================================================


class TestCrossFeatureIntegration:
    @pytest.mark.asyncio
    async def test_federated_multiple_fusion_modes_same_sources(self):
        """All three fusion modes return valid results from same sources."""
        from synapsekit.retrieval.federated import FederatedRetriever

        a = _MockRetriever([("doc1", 0.9), ("doc2", 0.7)])
        b = _MockRetriever([("doc3", 0.8), ("doc1", 0.85)])

        for fusion in ("rrf", "score", "interleave"):
            fr = FederatedRetriever(
                sources=[{"retriever": a}, {"retriever": b}],
                fusion=fusion,
                top_k=3,
            )
            results = await fr.retrieve("q")
            assert len(results) >= 1
            assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_cost_quality_router_full_explore_exploit_cycle(self):
        """Router correctly transitions from explore → exploit and picks cheapest."""
        from synapsekit.llm.base import LLMConfig
        from synapsekit.llm.cost_quality_router import CostQualityRouter

        def _make(name, response):
            m = MagicMock()
            m.config = LLMConfig(model=name, api_key="", provider="openai")
            m._input_tokens = 0
            m._output_tokens = 5
            m.generate = AsyncMock(return_value=response)
            return m

        cheap = _make("cheap-model", "cheap answer")
        expensive = _make("expensive-model", "expensive answer")

        router = CostQualityRouter(
            candidates=[cheap, expensive],
            explore_n=2,
            quality_threshold=0.0,
        )

        # Explore phase (2 calls)
        await router.generate("q1")
        await router.generate("q2")
        assert router._mode == "explore"

        # Exploit phase
        # Force cheap to have lower avg_cost
        router._stats["cheap-model"]["avg_cost"] = 0.001
        router._stats["expensive-model"]["avg_cost"] = 0.01
        router._stats["cheap-model"]["avg_quality"] = 0.9
        router._stats["expensive-model"]["avg_quality"] = 0.9

        result = await router.generate("q3")
        assert router._mode == "exploit"
        assert result == "cheap answer"

    @pytest.mark.asyncio
    async def test_prompt_optimizer_picks_highest_score(self):
        """PromptOptimizer returns the variant that scores highest."""
        from unittest.mock import patch as _patch

        from synapsekit.evaluation.optimizer import PromptOptimizer

        scores = {"Be concise": 0.9, "Explain": 0.6}

        async def _mock_eval(prompt: str = "") -> dict:
            for key, score in scores.items():
                if key in prompt:
                    return {"score": score}
            return {"score": 0.5}

        _mock_eval._eval_case_meta = {"name": "mock_eval"}

        llm = AsyncMock()
        opt = PromptOptimizer(
            llm=llm,
            eval_suite=".",
            metric="score",
            variants=["Be concise: summarise this", "Explain: summarise this"],
        )
        with _patch.object(opt, "_load_eval_cases", return_value=[("mock_eval", _mock_eval)]):
            best = await opt.run(
                base_prompt="Summarise this text",
                instructions="Make it concise",
            )
        assert best.text == "Be concise: summarise this"
        assert best.score == pytest.approx(0.9, abs=0.01)

    @pytest.mark.asyncio
    async def test_federated_retriever_dedup_exact_across_3_sources(self):
        """Dedup works correctly when same doc returned by 3 different sources."""
        from synapsekit.retrieval.federated import FederatedRetriever

        shared = _MockRetriever([("the shared document", 0.9)])
        fr = FederatedRetriever(
            sources=[
                {"retriever": shared},
                {"retriever": shared},
                {"retriever": shared},
            ],
            fusion="rrf",
            top_k=5,
        )
        results = await fr.retrieve("q")
        assert results.count("the shared document") == 1

    @pytest.mark.asyncio
    async def test_reasoning_llm_response_fields_complete(self):
        """ReasoningLLM agenerate() response has all required fields."""
        from synapsekit.llm.providers.openai_reasoning import OpenAIReasoning
        from synapsekit.llm.reasoning import ReasoningResponse

        class _FakeCompletion:
            class choices:  # noqa: N801
                pass

            usage = type(
                "U",
                (),
                {"completion_tokens_details": type("D", (), {"reasoning_tokens": 100})()},
            )()

            def __init__(self):
                msg = type(
                    "M",
                    (),
                    {
                        "content": "The answer is 42",
                        "refusal": None,
                    },
                )()
                c = type("C", (), {"message": msg, "finish_reason": "stop"})()
                self.choices = [c]
                self.model = "o1-mini"
                self.usage = type(
                    "U",
                    (),
                    {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "completion_tokens_details": type("D", (), {"reasoning_tokens": 8})(),
                    },
                )()

        class _FakeClient:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    async def create(*a, **kw):
                        return _FakeCompletion()

        provider = OpenAIReasoning.__new__(OpenAIReasoning)
        provider.model = "o1-mini"
        provider.thinking = True
        provider.api_key = None
        provider.budget_tokens = None
        provider.provider = "openai"
        provider._client = _FakeClient()

        resp = await provider.generate("What is 6x7?")
        assert isinstance(resp, ReasoningResponse)
        assert resp.answer == "The answer is 42"
        assert resp.total_tokens > 0
        assert resp.provider == "openai"
