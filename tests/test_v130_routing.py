"""Tests for CostRouter and FallbackChain (v1.3.0)."""

from __future__ import annotations

import pytest

from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.llm.cost_router import (
    QUALITY_TABLE,
    CostRouter,
    CostRouterConfig,
    RouterModelSpec,
)
from synapsekit.llm.fallback_chain import FallbackChain, FallbackChainConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockLLM(BaseLLM):
    """Minimal mock LLM for testing."""

    def __init__(self, model: str = "mock", response: str = "mock response"):
        super().__init__(LLMConfig(model=model, api_key="test", provider="openai"))
        self._response = response

    async def stream(self, prompt, **kw):
        for word in self._response.split(" "):
            yield word + " "

    async def generate(self, prompt, **kw):
        return self._response


class _ErrorLLM(BaseLLM):
    """LLM that always raises."""

    def __init__(self, model: str = "error"):
        super().__init__(LLMConfig(model=model, api_key="test", provider="openai"))

    async def stream(self, prompt, **kw):
        raise RuntimeError("LLM error")
        yield ""  # pragma: no cover

    async def generate(self, prompt, **kw):
        raise RuntimeError("LLM error")


# ===========================================================================
# CostRouter tests
# ===========================================================================


class TestCostRouter:
    def test_quality_table_populated(self):
        assert len(QUALITY_TABLE) > 10
        assert "gpt-4o" in QUALITY_TABLE
        assert 0 < QUALITY_TABLE["gpt-4o"] <= 1.0

    def test_rank_candidates_filters_by_quality(self):
        config = CostRouterConfig(
            models=[
                RouterModelSpec(model="gpt-4o", api_key="k"),
                RouterModelSpec(model="gpt-4.1-nano", api_key="k"),
            ],
            quality_threshold=0.85,
        )
        router = CostRouter(config)
        # gpt-4.1-nano has quality 0.65, should be filtered out
        assert len(router._candidates) == 1
        assert router._candidates[0].model == "gpt-4o"

    def test_rank_candidates_sorts_cheapest_first(self):
        config = CostRouterConfig(
            models=[
                RouterModelSpec(model="gpt-4o", api_key="k"),
                RouterModelSpec(model="gpt-4o-mini", api_key="k"),
            ],
            quality_threshold=0.0,
        )
        router = CostRouter(config)
        assert router._candidates[0].model == "gpt-4o-mini"

    async def test_generate_uses_cheapest(self):
        config = CostRouterConfig(
            models=[
                RouterModelSpec(model="gpt-4o", api_key="k"),
                RouterModelSpec(model="gpt-4o-mini", api_key="k"),
            ],
            quality_threshold=0.0,
        )
        router = CostRouter(config)
        # Patch _build_llm to return mock
        router._build_llm = lambda spec: _MockLLM(spec.model, f"answer from {spec.model}")

        result = await router.generate("test")
        assert "gpt-4o-mini" in result
        assert router.selected_model == "gpt-4o-mini"

    async def test_stream_delegation(self):
        config = CostRouterConfig(
            models=[RouterModelSpec(model="gpt-4o-mini", api_key="k")],
        )
        router = CostRouter(config)
        router._build_llm = lambda spec: _MockLLM(spec.model, "streamed response")

        tokens = []
        async for token in router.stream("test"):
            tokens.append(token)
        assert "".join(tokens).strip() == "streamed response"

    async def test_fallback_on_error(self):
        config = CostRouterConfig(
            models=[
                RouterModelSpec(model="gpt-4o-mini", api_key="k"),
                RouterModelSpec(model="gpt-4o", api_key="k"),
            ],
            fallback_on_error=True,
        )
        router = CostRouter(config)

        call_count = 0

        def mock_build(spec):
            nonlocal call_count
            call_count += 1
            if spec.model == "gpt-4o-mini":
                return _ErrorLLM(spec.model)
            return _MockLLM(spec.model, "fallback answer")

        router._build_llm = mock_build
        result = await router.generate("test")
        assert result == "fallback answer"
        assert router.selected_model == "gpt-4o"

    async def test_no_fallback_raises(self):
        config = CostRouterConfig(
            models=[RouterModelSpec(model="gpt-4o-mini", api_key="k")],
            fallback_on_error=False,
        )
        router = CostRouter(config)
        router._build_llm = lambda spec: _ErrorLLM(spec.model)

        with pytest.raises(RuntimeError):
            await router.generate("test")

    def test_selected_model_initially_none(self):
        config = CostRouterConfig(
            models=[RouterModelSpec(model="gpt-4o-mini", api_key="k")],
        )
        router = CostRouter(config)
        assert router.selected_model is None

    def test_drop_in_compatibility(self):
        """CostRouter is a BaseLLM subclass."""
        config = CostRouterConfig(
            models=[RouterModelSpec(model="gpt-4o-mini", api_key="k")],
        )
        router = CostRouter(config)
        assert isinstance(router, BaseLLM)


# ===========================================================================
# FallbackChain tests
# ===========================================================================


class TestFallbackChain:
    async def test_uses_first_successful_model(self):
        chain = FallbackChain(
            FallbackChainConfig(
                models=[_MockLLM("a", "answer A"), _MockLLM("b", "answer B")],
            )
        )
        result = await chain.generate("test")
        assert result == "answer A"
        assert chain.used_model is not None
        assert chain.used_model.config.model == "a"

    async def test_escalates_on_error(self):
        chain = FallbackChain(
            FallbackChainConfig(
                models=[_ErrorLLM("bad"), _MockLLM("good", "ok")],
            )
        )
        result = await chain.generate("test")
        assert result == "ok"

    async def test_escalates_on_short_response(self):
        chain = FallbackChain(
            FallbackChainConfig(
                models=[_MockLLM("a", ""), _MockLLM("b", "long enough response")],
                min_response_length=5,
            )
        )
        result = await chain.generate("test")
        assert result == "long enough response"

    async def test_stream_with_fallback(self):
        chain = FallbackChain(
            FallbackChainConfig(
                models=[_ErrorLLM("bad"), _MockLLM("good", "streamed ok")],
            )
        )
        tokens = []
        async for token in chain.stream("test"):
            tokens.append(token)
        assert "streamed ok" in "".join(tokens)

    async def test_all_fail_raises(self):
        chain = FallbackChain(FallbackChainConfig(models=[_ErrorLLM("a"), _ErrorLLM("b")]))
        with pytest.raises(RuntimeError):
            await chain.generate("test")

    async def test_drop_in_compatibility(self):
        chain = FallbackChain(FallbackChainConfig(models=[_MockLLM()]))
        assert isinstance(chain, BaseLLM)
