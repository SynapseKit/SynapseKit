from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from synapsekit.llm import EffortLevel as PublicEffortLevel
from synapsekit.llm import classify_effort as public_classify_effort
from synapsekit.llm.effort import EffortLevel, classify_effort, resolve_effort
from synapsekit.llm.providers.anthropic_thinking import AnthropicThinking
from synapsekit.llm.providers.google_thinking import GoogleThinking
from synapsekit.llm.providers.openai_reasoning import OpenAIReasoning
from synapsekit.llm.reasoning import ReasoningLLM


def test_effort_api_is_exported_from_llm_package() -> None:
    assert PublicEffortLevel is EffortLevel
    assert public_classify_effort is classify_effort


def test_effort_levels_are_provider_agnostic() -> None:
    assert [level.value for level in EffortLevel] == ["low", "medium", "high"]


def test_default_classifier_is_deterministic_for_simple_medium_and_hard_prompts() -> None:
    prompts = [
        ("What is 2 + 2?", EffortLevel.LOW),
        ("Explain how a queue works and give a short example.", EffortLevel.MEDIUM),
        (
            "Compare breadth-first and depth-first traversal, derive their complexity, "
            "and recommend one for shortest paths.",
            EffortLevel.HIGH,
        ),
    ]

    for prompt, expected in prompts:
        assert classify_effort(prompt) is expected
        assert classify_effort(prompt) is expected


@pytest.mark.parametrize(
    ("word_count", "expected"),
    [
        (20, EffortLevel.LOW),
        (21, EffortLevel.MEDIUM),
        (69, EffortLevel.MEDIUM),
        (70, EffortLevel.HIGH),
    ],
)
def test_default_classifier_word_count_thresholds(word_count, expected) -> None:
    prompt = " ".join(["item"] * word_count)

    assert classify_effort(prompt) is expected


def test_proof_tasks_are_classified_as_high_effort() -> None:
    assert classify_effort("Give a proof that square root of 2 is irrational.") is EffortLevel.HIGH


def test_auto_effort_uses_injected_classifier_for_each_prompt() -> None:
    seen: list[str] = []

    def classifier(prompt: str) -> EffortLevel:
        seen.append(prompt)
        return EffortLevel.HIGH if "proof" in prompt else EffortLevel.LOW

    assert resolve_effort("auto", "Give a proof", classifier) is EffortLevel.HIGH
    assert resolve_effort("auto", "Say hello", classifier) is EffortLevel.LOW
    assert seen == ["Give a proof", "Say hello"]


@pytest.mark.parametrize(
    "classifier_error",
    [TypeError("classifier failed"), ValueError("classifier failed")],
)
def test_auto_effort_propagates_classifier_exceptions(classifier_error) -> None:
    def classifier(prompt: str) -> EffortLevel:
        del prompt
        raise classifier_error

    with pytest.raises(type(classifier_error), match="classifier failed") as exc_info:
        resolve_effort("auto", "Explain this", classifier)

    assert exc_info.value is classifier_error


def test_auto_effort_rejects_invalid_classifier_return_value() -> None:
    with pytest.raises(
        ValueError,
        match="effort_classifier must return EffortLevel low, medium, or high",
    ):
        resolve_effort("auto", "Explain this", lambda prompt: "urgent")


def test_effort_conflicts_with_legacy_budget_tokens() -> None:
    with pytest.raises(ValueError, match="cannot be used together"):
        ReasoningLLM("o3", budget_tokens=512, effort=EffortLevel.LOW)


@pytest.mark.parametrize(
    ("provider_class", "model"),
    [
        (OpenAIReasoning, "o3"),
        (AnthropicThinking, "claude-opus-4-6"),
        (GoogleThinking, "gemini-2.5-pro"),
    ],
)
def test_providers_reject_effort_when_thinking_is_disabled(provider_class, model) -> None:
    with pytest.raises(ValueError, match="effort cannot be used when thinking is disabled"):
        provider_class(model=model, thinking=False, effort=EffortLevel.LOW)


@pytest.mark.parametrize("model", ["o3", "claude-opus-4-6", "gemini-2.5-pro"])
def test_reasoning_llm_rejects_effort_when_thinking_is_disabled(model) -> None:
    with pytest.raises(ValueError, match="effort cannot be used when thinking is disabled"):
        ReasoningLLM(model, thinking=False, effort=EffortLevel.LOW)


def test_effort_is_rejected_for_reasoning_providers_without_a_supported_knob() -> None:
    with pytest.raises(ValueError, match="supports effort routing only for"):
        ReasoningLLM("deepseek-r1", effort=EffortLevel.HIGH)


@pytest.mark.asyncio
async def test_openai_effort_is_sent_to_generate_and_stream_requests() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer", reasoning=None))],
        usage=SimpleNamespace(
            completion_tokens=2,
            total_tokens=2,
            completion_tokens_details=None,
        ),
    )

    class _FakeStream:
        def __aiter__(self):
            return self._chunks()

        async def _chunks(self):
            yield SimpleNamespace(choices=[])

    create = AsyncMock(side_effect=[response, _FakeStream()])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider = OpenAIReasoning(model="o3", effort=EffortLevel.HIGH)
    provider._get_client = lambda: client

    await provider.generate("Solve this")
    chunks = [chunk async for chunk in provider.stream("Solve this")]

    assert chunks == []
    assert [call.kwargs["reasoning_effort"] for call in create.call_args_list] == [
        "high",
        "high",
    ]


@pytest.mark.asyncio
async def test_auto_effort_classifies_the_actual_openai_prompt() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer", reasoning=None))],
        usage=SimpleNamespace(
            completion_tokens=2,
            total_tokens=2,
            completion_tokens_details=None,
        ),
    )
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    classified: list[str] = []

    def classifier(prompt: str) -> EffortLevel:
        classified.append(prompt)
        return EffortLevel.MEDIUM

    provider = OpenAIReasoning(model="o3", effort="auto", effort_classifier=classifier)
    provider._get_client = lambda: client

    await provider.generate("Explain this")

    assert classified == ["Explain this"]
    assert create.call_args.kwargs["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_anthropic_effort_uses_adaptive_thinking_and_output_config() -> None:
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="answer")],
        usage=SimpleNamespace(input_tokens=3, output_tokens=2, thinking_tokens=0, total_tokens=5),
    )
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    provider = AnthropicThinking(model="claude-opus-4-6", effort=EffortLevel.LOW)
    provider._get_client = lambda: client

    await provider.generate("Answer simply")

    kwargs = create.call_args.kwargs
    assert kwargs["output_config"] == {"effort": "low"}
    assert kwargs["thinking"] == {"type": "adaptive"}


def test_legacy_provider_budget_behavior_is_unchanged_when_effort_is_omitted() -> None:
    openai_provider = OpenAIReasoning(model="o3", budget_tokens=256)
    anthropic_provider = AnthropicThinking(model="claude-3-7-sonnet", budget_tokens=512)
    google_provider = GoogleThinking(model="gemini-2.5-pro", budget_tokens=768)

    assert openai_provider._reasoning_effort() == "low"
    assert anthropic_provider._thinking_payload() == {
        "type": "enabled",
        "budget_tokens": 512,
    }
    assert google_provider._generation_config() == {"thinkingConfig": {"thinkingBudget": 768}}


@pytest.mark.parametrize("reasoner_class", [OpenAIReasoning, ReasoningLLM])
def test_uninitialized_reasoner_defaults_to_legacy_effort(reasoner_class) -> None:
    provider = reasoner_class.__new__(reasoner_class)

    assert provider._effort_for_prompt("Explain this") is None


@pytest.mark.asyncio
async def test_google_generate_defaults_to_legacy_budget_without_effort_attributes() -> None:
    response = SimpleNamespace(
        candidates=[],
        text="answer",
        usage_metadata=SimpleNamespace(
            candidates_token_count=2,
            thoughts_token_count=1,
            total_token_count=3,
        ),
    )
    generate = AsyncMock(return_value=response)
    provider = GoogleThinking.__new__(GoogleThinking)
    provider.model = "gemini-2.5-pro"
    provider.budget_tokens = 768
    provider.thinking = True
    provider.api_key = None
    provider.provider = "google"
    provider._model = SimpleNamespace(generate_content_async=generate)

    await provider.generate("Explain this")

    assert generate.call_args.kwargs["generation_config"] == {
        "thinkingConfig": {"thinkingBudget": 768}
    }


@pytest.mark.asyncio
async def test_google_stream_defaults_to_legacy_budget_without_effort_attributes() -> None:
    class _FakeStream:
        def __aiter__(self):
            return self._chunks()

        async def _chunks(self):
            yield SimpleNamespace(text="answer")

    generate = AsyncMock(return_value=_FakeStream())
    provider = GoogleThinking.__new__(GoogleThinking)
    provider.model = "gemini-2.5-pro"
    provider.budget_tokens = 768
    provider.thinking = True
    provider.api_key = None
    provider.provider = "google"
    provider._model = SimpleNamespace(generate_content_async=generate)

    chunks = [chunk async for chunk in provider.stream("Explain this")]

    assert [chunk.text for chunk in chunks] == ["answer"]
    assert generate.call_args.kwargs["generation_config"] == {
        "thinkingConfig": {"thinkingBudget": 768}
    }


@pytest.mark.asyncio
async def test_google_effort_uses_google_genai_thinking_config() -> None:
    class _HttpOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _ThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _GenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_types = SimpleNamespace(
        HttpOptions=_HttpOptions,
        ThinkingConfig=_ThinkingConfig,
        GenerateContentConfig=_GenerateContentConfig,
    )
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    fake_genai = types.ModuleType("google.genai")
    fake_genai.types = fake_types
    fake_google.genai = fake_genai

    response = SimpleNamespace(
        candidates=[],
        text="answer",
        usage_metadata=SimpleNamespace(
            candidates_token_count=2,
            thoughts_token_count=5,
            total_token_count=7,
        ),
    )
    generate = AsyncMock(return_value=response)
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    provider = GoogleThinking(model="gemini-2.5-pro", effort=EffortLevel.MEDIUM)
    provider._get_effort_client = lambda: client

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "google", fake_google)
        monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
        result = await provider.generate("Explain this")

    assert result.answer == "answer"
    config = generate.call_args.kwargs["config"]
    assert config.kwargs["thinking_config"].kwargs == {"thinking_budget": 1024}
    assert config.kwargs["http_options"].kwargs == {"timeout": 30_000}


@pytest.mark.asyncio
async def test_reasoning_llm_tool_call_uses_explicit_openai_effort() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=[]),
            )
        ],
        usage=SimpleNamespace(completion_tokens=1, total_tokens=2),
    )
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm = ReasoningLLM("o3", effort=EffortLevel.HIGH)
    llm._get_openai_client = lambda: client

    await llm._openai_call_with_tools([{"role": "user", "content": "Calculate"}], [])

    assert create.call_args.kwargs["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_google_effort_uses_thinking_budget_for_streaming() -> None:
    class _HttpOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _ThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _GenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeStream:
        def __aiter__(self):
            return self._chunks()

        async def _chunks(self):
            yield SimpleNamespace(text="answer")

    fake_types = SimpleNamespace(
        HttpOptions=_HttpOptions,
        ThinkingConfig=_ThinkingConfig,
        GenerateContentConfig=_GenerateContentConfig,
    )
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    fake_genai = types.ModuleType("google.genai")
    fake_genai.types = fake_types
    fake_google.genai = fake_genai

    generate_stream = AsyncMock(return_value=_FakeStream())
    client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=generate_stream))
    )
    provider = GoogleThinking(model="gemini-2.5-pro", effort=EffortLevel.LOW)
    provider._get_effort_client = lambda: client

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "google", fake_google)
        monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
        chunks = [chunk async for chunk in provider.stream("Short reply")]

    assert [chunk.text for chunk in chunks] == ["answer"]
    generate_stream.assert_awaited_once()
    config = generate_stream.call_args.kwargs["config"]
    assert config.kwargs["thinking_config"].kwargs == {"thinking_budget": 512}
    assert config.kwargs["http_options"].kwargs == {"timeout": 30_000}


@pytest.mark.asyncio
async def test_anthropic_effort_uses_output_config_for_streaming() -> None:
    class _FakeStream:
        def __aiter__(self):
            return self._events()

        async def _events(self):
            yield SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="answer"),
            )

    create = AsyncMock(return_value=_FakeStream())
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    provider = AnthropicThinking(model="claude-opus-4-6", effort=EffortLevel.HIGH)
    provider._get_client = lambda: client

    chunks = [chunk async for chunk in provider.stream("Analyze carefully")]

    assert [chunk.text for chunk in chunks] == ["answer"]
    assert create.call_args.kwargs["output_config"] == {"effort": "high"}
    assert create.call_args.kwargs["thinking"] == {"type": "adaptive"}


@pytest.mark.asyncio
async def test_reasoning_llm_anthropic_tool_call_uses_effort_controls() -> None:
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="done")],
        usage=SimpleNamespace(input_tokens=2, output_tokens=1, thinking_tokens=0),
    )
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    llm = ReasoningLLM("claude-opus-4-6", effort=EffortLevel.LOW)
    llm._get_anthropic_client = lambda: client

    await llm._anthropic_call_with_tools([{"role": "user", "content": "Answer simply"}], [])

    kwargs = create.call_args.kwargs
    assert kwargs["output_config"] == {"effort": "low"}
    assert kwargs["thinking"] == {"type": "adaptive"}


@pytest.mark.asyncio
async def test_reasoning_llm_passes_auto_classifier_to_openai_provider() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer", reasoning=None))],
        usage=SimpleNamespace(
            completion_tokens=2,
            total_tokens=2,
            completion_tokens_details=None,
        ),
    )
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    classified: list[str] = []

    def classifier(prompt: str) -> EffortLevel:
        classified.append(prompt)
        return EffortLevel.LOW

    llm = ReasoningLLM("o3", effort="auto", effort_classifier=classifier)
    llm._provider._get_client = lambda: client
    prompt = "Compare these two algorithms."

    await llm._provider.generate(prompt)

    assert classified == [prompt]
    assert create.call_args.kwargs["reasoning_effort"] == "low"
