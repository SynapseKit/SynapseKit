"""Tests for PromptRepairMiddleware."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.llm.repair import (
    _REFUSAL_PHRASES,
    _REPAIR_INSTRUCTIONS,
    PromptRepairMiddleware,
    RepairEvent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kw: Any) -> LLMConfig:
    defaults = dict(model="test-model", api_key="test-key", provider="test")
    defaults.update(kw)
    return LLMConfig(**defaults)  # type: ignore[arg-type]


class _FakeLLM(BaseLLM):
    """A minimal BaseLLM stub whose generate() returns pre-configured responses."""

    def __init__(self, responses: list[str], config: LLMConfig | None = None) -> None:
        super().__init__(config or _make_config())
        self._responses = list(responses)
        self._call_count = 0
        self.received_prompts: list[str] = []

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
        self.received_prompts.append(prompt)
        response = (
            self._responses[self._call_count]
            if self._call_count < len(self._responses)
            else self._responses[-1]
        )
        self._call_count += 1
        yield response


def _wrap(responses: list[str], **kw: Any) -> tuple[PromptRepairMiddleware, _FakeLLM]:
    inner = _FakeLLM(responses)
    middleware = PromptRepairMiddleware(inner, **kw)
    return middleware, inner


# ===========================================================================
# 1. Initialisation / validation
# ===========================================================================


class TestInit:
    def test_negative_max_retries_raises(self) -> None:
        inner = _FakeLLM(["ok"])
        with pytest.raises(ValueError, match="max_retries"):
            PromptRepairMiddleware(inner, max_retries=-1)

    def test_zero_max_retries_is_valid(self) -> None:
        inner = _FakeLLM(["ok"])
        mw = PromptRepairMiddleware(inner, max_retries=0)
        assert mw._max_retries == 0

    def test_default_detect_includes_all_modes(self) -> None:
        inner = _FakeLLM(["ok"])
        mw = PromptRepairMiddleware(inner)
        assert {"empty", "refusal", "json_error", "tool_loop"} == mw._detect

    def test_custom_detect_limits_modes(self) -> None:
        inner = _FakeLLM(["ok"])
        mw = PromptRepairMiddleware(inner, detect=["empty"])
        assert mw._detect == {"empty"}

    def test_expect_json_adds_json_error_mode(self) -> None:
        inner = _FakeLLM(["ok"])
        mw = PromptRepairMiddleware(inner, detect=["empty"], expect_json=True)
        assert "json_error" in mw._detect

    def test_custom_repair_instruction_merges(self) -> None:
        inner = _FakeLLM(["ok"])
        custom = {"empty": "CUSTOM_EMPTY_INSTRUCTION"}
        mw = PromptRepairMiddleware(inner, repair_instructions=custom)
        assert mw._repair_instructions["empty"] == "CUSTOM_EMPTY_INSTRUCTION"
        # Other defaults should still be present.
        assert "refusal" in mw._repair_instructions

    def test_repr_contains_key_info(self) -> None:
        inner = _FakeLLM(["ok"])
        mw = PromptRepairMiddleware(inner, max_retries=2, detect=["empty"])
        r = repr(mw)
        assert "PromptRepairMiddleware" in r
        assert "max_retries=2" in r
        assert "empty" in r


# ===========================================================================
# 2. Failure-mode detection helpers
# ===========================================================================


class TestDetectFailure:
    def test_empty_string_detected(self) -> None:
        mw, _ = _wrap(["ok"])
        assert mw._detect_failure("") == "empty"

    def test_whitespace_only_detected(self) -> None:
        mw, _ = _wrap(["ok"])
        assert mw._detect_failure("   \n\t  ") == "empty"

    def test_non_empty_response_not_empty(self) -> None:
        mw, _ = _wrap(["ok"])
        assert mw._detect_failure("hello") != "empty"

    @pytest.mark.parametrize("phrase", list(_REFUSAL_PHRASES)[:6])
    def test_refusal_phrases_detected(self, phrase: str) -> None:
        mw, _ = _wrap(["ok"])
        response = f"{phrase.capitalize()}, I cannot do that."
        assert mw._detect_failure(response) == "refusal"

    def test_normal_response_not_refusal(self) -> None:
        mw, _ = _wrap(["ok"])
        assert mw._detect_failure("The answer is 42.") != "refusal"

    def test_valid_json_not_detected_as_error(self) -> None:
        mw, _ = _wrap(["ok"])
        assert mw._detect_failure('{"name": "Alice", "age": 30}') != "json_error"

    def test_truncated_json_detected(self) -> None:
        mw, _ = _wrap(["ok"])
        assert mw._detect_failure('{"name": "Alice"') == "json_error"

    def test_json_array_valid(self) -> None:
        mw, _ = _wrap(["ok"])
        assert mw._detect_failure("[1, 2, 3]") != "json_error"

    def test_markdown_fenced_json_valid(self) -> None:
        mw, _ = _wrap(["ok"])
        resp = '```json\n{"key": "value"}\n```'
        assert mw._detect_failure(resp) != "json_error"

    def test_markdown_fenced_broken_json_detected(self) -> None:
        mw, _ = _wrap(["ok"])
        resp = '```json\n{"key":\n```'
        assert mw._detect_failure(resp) == "json_error"

    def test_empty_takes_precedence_over_refusal(self) -> None:
        mw, _ = _wrap(["ok"])
        # Empty string — should be caught as empty before refusal check.
        assert mw._detect_failure("") == "empty"

    def test_disabled_mode_not_detected(self) -> None:
        inner = _FakeLLM(["ok"])
        mw = PromptRepairMiddleware(inner, detect=["refusal"])  # empty disabled
        # Blank response — "empty" mode is off, "json_error" is off.
        assert mw._detect_failure("") is None  # not detected

    def test_json_mode_off_ignores_bad_json(self) -> None:
        inner = _FakeLLM(["ok"])
        mw = PromptRepairMiddleware(inner, detect=["empty", "refusal"])
        assert mw._detect_failure('{"broken":') is None


# ===========================================================================
# 3. generate() — repair loop behaviour
# ===========================================================================


class TestGenerateRepairLoop:
    @pytest.mark.asyncio
    async def test_good_response_returned_immediately(self) -> None:
        mw, inner = _wrap(["Good answer."], detect=["empty", "refusal"])
        result = await mw.generate("Tell me something.")
        assert result == "Good answer."
        assert inner._call_count == 1

    @pytest.mark.asyncio
    async def test_empty_response_triggers_repair_and_retries(self) -> None:
        # First call returns empty, second returns valid.
        mw, inner = _wrap(["", "Fixed answer."], detect=["empty"])
        result = await mw.generate("Tell me something.")
        assert result == "Fixed answer."
        assert inner._call_count == 2

    @pytest.mark.asyncio
    async def test_repair_suffix_appended_to_prompt(self) -> None:
        mw, inner = _wrap(["", "ok"])
        await mw.generate("Initial prompt.")
        # Second call should have the repair instruction appended.
        assert inner.received_prompts[1].startswith("Initial prompt.")
        assert _REPAIR_INSTRUCTIONS["empty"] in inner.received_prompts[1]

    @pytest.mark.asyncio
    async def test_refusal_triggers_repair(self) -> None:
        mw, inner = _wrap(
            ["I'm sorry, I cannot help with that.", "Here is your data."],
            detect=["refusal"],
        )
        result = await mw.generate("Extract names from the document.")
        assert result == "Here is your data."
        assert inner._call_count == 2

    @pytest.mark.asyncio
    async def test_refusal_repair_suffix_appended(self) -> None:
        mw, inner = _wrap(["I'm sorry, I cannot help.", "ok"])
        await mw.generate("prompt")
        assert _REPAIR_INSTRUCTIONS["refusal"] in inner.received_prompts[1]

    @pytest.mark.asyncio
    async def test_json_error_triggers_repair(self) -> None:
        mw, inner = _wrap(['{"name": "Alice"', '{"name": "Alice"}'], expect_json=True)
        result = await mw.generate("Return JSON.")
        assert result == '{"name": "Alice"}'
        assert inner._call_count == 2

    @pytest.mark.asyncio
    async def test_json_error_repair_suffix_appended(self) -> None:
        mw, inner = _wrap(['{"bad":', '{"ok": true}'], expect_json=True)
        await mw.generate("prompt")
        assert _REPAIR_INSTRUCTIONS["json_error"] in inner.received_prompts[1]

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self) -> None:
        # Always returns empty — should give up after max_retries.
        mw, inner = _wrap([""], max_retries=2)
        result = await mw.generate("prompt")
        # Returns the last bad response.
        assert result == ""
        # 1 original + 2 retries = 3 total calls.
        assert inner._call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_zero_no_repair(self) -> None:
        mw, inner = _wrap([""], max_retries=0)
        result = await mw.generate("prompt")
        assert result == ""
        assert inner._call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_failures_before_success(self) -> None:
        mw, inner = _wrap(["", "", "success"], max_retries=3, detect=["empty"])
        result = await mw.generate("prompt")
        assert result == "success"
        assert inner._call_count == 3

    @pytest.mark.asyncio
    async def test_on_repair_callback_called(self) -> None:
        events: list[RepairEvent] = []
        mw, _ = _wrap(["", "ok"], on_repair=events.append, detect=["empty"])
        await mw.generate("prompt")
        assert len(events) == 1
        assert events[0].failure_mode == "empty"
        assert events[0].attempt == 1

    @pytest.mark.asyncio
    async def test_on_repair_callback_receives_correct_event(self) -> None:
        events: list[RepairEvent] = []
        mw, _ = _wrap(["", "ok"], on_repair=events.append)
        await mw.generate("my prompt")
        ev = events[0]
        assert ev.original_response == ""
        assert "my prompt" in ev.repaired_prompt
        assert _REPAIR_INSTRUCTIONS["empty"] in ev.repaired_prompt

    @pytest.mark.asyncio
    async def test_on_repair_callback_exception_does_not_crash(self) -> None:
        def _bad_callback(ev: RepairEvent) -> None:
            raise RuntimeError("callback exploded")

        mw, _ = _wrap(["", "ok"], on_repair=_bad_callback)
        result = await mw.generate("prompt")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_logging_on_give_up(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        mw, _ = _wrap([""], max_retries=1)
        with caplog.at_level(logging.WARNING, logger="synapsekit.llm.repair"):
            await mw.generate("prompt")
        assert "giving up" in caplog.text.lower()


# ===========================================================================
# 4. Tool-loop detection
# ===========================================================================


class TestToolLoopDetection:
    def test_no_loop_below_threshold(self) -> None:
        mw, _ = _wrap(["ok"], tool_loop_threshold=3)
        mw.record_tool_call("search", {"q": "python"})
        mw.record_tool_call("search", {"q": "python"})
        # Only 2 — threshold is 3.
        assert not mw.record_tool_call("search", {"q": "different"})

    def test_loop_detected_at_threshold(self) -> None:
        mw, _ = _wrap(["ok"], tool_loop_threshold=3)
        mw.record_tool_call("search", {"q": "python"})
        mw.record_tool_call("search", {"q": "python"})
        # Third identical call hits the threshold.
        assert mw.record_tool_call("search", {"q": "python"})

    def test_different_args_reset_consecutive_count(self) -> None:
        mw, _ = _wrap(["ok"], tool_loop_threshold=3)
        mw.record_tool_call("search", {"q": "python"})
        mw.record_tool_call("search", {"q": "python"})
        mw.record_tool_call("search", {"q": "java"})  # different args — breaks streak
        # Even if we call "python" again it's not 3 consecutive.
        assert not mw.record_tool_call("search", {"q": "python"})

    def test_different_tool_name_does_not_count(self) -> None:
        mw, _ = _wrap(["ok"], tool_loop_threshold=3)
        mw.record_tool_call("search", {"q": "x"})
        mw.record_tool_call("calculator", {"q": "x"})
        assert not mw.record_tool_call("search", {"q": "x"})

    def test_reset_tool_history_clears_state(self) -> None:
        mw, _ = _wrap(["ok"], tool_loop_threshold=3)
        mw.record_tool_call("search", {"q": "x"})
        mw.record_tool_call("search", {"q": "x"})
        mw.reset_tool_history()
        # History cleared — not a loop.
        assert not mw.record_tool_call("search", {"q": "x"})

    def test_tool_loop_mode_off_never_detects(self) -> None:
        inner = _FakeLLM(["ok"])
        mw = PromptRepairMiddleware(inner, detect=["empty"])  # tool_loop off
        mw.record_tool_call("search", {"q": "x"})
        mw.record_tool_call("search", {"q": "x"})
        assert not mw.record_tool_call("search", {"q": "x"})

    def test_loop_repair_instruction_property(self) -> None:
        mw, _ = _wrap(["ok"])
        assert mw.loop_repair_instruction == _REPAIR_INSTRUCTIONS["tool_loop"]

    def test_custom_loop_instruction(self) -> None:
        inner = _FakeLLM(["ok"])
        mw = PromptRepairMiddleware(inner, repair_instructions={"tool_loop": "STOP LOOPING!"})
        assert mw.loop_repair_instruction == "STOP LOOPING!"


# ===========================================================================
# 5. Passthrough / interface contract
# ===========================================================================


class TestPassthrough:
    @pytest.mark.asyncio
    async def test_stream_delegates_to_inner_llm(self) -> None:
        inner = _FakeLLM(["hello world"])
        mw = PromptRepairMiddleware(inner)
        tokens = [t async for t in mw.stream("prompt")]
        assert "".join(tokens) == "hello world"

    def test_tokens_used_mirrors_inner_llm(self) -> None:
        inner = _FakeLLM(["ok"])
        inner._input_tokens = 10
        inner._output_tokens = 5
        mw = PromptRepairMiddleware(inner)
        assert mw.tokens_used == {"input": 10, "output": 5}

    def test_supports_multimodal_mirrors_inner(self) -> None:
        inner = _FakeLLM(["ok"])
        inner.supports_multimodal = True  # type: ignore[attr-defined]
        mw = PromptRepairMiddleware(inner)
        assert mw.supports_multimodal is True

    def test_supports_multimodal_false_by_default(self) -> None:
        inner = _FakeLLM(["ok"])
        mw = PromptRepairMiddleware(inner)
        assert mw.supports_multimodal is False

    @pytest.mark.asyncio
    async def test_generate_with_messages_delegates(self) -> None:
        inner = _FakeLLM(["ok"])
        inner.generate_with_messages = AsyncMock(return_value="msg_response")  # type: ignore
        mw = PromptRepairMiddleware(inner)
        result = await mw.generate_with_messages([{"role": "user", "content": "hi"}])
        assert result == "msg_response"

    @pytest.mark.asyncio
    async def test_call_with_tools_delegates(self) -> None:
        inner = _FakeLLM(["ok"])
        inner.call_with_tools = AsyncMock(return_value={"tool": "result"})  # type: ignore
        mw = PromptRepairMiddleware(inner)
        result = await mw.call_with_tools([], [])
        assert result == {"tool": "result"}

    @pytest.mark.asyncio
    async def test_aclose_delegates(self) -> None:
        inner = _FakeLLM(["ok"])
        inner.aclose = AsyncMock()  # type: ignore
        mw = PromptRepairMiddleware(inner)
        await mw.aclose()
        inner.aclose.assert_awaited_once()

    def test_config_property_returns_inner_config(self) -> None:
        cfg = _make_config(model="gpt-4o")
        inner = _FakeLLM(["ok"], config=cfg)
        mw = PromptRepairMiddleware(inner)
        assert mw.config.model == "gpt-4o"


# ===========================================================================
# 6. RepairEvent dataclass
# ===========================================================================


class TestRepairEvent:
    def test_fields_are_set(self) -> None:
        ev = RepairEvent(
            attempt=1,
            failure_mode="empty",
            original_response="",
            repaired_prompt="Try again.",
        )
        assert ev.attempt == 1
        assert ev.failure_mode == "empty"
        assert ev.original_response == ""
        assert ev.repaired_prompt == "Try again."


# ===========================================================================
# 7. Integration — full repair cycle via public import
# ===========================================================================


class TestPublicImport:
    def test_importable_from_synapsekit(self) -> None:
        from synapsekit import PromptRepairMiddleware as PRM
        from synapsekit import RepairEvent as RE

        assert PRM is PromptRepairMiddleware
        assert RE is RepairEvent

    def test_importable_from_synapsekit_llm(self) -> None:
        from synapsekit.llm import PromptRepairMiddleware as PRM
        from synapsekit.llm import RepairEvent as RE

        assert PRM is PromptRepairMiddleware
        assert RE is RepairEvent

    @pytest.mark.asyncio
    async def test_end_to_end_empty_then_valid(self) -> None:
        """Full cycle: empty response → repair → valid response."""
        from synapsekit import PromptRepairMiddleware

        inner = _FakeLLM(["", "The sky is blue."])
        mw = PromptRepairMiddleware(inner, max_retries=2)
        result = await mw.generate("What colour is the sky?")
        assert result == "The sky is blue."

    @pytest.mark.asyncio
    async def test_end_to_end_json_repair(self) -> None:
        """Full cycle: broken JSON → repair → valid JSON."""
        from synapsekit import PromptRepairMiddleware

        inner = _FakeLLM(['{"name": "Bob"', '{"name": "Bob"}'])
        mw = PromptRepairMiddleware(inner, expect_json=True, max_retries=2)
        result = await mw.generate("Return user as JSON.")
        parsed = json.loads(result)
        assert parsed["name"] == "Bob"
