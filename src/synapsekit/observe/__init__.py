from __future__ import annotations

import functools
import inspect
import time
from collections.abc import AsyncGenerator
from typing import Any

from ..llm.base import BaseLLM
from ..observability.tracer import COST_TABLE
from .runtime import (
    clear_exported_spans,
    configure,
    current_span,
    end_span,
    get_config,
    get_exporter,
    is_enabled,
    record_exception,
    reset,
    start_span,
    trace,
)
from .exporters import (
    ConsoleExporter,
    HoneycombExporter,
    JaegerExporter,
    LangfuseExporter,
    OTLPExporter,
)
from .spans import SpanAttributes, SpanBuilder

_PATCHED_SENTINEL = "__synapsekit_observe_patched__"
_INIT_SUBCLASS_SENTINEL = "__synapsekit_observe_init_subclass__"
_ORIGINAL_INIT_SUBCLASS = "__synapsekit_observe_original_init_subclass__"


def _recursive_subclasses(cls: type) -> list[type]:
    found: list[type] = []
    for subcls in cls.__subclasses__():
        found.append(subcls)
        found.extend(_recursive_subclasses(subcls))
    return found


def _token_snapshot(llm: Any) -> tuple[int, int]:
    used = getattr(llm, "tokens_used", None)
    if isinstance(used, dict):
        return int(used.get("input", 0)), int(used.get("output", 0))
    return int(getattr(llm, "_input_tokens", 0)), int(getattr(llm, "_output_tokens", 0))


def _token_delta(llm: Any, before: tuple[int, int]) -> dict[str, int]:
    current = _token_snapshot(llm)
    return {
        "input": max(0, current[0] - before[0]),
        "output": max(0, current[1] - before[1]),
    }


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = COST_TABLE.get(model, {})
    return (prompt_tokens * pricing.get("input", 0.0)) + (
        completion_tokens * pricing.get("output", 0.0)
    )


def _llm_start_attributes(llm: Any, payload: Any | None = None) -> dict[str, Any]:
    config = getattr(llm, "config", None)
    attrs = {
        SpanAttributes.LLM_MODEL: getattr(config, "model", getattr(llm, "model", "unknown")),
        SpanAttributes.LLM_PROVIDER: getattr(config, "provider", "unknown"),
    }
    if get_config().trace_llm_inputs and payload is not None:
        attrs[SpanAttributes.LLM_INPUT] = payload
    return attrs


def _llm_end_attributes(
    llm: Any,
    before: tuple[int, int],
    output: Any | None = None,
) -> dict[str, Any]:
    config = getattr(llm, "config", None)
    model = getattr(config, "model", getattr(llm, "model", "unknown"))
    delta = _token_delta(llm, before)
    attrs: dict[str, Any] = {
        SpanAttributes.LLM_PROMPT_TOKENS: delta["input"],
        SpanAttributes.LLM_COMPLETION_TOKENS: delta["output"],
        SpanAttributes.LLM_TOTAL_TOKENS: delta["input"] + delta["output"],
    }
    if get_config().cost_tracking:
        attrs[SpanAttributes.LLM_COST_USD] = round(
            _estimate_cost(model, delta["input"], delta["output"]),
            6,
        )
    if get_config().trace_llm_outputs and output is not None:
        attrs[SpanAttributes.LLM_OUTPUT] = output
    return attrs


def _guess_payload(method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any | None:
    if method_name in {"stream", "generate"}:
        if args:
            return args[0]
        return kwargs.get("prompt")
    if args:
        return args[0]
    return kwargs.get("messages")


def _patch_async_generator_method(cls: type, method_name: str) -> None:
    method = getattr(cls, method_name, None)
    if method is None or getattr(method, _PATCHED_SENTINEL, False):
        return
    if not inspect.isasyncgenfunction(method):
        return

    @functools.wraps(method)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
        payload = _guess_payload(method_name, args, kwargs)
        span = start_span("llm.generate", _llm_start_attributes(self, payload))
        before = _token_snapshot(self)
        started_at = time.perf_counter()
        chunks: list[str] = []
        try:
            async for chunk in method(self, *args, **kwargs):
                chunks.append(chunk)
                yield chunk
        except Exception as exc:
            record_exception(span, exc)
            raise
        finally:
            attrs = _llm_end_attributes(self, before, "".join(chunks) if chunks else None)
            attrs[SpanAttributes.LLM_LATENCY_MS] = round(
                (time.perf_counter() - started_at) * 1000,
                3,
            )
            end_span(span, attributes=attrs)

    setattr(wrapped, _PATCHED_SENTINEL, True)
    setattr(cls, method_name, wrapped)


def _patch_async_method(cls: type, method_name: str) -> None:
    method = getattr(cls, method_name, None)
    if method is None or getattr(method, _PATCHED_SENTINEL, False):
        return
    if not inspect.iscoroutinefunction(method):
        return

    @functools.wraps(method)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        payload = _guess_payload(method_name, args, kwargs)
        span = start_span("llm.generate", _llm_start_attributes(self, payload))
        before = _token_snapshot(self)
        started_at = time.perf_counter()
        result: Any | None = None
        try:
            result = await method(self, *args, **kwargs)
            return result
        except Exception as exc:
            record_exception(span, exc)
            raise
        finally:
            extra = _llm_end_attributes(self, before)
            if get_config().trace_llm_outputs and result is not None:
                if isinstance(result, dict):
                    extra[SpanAttributes.LLM_OUTPUT] = result
                else:
                    extra[SpanAttributes.LLM_OUTPUT] = str(result)
            if isinstance(result, dict) and result.get("tool_calls") is not None:
                extra[SpanAttributes.LLM_TOOL_CALLS] = len(result.get("tool_calls") or [])
            extra[SpanAttributes.LLM_LATENCY_MS] = round(
                (time.perf_counter() - started_at) * 1000,
                3,
            )
            end_span(span, attributes=extra)

    setattr(wrapped, _PATCHED_SENTINEL, True)
    setattr(cls, method_name, wrapped)


def _instrument_llm_class(cls: type[BaseLLM]) -> None:
    _patch_async_generator_method(cls, "stream")
    _patch_async_generator_method(cls, "stream_with_messages")
    _patch_async_method(cls, "call_with_tools")


def _instrument_existing_llms() -> None:
    for llm_cls in [BaseLLM, *_recursive_subclasses(BaseLLM)]:
        _instrument_llm_class(llm_cls)


def _install_future_llm_instrumentation() -> None:
    if getattr(BaseLLM, _INIT_SUBCLASS_SENTINEL, False):
        return

    original = BaseLLM.__init_subclass__
    original_func = getattr(original, "__func__", None)
    setattr(BaseLLM, _ORIGINAL_INIT_SUBCLASS, original)

    @classmethod
    def observed_init_subclass(cls, **kwargs: Any) -> None:
        if original_func is not None:
            original_func(cls, **kwargs)
        else:
            original(**kwargs)
        _instrument_llm_class(cls)

    setattr(observed_init_subclass, _INIT_SUBCLASS_SENTINEL, True)
    BaseLLM.__init_subclass__ = observed_init_subclass


def _instrument() -> None:
    _install_future_llm_instrumentation()
    _instrument_existing_llms()


_instrument()
configure()

__all__ = [
    "ConsoleExporter",
    "HoneycombExporter",
    "JaegerExporter",
    "LangfuseExporter",
    "OTLPExporter",
    "SpanAttributes",
    "SpanBuilder",
    "clear_exported_spans",
    "configure",
    "current_span",
    "get_exporter",
    "is_enabled",
    "reset",
    "start_span",
    "end_span",
    "trace",
]
