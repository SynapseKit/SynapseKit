from __future__ import annotations

import inspect
import random
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Protocol

from .exporters import (
    ConsoleExporter,
    HoneycombExporter,
    JaegerExporter,
    LangfuseExporter,
    OTLPExporter,
)

REDACTED = "[REDACTED]"


class SpanExporter(Protocol):
    service_name: str
    kind: str
    endpoint: str | None
    spans: list["ObserveSpan"]

    def export(self, span: "ObserveSpan") -> None: ...

    def clear(self) -> None: ...

    def export_dicts(self) -> list[dict[str, Any]]: ...


@dataclass
class ObserveConfig:
    exporter: str | SpanExporter = "console"
    endpoint: str | None = None
    service_name: str = "synapsekit"
    trace_llm_inputs: bool = True
    trace_llm_outputs: bool = True
    cost_tracking: bool = True
    sample_rate: float = 1.0
    redact_keys: tuple[str, ...] = ()


@dataclass
class ObserveSpan:
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    parent: ObserveSpan | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    status: str = "ok"
    children: list[ObserveSpan] = field(default_factory=list)
    _context_token: Token[ObserveSpan | None] | None = field(default=None, repr=False, compare=False)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = sanitize_value(key, value)

    def set_status(self, status: str) -> None:
        self.status = status

    @property
    def duration_ms(self) -> float:
        end = self.end_time if self.end_time is not None else time.time()
        return (end - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "attributes": {k: sanitize_value(k, v) for k, v in self.attributes.items()},
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "children": [child.to_dict() for child in self.children],
        }


class InMemoryExporter:
    def __init__(
        self,
        *,
        service_name: str = "synapsekit",
        kind: str = "console",
        endpoint: str | None = None,
    ) -> None:
        self.service_name = service_name
        self.kind = kind
        self.endpoint = endpoint
        self.spans: list[ObserveSpan] = []

    def export(self, span: ObserveSpan) -> None:
        self.spans.append(span)

    def clear(self) -> None:
        self.spans.clear()

    def export_dicts(self) -> list[dict[str, Any]]:
        return [span.to_dict() for span in self.spans]


@dataclass
class _ObserveState:
    config: ObserveConfig = field(default_factory=ObserveConfig)
    exporter: SpanExporter = field(default_factory=InMemoryExporter)
    enabled: bool = False


_STATE = _ObserveState()
_CURRENT_SPAN: ContextVar[ObserveSpan | None] = ContextVar("synapsekit_observe_current_span", default=None)


def _normalize_sample_rate(sample_rate: float) -> float:
    return min(1.0, max(0.0, float(sample_rate)))


def _make_exporter(
    exporter: str | SpanExporter,
    *,
    service_name: str,
    endpoint: str | None,
) -> SpanExporter:
    if hasattr(exporter, "export") and hasattr(exporter, "clear"):
        return exporter  # type: ignore[return-value]

    kind = str(exporter).lower()
    if kind == "console":
        return ConsoleExporter(service_name=service_name, endpoint=endpoint)
    if kind == "otlp":
        return OTLPExporter(service_name=service_name, endpoint=endpoint)
    if kind == "jaeger":
        return JaegerExporter(service_name=service_name, endpoint=endpoint)
    if kind == "langfuse":
        return LangfuseExporter(service_name=service_name, endpoint=endpoint)
    if kind == "honeycomb":
        return HoneycombExporter(service_name=service_name, endpoint=endpoint)
    raise ValueError(
        "Unsupported exporter. Use one of: console, otlp, jaeger, langfuse, honeycomb."
    )


def configure(
    *,
    exporter: str | SpanExporter = "console",
    endpoint: str | None = None,
    service_name: str = "synapsekit",
    trace_llm_inputs: bool = True,
    trace_llm_outputs: bool = True,
    cost_tracking: bool = True,
    sample_rate: float = 1.0,
    redact_keys: list[str] | tuple[str, ...] | None = None,
) -> SpanExporter:
    config = ObserveConfig(
        exporter=exporter,
        endpoint=endpoint,
        service_name=service_name,
        trace_llm_inputs=trace_llm_inputs,
        trace_llm_outputs=trace_llm_outputs,
        cost_tracking=cost_tracking,
        sample_rate=_normalize_sample_rate(sample_rate),
        redact_keys=tuple(redact_keys or ()),
    )
    _STATE.config = config
    _STATE.exporter = _make_exporter(
        exporter,
        service_name=service_name,
        endpoint=endpoint,
    )
    _STATE.enabled = True
    return _STATE.exporter


def reset() -> None:
    _STATE.exporter.clear()
    _STATE.enabled = False
    _STATE.config = ObserveConfig()
    _CURRENT_SPAN.set(None)


def is_enabled() -> bool:
    return _STATE.enabled


def get_config() -> ObserveConfig:
    return _STATE.config


def get_exporter() -> SpanExporter:
    return _STATE.exporter


def clear_exported_spans() -> None:
    _STATE.exporter.clear()


def current_span() -> ObserveSpan | None:
    return _CURRENT_SPAN.get()


def sanitize_value(key: str, value: Any) -> Any:
    redact = {item.lower() for item in _STATE.config.redact_keys}

    def _sanitize(inner_key: str, inner_value: Any) -> Any:
        if inner_key.lower() in redact:
            return REDACTED
        if isinstance(inner_value, dict):
            return {str(k): _sanitize(str(k), v) for k, v in inner_value.items()}
        if isinstance(inner_value, list):
            return [_sanitize("", item) for item in inner_value]
        if isinstance(inner_value, tuple):
            return [_sanitize("", item) for item in inner_value]
        if isinstance(inner_value, (str, int, float, bool)) or inner_value is None:
            return inner_value
        return str(inner_value)

    return _sanitize(key, value)


def start_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    *,
    parent: ObserveSpan | None = None,
    set_current: bool = True,
) -> ObserveSpan | None:
    if not _STATE.enabled:
        return None

    resolved_parent = parent if parent is not None else current_span()
    if resolved_parent is None and random.random() > _STATE.config.sample_rate:
        return None

    span = ObserveSpan(
        name=name,
        attributes={
            key: sanitize_value(key, value) for key, value in (attributes or {}).items() if value is not None
        },
        parent=resolved_parent,
    )
    if resolved_parent is not None:
        resolved_parent.children.append(span)
    if set_current:
        span._context_token = _CURRENT_SPAN.set(span)
    return span


def end_span(
    span: ObserveSpan | None,
    *,
    attributes: dict[str, Any] | None = None,
    error: Exception | None = None,
) -> None:
    if span is None:
        return

    for key, value in (attributes or {}).items():
        if value is not None:
            span.set_attribute(key, value)

    if error is not None:
        span.set_status("error")
        span.set_attribute("error", str(error))

    span.end_time = time.time()
    span.set_attribute("observe.duration_ms", round(span.duration_ms, 3))
    if span._context_token is not None:
        _CURRENT_SPAN.reset(span._context_token)
        span._context_token = None
    if span.parent is None:
        _STATE.exporter.export(span)


def record_exception(span: ObserveSpan | None, exc: Exception) -> None:
    if span is None:
        return
    span.set_status("error")
    span.set_attribute("error", str(exc))


def trace(name: str):
    def decorator(func):
        if inspect.iscoroutinefunction(func):
            async def async_wrapper(*args: Any, **kwargs: Any):
                span = start_span(name)
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as exc:
                    record_exception(span, exc)
                    raise
                finally:
                    end_span(span)

            async_wrapper.__name__ = getattr(func, "__name__", "wrapped")
            async_wrapper.__doc__ = getattr(func, "__doc__")
            return async_wrapper

        def sync_wrapper(*args: Any, **kwargs: Any):
            span = start_span(name)
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                record_exception(span, exc)
                raise
            finally:
                end_span(span)

        sync_wrapper.__name__ = getattr(func, "__name__", "wrapped")
        sync_wrapper.__doc__ = getattr(func, "__doc__")
        return sync_wrapper

    return decorator
