"""VerifiableAgent — wraps any existing agent without changing its code.

A transparent proxy: attribute access is forwarded to the wrapped agent,
and callables are instrumented on the fly to emit hash-chained
:class:`~synapsekit.audit.types.AuditRecord`\\ s around every call made
*through the proxy*. Method names are mapped onto the stable
:class:`~synapsekit.audit.types.EventKind` taxonomy — for kinds that
come in call/response pairs (``LLM_CALL``/``LLM_RESPONSE``,
``TOOL_CALL``/``TOOL_RESULT``), two linked records are emitted: one
immediately before invocation, one after completion (or ``ERROR`` on an
exception), with the response/result as a child of the call. Kinds with
no natural pair (``RETRIEVAL``, ``MEMORY_READ``/``WRITE``, ``DECISION``,
``STATE_CHANGE``) get a single record after completion, switching to
``ERROR`` on an exception.

Calls made through the proxy while another proxied call is in flight
automatically get the right ``parent_id`` via a :mod:`contextvars`
stack — e.g. calling ``verifiable.retrieve(...)`` from inside a handler
that's running because of ``verifiable.run(...)`` records the retrieval
as a child of that run's call event, with no manual wiring.

Because this is an external proxy, calls the *agent's own methods* make
directly on ``self`` are, by default, invisible to it — there is no
source to change, but also nothing to intercept there. Pass
``instrument_attrs=["llm", "retriever", ...]`` to close that gap for
agents composed of named sub-components: each listed attribute's public
methods are monkey-patched *in place* on the live object (not just on
the outer proxy), so the agent's own internal ``self.llm.generate(...)``
calls are captured too, without touching the agent's source. This still
can't help fully inlined logic with no delegated sub-object — for that,
construct with already-:func:`audited` tools/LLM calls instead.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .compliance import (
    COMPLIANCE_MODES,
    PAYLOAD_KEY,
    ComplianceMode,
    RemoteBiometricIdentificationProfile,
    RetentionPolicy,
    validate_purposes,
)
from .redact import PIIRedactor
from .trace import AuditTracer
from .types import AuditRecord, EventKind

_current_event_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "synapsekit_audit_current_event_id", default=None
)

#: Kinds with a natural call/response pair — two records are emitted.
_PAIRED_KIND_KEYWORDS: list[tuple[EventKind, EventKind, tuple[str, ...]]] = [
    (
        EventKind.LLM_CALL,
        EventKind.LLM_RESPONSE,
        ("generate", "chat", "complete", "predict", "llm"),
    ),
    (
        EventKind.TOOL_CALL,
        EventKind.TOOL_RESULT,
        ("run", "step", "tool", "call", "execute", "act", "invoke"),
    ),
]

#: Kinds with no natural pair — a single record is emitted after completion.
_SINGLE_KIND_KEYWORDS: list[tuple[EventKind, tuple[str, ...]]] = [
    (EventKind.RETRIEVAL, ("retrieve", "search", "query", "lookup")),
    (EventKind.MEMORY_WRITE, ("remember", "save_memory", "write_memory", "store_memory")),
    (EventKind.MEMORY_READ, ("recall", "load_memory", "read_memory", "get_memory")),
    (EventKind.DECISION, ("decide", "choose", "route", "plan")),
    (EventKind.STATE_CHANGE, ("transition", "set_state", "advance")),
    (EventKind.USER_INPUT, ("user_input", "on_message", "receive_input")),
]


def infer_kind_pair(method_name: str) -> tuple[EventKind, EventKind | None]:
    """Map a method name onto ``(call_kind, result_kind)``.

    ``result_kind`` is ``None`` for kinds with no natural call/response
    pair — the caller should emit a single record for those, not two.
    Falls back to ``(EventKind.SYSTEM_EVENT, None)`` for anything
    unrecognized rather than allowing an arbitrary string, per the
    taxonomy being a closed, public specification.
    """
    # Single-kind keywords are checked first: they tend to be more
    # specific ("recall", "decide") than the paired-kind keywords, which
    # include short, generic substrings ("call", "step", "run") prone to
    # false positives inside longer method names (e.g. "call" inside
    # "recall_memory", "step" inside "decide_next_step").
    name = method_name.lower()
    for kind, keywords in _SINGLE_KIND_KEYWORDS:
        if any(kw in name for kw in keywords):
            return kind, None
    for call_kind, result_kind, keywords in _PAIRED_KIND_KEYWORDS:
        if any(kw in name for kw in keywords):
            return call_kind, result_kind
    return EventKind.SYSTEM_EVENT, None


def _safe_value(value: Any, *, _depth: int = 0) -> Any:
    """Best-effort coercion to something canonical-JSON-serializable.

    Audit payloads must hash deterministically, so anything that isn't a
    primitive/dict/list gets rendered to its ``repr()``-free string form
    rather than being hashed by object identity (which would make the
    same logical call hash differently every run).
    """
    if _depth > 6:
        return "<max-depth-exceeded>"
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(k): _safe_value(v, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_safe_value(v, _depth=_depth + 1) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _safe_value(value.to_dict(), _depth=_depth + 1)
        except Exception:
            pass
    return str(value)


def _aggregate_stream(chunks: list[Any]) -> Any:
    """Deterministically summarize a completed async-gen stream.

    All-``str`` streams (the common token-streaming case) are joined into
    the full text; anything else falls back to the list of safe values.
    The result is a stable function of the chunks alone — no object
    identity or memory address ever leaks in, so the audit hash stays
    reproducible run to run.
    """
    if chunks and all(isinstance(c, str) for c in chunks):
        return "".join(chunks)
    return _safe_value(chunks)


class VerifiableAgent:
    """Wraps ``agent`` so every method call is captured in a hash-chained trace.

    Usage::

        tracer = AuditTracer()
        verifiable = VerifiableAgent(my_agent, tracer=tracer)
        result = await verifiable.run("do the thing")  # traced automatically
        records = tracer.drain()
    """

    def __init__(
        self,
        agent: Any,
        *,
        tracer: AuditTracer | None = None,
        redactor: PIIRedactor | None = None,
        actor: str | None = None,
        kind_overrides: dict[str, EventKind] | None = None,
        instrument_attrs: list[str] | None = None,
        compliance: ComplianceMode | None = None,
        retention: RetentionPolicy | None = None,
        purposes: Mapping[EventKind | str, Sequence[str]] | None = None,
        profile: RemoteBiometricIdentificationProfile | None = None,
    ) -> None:
        """Wrap ``agent``.

        ``compliance="eu_ai_act"`` turns on the EU AI Act record-keeping
        mode (see :mod:`synapsekit.audit.compliance`): ``retention`` is then
        required — ``RetentionPolicy(role="provider")`` (Article 19(1)) or
        ``RetentionPolicy(role="deployer")`` (Article 26(6)), six months
        unless a period is given — ``purposes`` optionally narrows the
        Article 12(2) purposes per event kind, and ``profile`` adds the
        Article 12(3) fields for Annex III point 1(a) systems.
        """
        if compliance is not None and compliance not in COMPLIANCE_MODES:
            raise ValueError(
                f"compliance must be one of {list(COMPLIANCE_MODES)}, got {compliance!r}"
            )
        if compliance is None:
            given = [
                name
                for name, value in (
                    ("retention", retention),
                    ("purposes", purposes),
                    ("profile", profile),
                )
                if value is not None
            ]
            if given:
                raise ValueError(
                    f"{', '.join(given)} only apply with compliance='eu_ai_act'; "
                    "pass compliance='eu_ai_act' or drop them"
                )
            purpose_map: dict[str, tuple[str, ...]] = {}
        else:
            if retention is None:
                raise ValueError(
                    "compliance='eu_ai_act' needs a retention policy, and its role decides the "
                    "article: pass retention=RetentionPolicy(role='provider') (Article 19(1)) or "
                    "retention=RetentionPolicy(role='deployer') (Article 26(6)); the period "
                    "defaults to six months"
                )
            if not isinstance(retention, RetentionPolicy):
                raise TypeError(
                    f"retention must be a RetentionPolicy, got {type(retention).__name__}"
                )
            if profile is not None and not isinstance(
                profile, RemoteBiometricIdentificationProfile
            ):
                raise TypeError(
                    "profile must be a RemoteBiometricIdentificationProfile, "
                    f"got {type(profile).__name__}"
                )
            purpose_map = validate_purposes(purposes)
            # Every record is tagged by its kind, so an override outside the
            # taxonomy would only fail after the wrapped call had already run.
            for method_name, override in (kind_overrides or {}).items():
                value = override.value if isinstance(override, EventKind) else override
                if type(value) is not str or value not in purpose_map:
                    raise ValueError(
                        f"kind_overrides[{method_name!r}] must be an EventKind, got {override!r}"
                    )
        object.__setattr__(self, "_agent", agent)
        object.__setattr__(self, "tracer", tracer if tracer is not None else AuditTracer())
        object.__setattr__(self, "_redactor", redactor)
        object.__setattr__(self, "_actor", actor or f"agent:{type(agent).__name__}")
        object.__setattr__(self, "_kind_overrides", kind_overrides or {})
        object.__setattr__(self, "_compliance", compliance)
        object.__setattr__(self, "_compliance_retention", retention)
        object.__setattr__(self, "_compliance_profile", profile)
        object.__setattr__(self, "_compliance_purposes", purpose_map)
        object.__setattr__(self, "_compliance_ended", False)
        if compliance is not None:
            self._record_lifecycle("agent_lifetime_started")
        for attr_name in instrument_attrs or []:
            self._instrument_attr_in_place(attr_name)

    def _tag(self, kind: EventKind | str, payload: dict[str, Any]) -> dict[str, Any]:
        """Add the Article 12(2) purposes to ``payload`` (inside the hashed payload)."""
        if self._compliance is None:
            return payload
        kind_value = kind.value if isinstance(kind, EventKind) else str(kind)
        tag = {"purposes": list(self._compliance_purposes[kind_value])}
        return {**payload, PAYLOAD_KEY: {**payload.get(PAYLOAD_KEY, {}), **tag}}

    def _record_compliance_event(
        self, kind: EventKind, payload: dict[str, Any], *, redact: bool = True
    ) -> AuditRecord:
        # Lifecycle events carry only this wrapper's own configuration (no
        # call data), so they skip the wrapper's redactor: a custom detector
        # must not be able to rewrite the recorded legal basis.
        tracer: AuditTracer = self.tracer
        return tracer.record(
            kind,
            self._finalize(payload, kind) if redact else self._tag(kind, payload),
            actor=self._actor,
            parent_id=_current_event_id.get(),
        )

    def _record_lifecycle(self, event: str) -> AuditRecord | None:
        if event == "agent_lifetime_ended":
            if self._compliance_ended:
                return None
            object.__setattr__(self, "_compliance_ended", True)
        retention = self._compliance_retention
        assert retention is not None  # guaranteed in compliance mode
        profile = self._compliance_profile
        body: dict[str, Any] = {"event": event, "mode": self._compliance}
        if event == "agent_lifetime_started":
            body.update(
                {
                    "regulation": "Regulation (EU) 2024/1689",
                    "retention": retention.to_dict(),
                    "purpose_map": {k: list(v) for k, v in self._compliance_purposes.items()},
                    "profile": profile.to_dict() if profile is not None else None,
                }
            )
        return self._record_compliance_event(
            EventKind.SYSTEM_EVENT, {PAYLOAD_KEY: body}, redact=False
        )

    def _kinds_for(self, name: str) -> tuple[EventKind, EventKind | None]:
        override = self._kind_overrides.get(name)
        if override is not None:
            return override, None
        return infer_kind_pair(name)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._agent, name)
        if not callable(attr):
            return attr
        call_kind, result_kind = self._kinds_for(name)
        return self._instrument(name, attr, call_kind, result_kind)

    def _instrument_attr_in_place(self, attr_name: str) -> None:
        """Monkey-patch a named sub-attribute's methods on the *live* agent object.

        Unlike the outer proxy (which only sees calls made through it),
        this rewrites ``self._agent.<attr_name>``'s own methods so the
        agent's *internal* code — e.g. ``self.llm.generate(...)`` called
        from within its own ``run()`` — routes through the tracer too.
        Best-effort: attributes that don't support instance-level
        attribute assignment (slots, builtins, immutable types) are left
        alone rather than raising.
        """
        target = getattr(self._agent, attr_name, None)
        if target is None:
            return
        for method_name in dir(target):
            if method_name.startswith("_"):
                continue
            method = getattr(target, method_name, None)
            if not callable(method):
                continue
            call_kind, result_kind = self._kinds_for(method_name)
            instrumented = self._instrument(method_name, method, call_kind, result_kind)
            try:
                object.__setattr__(target, method_name, instrumented)
            except (AttributeError, TypeError):
                continue

    def _base_payload(self, name: str, args: tuple, kwargs: dict) -> dict[str, Any]:
        return {
            "method": name,
            "args": _safe_value(list(args)),
            "kwargs": _safe_value(kwargs),
        }

    def _finalize(self, payload: dict[str, Any], kind: EventKind | str) -> dict[str, Any]:
        # Redaction must cover the full payload — including output/error,
        # not just the pre-call args/kwargs — since those can echo back
        # PII from the call's inputs (see redact.py's module docstring).
        if self._redactor is not None:
            payload = self._redactor.redact_payload(payload)
        # Compliance tags go on after redaction so a detector can never
        # rewrite an article reference such as "12(2)(a)".
        return self._tag(kind, payload)

    def _record_call(
        self, call_kind: EventKind, payload: dict[str, Any], parent_id: str | None, event_id: str
    ) -> None:
        self.tracer.record(
            call_kind,
            self._finalize(payload, call_kind),
            actor=self._actor,
            parent_id=parent_id,
            event_id=event_id,
        )

    def _record_outcome(
        self,
        *,
        paired: bool,
        result_kind: EventKind | None,
        call_kind: EventKind,
        payload: dict[str, Any],
        parent_id: str | None,
        call_event_id: str,
        is_error: bool,
    ) -> None:
        if is_error:
            kind = EventKind.ERROR
        elif paired:
            assert result_kind is not None  # paired implies a result_kind was supplied
            kind = result_kind
        else:
            kind = call_kind
        if paired:
            # The response/result is a child of its own call event.
            self.tracer.record(
                kind, self._finalize(payload, kind), actor=self._actor, parent_id=call_event_id
            )
        else:
            # No pairing — this IS the call event (reuses its event_id).
            self.tracer.record(
                kind,
                self._finalize(payload, kind),
                actor=self._actor,
                parent_id=parent_id,
                event_id=call_event_id,
            )

    def _instrument(
        self, name: str, fn: Callable[..., Any], call_kind: EventKind, result_kind: EventKind | None
    ) -> Callable[..., Any]:
        paired = result_kind is not None

        if inspect.isasyncgenfunction(fn):

            @functools.wraps(fn)
            async def asyncgen_wrapped(*args: Any, **kwargs: Any) -> Any:
                # Streaming methods (stream/stream_steps/...) must not be
                # recorded via the sync wrapper: that would record
                # status:ok with output "<async_generator object at 0x...>"
                # before a single token was produced — the memory address
                # makes the hash-chain non-deterministic, and any error
                # raised mid-stream would never be recorded at all.
                parent_id = _current_event_id.get()
                call_event_id = uuid.uuid4().hex
                base_payload = self._base_payload(name, args, kwargs)
                if paired:
                    self._record_call(call_kind, dict(base_payload), parent_id, call_event_id)
                chunks: list[Any] = []
                inner = fn(*args, **kwargs)
                try:
                    while True:
                        # Scope the contextvar to each step so nested
                        # audited calls made *while producing* a chunk get
                        # this stream as their parent. A ContextVar token
                        # can't be reset across the suspend/resume that a
                        # bare `async for … yield` implies (the resume may
                        # land in a different Context), so we set/reset
                        # around each __anext__ only.
                        token = _current_event_id.set(call_event_id)
                        try:
                            item = await inner.__anext__()
                        except StopAsyncIteration:
                            break
                        finally:
                            _current_event_id.reset(token)
                        chunks.append(item)
                        yield item
                except Exception as exc:
                    payload = {
                        **base_payload,
                        # A deterministic summary of what streamed before
                        # the failure — never the generator's repr/address.
                        "output": _safe_value(chunks),
                        "chunk_count": len(chunks),
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "status": "error",
                    }
                    self._record_outcome(
                        paired=paired,
                        result_kind=result_kind,
                        call_kind=call_kind,
                        payload=payload,
                        parent_id=parent_id,
                        call_event_id=call_event_id,
                        is_error=True,
                    )
                    raise
                else:
                    payload = {
                        **base_payload,
                        # Deterministic aggregate of the completed stream —
                        # the concatenation of yielded chunks when they are
                        # all strings, otherwise the list of safe values.
                        "output": _aggregate_stream(chunks),
                        "chunk_count": len(chunks),
                        "status": "ok",
                    }
                    self._record_outcome(
                        paired=paired,
                        result_kind=result_kind,
                        call_kind=call_kind,
                        payload=payload,
                        parent_id=parent_id,
                        call_event_id=call_event_id,
                        is_error=False,
                    )

            return asyncgen_wrapped

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
                parent_id = _current_event_id.get()
                call_event_id = uuid.uuid4().hex
                base_payload = self._base_payload(name, args, kwargs)
                if paired:
                    self._record_call(call_kind, dict(base_payload), parent_id, call_event_id)
                token = _current_event_id.set(call_event_id)
                try:
                    result = await fn(*args, **kwargs)
                    payload = {**base_payload, "output": _safe_value(result), "status": "ok"}
                    self._record_outcome(
                        paired=paired,
                        result_kind=result_kind,
                        call_kind=call_kind,
                        payload=payload,
                        parent_id=parent_id,
                        call_event_id=call_event_id,
                        is_error=False,
                    )
                    return result
                except Exception as exc:
                    payload = {
                        **base_payload,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "status": "error",
                    }
                    self._record_outcome(
                        paired=paired,
                        result_kind=result_kind,
                        call_kind=call_kind,
                        payload=payload,
                        parent_id=parent_id,
                        call_event_id=call_event_id,
                        is_error=True,
                    )
                    raise
                finally:
                    _current_event_id.reset(token)

            return async_wrapped

        @functools.wraps(fn)
        def sync_wrapped(*args: Any, **kwargs: Any) -> Any:
            parent_id = _current_event_id.get()
            call_event_id = uuid.uuid4().hex
            base_payload = self._base_payload(name, args, kwargs)
            if paired:
                self._record_call(call_kind, dict(base_payload), parent_id, call_event_id)
            token = _current_event_id.set(call_event_id)
            try:
                result = fn(*args, **kwargs)
                payload = {**base_payload, "output": _safe_value(result), "status": "ok"}
                self._record_outcome(
                    paired=paired,
                    result_kind=result_kind,
                    call_kind=call_kind,
                    payload=payload,
                    parent_id=parent_id,
                    call_event_id=call_event_id,
                    is_error=False,
                )
                return result
            except Exception as exc:
                payload = {
                    **base_payload,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "status": "error",
                }
                self._record_outcome(
                    paired=paired,
                    result_kind=result_kind,
                    call_kind=call_kind,
                    payload=payload,
                    parent_id=parent_id,
                    call_event_id=call_event_id,
                    is_error=True,
                )
                raise
            finally:
                _current_event_id.reset(token)

        return sync_wrapped

    @property
    def wrapped(self) -> Any:
        """Escape hatch to the underlying, un-instrumented agent."""
        return self._agent


def audited(
    kind: EventKind | str = EventKind.SYSTEM_EVENT,
    *,
    tracer: AuditTracer,
    redactor: PIIRedactor | None = None,
    actor: str = "system",
    name: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator form of :class:`VerifiableAgent` for standalone functions/tools.

    Emits a single record per call (kind ``kind`` on success, always
    ``EventKind.ERROR`` on an exception regardless of ``kind``). Useful
    when you want to audit one specific tool or LLM-call helper rather
    than wrapping an entire agent object::

        tracer = AuditTracer()

        @audited(EventKind.TOOL_CALL, tracer=tracer)
        async def search_web(query: str) -> str: ...
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        call_name = name or fn.__name__

        def build_payload(args: tuple, kwargs: dict) -> dict[str, Any]:
            return {
                "method": call_name,
                "args": _safe_value(list(args)),
                "kwargs": _safe_value(kwargs),
            }

        def finalize(payload: dict[str, Any]) -> dict[str, Any]:
            # Redact the full payload (including output/error), not just
            # the pre-call args/kwargs — see redact.py's module docstring.
            if redactor is not None:
                payload = redactor.redact_payload(payload)
            return payload

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
                parent_id = _current_event_id.get()
                event_id = uuid.uuid4().hex
                payload = build_payload(args, kwargs)
                # Default to ERROR so that a BaseException that skips both
                # the success and the ``except Exception`` branch — most
                # importantly asyncio.CancelledError, which is a
                # BaseException, not an Exception — still records a record
                # in ``finally`` instead of raising UnboundLocalError.
                emitted_kind: EventKind | str = EventKind.ERROR
                token = _current_event_id.set(event_id)
                try:
                    result = await fn(*args, **kwargs)
                    payload["output"] = _safe_value(result)
                    payload["status"] = "ok"
                    emitted_kind = kind
                    return result
                except Exception as exc:
                    payload["error"] = str(exc)
                    payload["error_type"] = type(exc).__name__
                    payload["status"] = "error"
                    emitted_kind = EventKind.ERROR
                    raise
                except BaseException as exc:
                    # Cancellation (and other BaseExceptions like
                    # KeyboardInterrupt) must still leave an audit trail,
                    # then propagate unchanged — swallowing a
                    # CancelledError would break cooperative cancellation.
                    payload["error"] = str(exc) or type(exc).__name__
                    payload["error_type"] = type(exc).__name__
                    payload["status"] = "cancelled"
                    emitted_kind = EventKind.ERROR
                    raise
                finally:
                    _current_event_id.reset(token)
                    tracer.record(
                        emitted_kind,
                        finalize(payload),
                        actor=actor,
                        parent_id=parent_id,
                        event_id=event_id,
                    )

            return async_wrapped

        @functools.wraps(fn)
        def sync_wrapped(*args: Any, **kwargs: Any) -> Any:
            parent_id = _current_event_id.get()
            event_id = uuid.uuid4().hex
            payload = build_payload(args, kwargs)
            # See async_wrapped: default to ERROR so a BaseException
            # (e.g. KeyboardInterrupt) that skips both branches still
            # records instead of raising UnboundLocalError.
            emitted_kind: EventKind | str = EventKind.ERROR
            token = _current_event_id.set(event_id)
            try:
                result = fn(*args, **kwargs)
                payload["output"] = _safe_value(result)
                payload["status"] = "ok"
                emitted_kind = kind
                return result
            except Exception as exc:
                payload["error"] = str(exc)
                payload["error_type"] = type(exc).__name__
                payload["status"] = "error"
                emitted_kind = EventKind.ERROR
                raise
            except BaseException as exc:
                payload["error"] = str(exc) or type(exc).__name__
                payload["error_type"] = type(exc).__name__
                payload["status"] = "cancelled"
                emitted_kind = EventKind.ERROR
                raise
            finally:
                _current_event_id.reset(token)
                tracer.record(
                    emitted_kind,
                    finalize(payload),
                    actor=actor,
                    parent_id=parent_id,
                    event_id=event_id,
                )

        return sync_wrapped

    return decorator
