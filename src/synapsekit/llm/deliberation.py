"""Parallel deliberation over independently generated, bounded responses.

Adapters must actually enforce the supplied provider-side output limit, including
reasoning tokens, and report per-request usage; BaseLLM alone does not guarantee
that contract. No judge model is invoked.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from typing import Literal, Protocol, overload


def _finite_number(value: object) -> bool:
    """Validate finite JSON-style numeric values without overflowing on huge ints."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None


class BoundedAdapter(Protocol):
    async def complete(self, prompt: str, *, max_output_tokens: int) -> Completion:
        """Enforce the output cap at the provider, including hidden reasoning."""
        ...


class OpenAIChatAdapter:
    """Adapter for AsyncOpenAI chat completions with SDK retries disabled.

    Only use models where max_completion_tokens bounds all output, including
    reasoning. Prices are optional caller-supplied per-token rates and exclude
    charges not represented by those rates (e.g. cached-input discounts).
    """

    def __init__(
        self,
        client,
        *,
        model: str,
        input_price_per_token: float | None = None,
        output_price_per_token: float | None = None,
    ):
        if getattr(client, "max_retries", None) != 0:
            raise ValueError("client must have max_retries=0 for bounded accounting")
        for rate in (input_price_per_token, output_price_per_token):
            if rate is not None and (not _finite_number(rate) or rate < 0):
                raise ValueError("prices must be finite and nonnegative")
        self.client = client
        self.model = model
        self.input_price = input_price_per_token
        self.output_price = output_price_per_token

    async def complete(self, prompt: str, *, max_output_tokens: int) -> Completion:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_output_tokens,
            stream=False,
        )
        usage = response.usage
        inputs = getattr(usage, "prompt_tokens", None)
        outputs = getattr(usage, "completion_tokens", None)
        cost = None
        if (
            type(inputs) is int
            and inputs >= 0
            and type(outputs) is int
            and outputs >= 0
            and self.input_price is not None
            and self.output_price is not None
        ):
            cost = inputs * self.input_price + outputs * self.output_price
        text = response.choices[0].message.content if response.choices else ""
        return Completion(text or "", inputs, outputs, cost)

    async def aclose(self) -> None:
        """Close the injected client; callers own its lifetime until then."""
        await self.client.close()


@dataclass(frozen=True)
class Trace:
    index: int
    status: str
    answer: str | None = None
    confidence: float | None = None
    completion: Completion | None = None
    error: str | None = None


@dataclass(frozen=True)
class DeliberationResult:
    answer: str | None
    confidence: float | None
    traces: tuple[Trace, ...]
    early_exit: bool
    total_input_tokens: int | None
    total_output_tokens: int | None
    total_cost: float | None


def _score(traces: list[Trace]) -> tuple[str | None, float | None, int]:
    groups: dict[str, list[float]] = {}
    for trace in sorted(traces, key=lambda t: t.index):
        if trace.status == "ok" and trace.answer is not None and trace.confidence is not None:
            groups.setdefault(trace.answer, []).append(trace.confidence)
    if not groups:
        return None, None, 0
    answer = max(groups, key=lambda key: sum(groups[key]))
    votes = groups[answer]
    return answer, sum(votes) / len(votes), len(votes)


def _safe_margin(traces: list[Trace], winner: str, remaining: int) -> bool:
    """Require a strict lead over every possible remaining-vote challenger."""
    groups: dict[str, list[float]] = {}
    for trace in sorted(traces, key=lambda t: t.index):
        if trace.status == "ok" and trace.answer is not None and trace.confidence is not None:
            groups.setdefault(trace.answer, []).append(trace.confidence)
    scores = {answer: sum(votes) for answer, votes in groups.items()}
    challenger = max((score for answer, score in scores.items() if answer != winner), default=0.0)
    return scores[winner] > challenger + remaining


@overload
def _total(traces: list[Trace], field: Literal["input_tokens", "output_tokens"]) -> int | None: ...


@overload
def _total(traces: list[Trace], field: Literal["cost"]) -> float | None: ...


def _total(traces: list[Trace], field: str) -> int | float | None:
    values: list[int | float | None] = [
        getattr(t.completion, field) if t.completion else None for t in traces
    ]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


async def _run(adapter: BoundedAdapter, prompt: str, cap: int, index: int) -> Trace:
    try:
        response = await adapter.complete(prompt, max_output_tokens=cap)
    except asyncio.CancelledError:
        raise
    except Exception:
        return Trace(index, "error", error="adapter request failed")
    if not isinstance(response, Completion):
        return Trace(index, "invalid", error="adapter must return Completion")
    # Untrustworthy usage is unknown, never zero. Reject malformed reports.
    for value in (response.input_tokens, response.output_tokens):
        if value is not None and (type(value) is not int or value < 0):
            return Trace(index, "invalid", error="invalid token usage")
    if response.cost is not None and (not _finite_number(response.cost) or response.cost < 0):
        return Trace(index, "invalid", error="invalid cost")
    try:
        data = json.loads(response.text)
        answer, confidence = data["answer"], data["confidence"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must be nonempty text")
        if not _finite_number(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be finite and between 0 and 1")
        if response.output_tokens is not None and response.output_tokens > cap:
            raise ValueError("adapter exceeded output cap")
        return Trace(index, "ok", answer.strip(), float(confidence), response)
    except (ValueError, TypeError, KeyError) as exc:
        return Trace(index, "invalid", completion=response, error=str(exc))


async def deliberate(
    adapter: BoundedAdapter,
    prompt: str,
    *,
    n: int,
    max_output_tokens: int,
    total_output_budget: int,
    early_exit_confidence: float | None = None,
) -> DeliberationResult:
    """Run N independent responses; reserve all output tokens before launch."""
    for name, value in (
        ("n", n),
        ("max_output_tokens", max_output_tokens),
        ("total_output_budget", total_output_budget),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if n * max_output_tokens > total_output_budget:
        raise ValueError("total output budget cannot cover all launched requests")
    if early_exit_confidence is not None and (
        not _finite_number(early_exit_confidence) or not 0 <= early_exit_confidence <= 1
    ):
        raise ValueError("early_exit_confidence must be finite and between 0 and 1")
    request = (
        "Independently answer the question concisely. Return only JSON with "
        '"answer" (string) and "confidence" (number from 0 to 1).\nQuestion:\n' + prompt
    )
    tasks = [asyncio.create_task(_run(adapter, request, max_output_tokens, i)) for i in range(n)]
    traces: list[Trace] = []
    pending = set(tasks)
    decision: tuple[str | None, float | None, int] | None = None
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            traces.extend(
                Trace(tasks.index(task), "cancelled") if task.cancelled() else task.result()
                for task in done
            )
            candidate = _score(traces)
            if (
                pending
                and early_exit_confidence is not None
                and candidate[2] >= 2
                and candidate[1] is not None
                and candidate[1] >= early_exit_confidence
                and candidate[0] is not None
                and _safe_margin(traces, candidate[0], len(pending))
            ):
                decision = candidate
                break
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    if decision is not None:
        for task in pending:
            index = tasks.index(task)
            if task.cancelled():
                traces.append(Trace(index, "cancelled"))
            elif task.exception() is not None:
                traces.append(Trace(index, "error", error="adapter request failed"))
            else:
                traces.append(task.result())
    answer, confidence, _ = _score(traces)
    traces.sort(key=lambda t: t.index)
    return DeliberationResult(
        answer,
        confidence,
        tuple(traces),
        decision is not None,
        _total(traces, "input_tokens"),
        _total(traces, "output_tokens"),
        _total(traces, "cost"),
    )
