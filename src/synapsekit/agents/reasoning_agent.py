from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Literal

from ..llm.base import BaseLLM
from ..llm.reasoning import ReasoningLLM, ReasoningUsage
from ..observability.budget_guard import BudgetExceededError
from ..serve.server import ServeMixin
from .base import BaseTool
from .executor import AgentConfig, AgentExecutor


@dataclass
class ReasoningDecision:
    complexity: Literal["simple", "complex"]
    reason: str
    score: float | None = None


@dataclass
class ReasoningAgentConfig:
    fast_llm: BaseLLM
    reasoning_llm: ReasoningLLM
    tools: list[BaseTool]
    agent_type: Literal["react", "function_calling"] = "react"
    max_iterations: int = 10
    system_prompt: str = "You are a helpful AI assistant."
    classifier_llm: BaseLLM | None = None
    classifier_prompt: str | None = None
    complexity_threshold: int = 2
    thinking_budget_tokens: int = 15000
    timeout_seconds: float | None = 30.0
    fallback_on_error: bool = True


_DEFAULT_CLASSIFIER_PROMPT = """\
Classify the user query as SIMPLE or COMPLEX.

SIMPLE: can be answered quickly, single-step, no deep reasoning.
COMPLEX: multi-step reasoning, math/proofs, architecture, debugging, or heavy analysis.

Return only one word: SIMPLE or COMPLEX.
Query: {query}
"""


class ComplexityClassifier:
    def __init__(
        self,
        llm: BaseLLM | None = None,
        prompt: str | None = None,
        threshold: int = 2,
    ) -> None:
        self._llm = llm
        self._prompt = prompt or _DEFAULT_CLASSIFIER_PROMPT
        self._threshold = threshold

    async def classify(self, query: str) -> ReasoningDecision:
        if self._llm is not None:
            decision = await self._classify_with_llm(query)
            if decision is not None:
                return decision
        return self._classify_with_heuristics(query)

    async def _classify_with_llm(self, query: str) -> ReasoningDecision | None:
        llm = self._llm
        if llm is None:
            return None
        prompt = self._prompt.format(query=query)
        response = await llm.generate(prompt)
        text = response.strip().lower()
        if text.startswith("simple"):
            return ReasoningDecision(complexity="simple", reason="llm: simple")
        if text.startswith("complex"):
            return ReasoningDecision(complexity="complex", reason="llm: complex")

        match = re.search(r"\b(simple|complex)\b", text)
        if match and match.group(1) == "complex":
            return ReasoningDecision(complexity="complex", reason="llm: inferred")
        if match:
            return ReasoningDecision(complexity="simple", reason="llm: inferred")
        return None

    def _classify_with_heuristics(self, query: str) -> ReasoningDecision:
        score = 0
        reasons: list[str] = []
        normalized = query.lower()
        words = len(normalized.split())

        if words >= 30 or len(normalized) >= 240:
            score += 1
            reasons.append("long query")
        if normalized.count("?") > 1:
            score += 1
            reasons.append("multiple questions")
        if re.search(r"[=\^∑∫]|\bmatrix\b|\btheorem\b|\bproof\b", normalized):
            score += 1
            reasons.append("math keywords")
        if re.search(r"\bderive\b|\boptimize\b|\bcomplexity\b|\barchitecture\b", normalized):
            score += 1
            reasons.append("reasoning keywords")

        reason = ", ".join(reasons) if reasons else "heuristic: short"
        if score >= self._threshold:
            return ReasoningDecision(
                complexity="complex",
                reason=reason,
                score=float(score),
            )
        return ReasoningDecision(
            complexity="simple",
            reason=reason,
            score=float(score),
        )


class _BudgetedReasoningLLM(BaseLLM):
    def __init__(self, llm: ReasoningLLM, budget_tokens: int) -> None:
        super().__init__(llm.config)
        self._llm = llm
        self._budget_tokens = budget_tokens
        self._thinking_used = 0
        self._last_usage: ReasoningUsage | None = None

    def reset_budget(self) -> None:
        self._thinking_used = 0
        self._last_usage = None

    def _record_usage(self) -> None:
        usage = self._llm.last_usage
        if usage is None:
            return
        self._last_usage = usage
        self._thinking_used += usage.thinking_tokens
        if self._budget_tokens is not None and self._thinking_used > self._budget_tokens:
            raise BudgetExceededError(
                f"Thinking token budget exceeded: {self._thinking_used} > {self._budget_tokens}",
                limit_type="thinking_tokens",
                limit_value=float(self._budget_tokens),
                current=float(self._thinking_used),
            )

    @property
    def last_usage(self) -> ReasoningUsage | None:
        return self._last_usage

    @property
    def thinking_tokens_used(self) -> int:
        return self._thinking_used

    async def generate(self, prompt: str, **kw: Any) -> str:
        result = await self._llm.generate(prompt, **kw)
        self._record_usage()
        return result

    async def generate_with_messages(self, messages: list[dict[str, Any]], **kw: Any) -> str:
        result = await self._llm.generate_with_messages(messages, **kw)
        self._record_usage()
        return result

    async def stream(self, prompt: str, **kw: Any):
        async for token in self._llm.stream(prompt, **kw):
            yield token
        self._record_usage()

    async def stream_with_messages(self, messages: list[dict[str, Any]], **kw: Any):
        async for token in self._llm.stream_with_messages(messages, **kw):
            yield token
        self._record_usage()

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        result = await self._llm.call_with_tools(messages, tools)
        self._record_usage()
        return result


class ReasoningAgent(ServeMixin):
    def __init__(self, config: ReasoningAgentConfig) -> None:
        self._config = config
        self._classifier = ComplexityClassifier(
            llm=config.classifier_llm,
            prompt=config.classifier_prompt,
            threshold=config.complexity_threshold,
        )

        if config.agent_type == "function_calling" and config.reasoning_llm.provider not in {
            "openai",
            "anthropic",
        }:
            raise ValueError(
                "ReasoningLLM function-calling is only supported for OpenAI or Anthropic reasoning models."
            )

        self._fast_executor = AgentExecutor(
            AgentConfig(
                llm=config.fast_llm,
                tools=config.tools,
                agent_type=config.agent_type,
                max_iterations=config.max_iterations,
                system_prompt=config.system_prompt,
            )
        )
        self._budgeted_reasoning_llm = _BudgetedReasoningLLM(
            config.reasoning_llm,
            budget_tokens=config.thinking_budget_tokens,
        )
        self._reasoning_executor = AgentExecutor(
            AgentConfig(
                llm=self._budgeted_reasoning_llm,
                tools=config.tools,
                agent_type=config.agent_type,
                max_iterations=config.max_iterations,
                system_prompt=config.system_prompt,
            )
        )

        self.last_decision: ReasoningDecision | None = None
        self.last_usage: ReasoningUsage | None = None
        self.last_fallback_reason: str | None = None

    async def run(self, query: str) -> str:
        self.last_fallback_reason = None
        self.last_usage = None
        self._budgeted_reasoning_llm.reset_budget()

        decision = await self._classifier.classify(query)
        self.last_decision = decision

        if decision.complexity == "simple":
            return await self._fast_executor.run(query)

        try:
            if self._config.timeout_seconds is not None:
                result = await asyncio.wait_for(
                    self._reasoning_executor.run(query),
                    timeout=self._config.timeout_seconds,
                )
            else:
                result = await self._reasoning_executor.run(query)
            self.last_usage = self._budgeted_reasoning_llm.last_usage
            return result
        except (asyncio.TimeoutError, BudgetExceededError) as exc:
            if not self._config.fallback_on_error:
                raise
            self.last_usage = self._budgeted_reasoning_llm.last_usage
            self.last_fallback_reason = str(exc)
            return await self._fast_executor.run(query)

    async def stream(self, query: str):
        answer = await self.run(query)
        # Use regex to split while preserving whitespace and punctuation for better "simulated" streaming
        tokens = re.findall(r"\S+|\s+", answer)
        for token in tokens:
            yield token
