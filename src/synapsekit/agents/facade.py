"""Agent facade — 10-line happy-path entry point."""

from __future__ import annotations

from ..llm._factory import make_llm
from ..memory.conversation import ConversationMemory
from .base import BaseTool
from .executor import AgentConfig, AgentExecutor


class SimpleAgent:
    """
    A thin wrapper around AgentExecutor for a simpler developer experience.
    """

    def __init__(self, executor: AgentExecutor, memory: ConversationMemory | None = None) -> None:
        self._executor = executor
        self._memory = memory

    def _build_prompt(self, prompt: str) -> str:
        if self._memory and len(self._memory) > 0:
            context = self._memory.format_context()
            return f"{context}\n\nUser: {prompt}"
        return prompt

    def _update_memory(self, prompt: str, answer: str) -> None:
        if self._memory is not None:
            self._memory.add("user", prompt)
            self._memory.add("assistant", answer)

    async def arun(self, prompt: str) -> str:
        """Async: run agent and return final answer."""
        full_prompt = self._build_prompt(prompt)
        answer = await self._executor.run(full_prompt)
        self._update_memory(prompt, answer)
        return answer

    def run(self, prompt: str) -> str:
        """Sync: run agent and return final answer."""
        full_prompt = self._build_prompt(prompt)
        answer = self._executor.run_sync(full_prompt)
        self._update_memory(prompt, answer)
        return answer


def agent(
    model: str,
    api_key: str,
    tools: list[BaseTool] | None = None,
    memory: bool = False,
    provider: str | None = None,
    system_prompt: str = "You are a helpful AI assistant.",
) -> SimpleAgent:
    """
    Factory function to quickly instantiate a working agent.
    """
    llm = make_llm(
        model=model,
        api_key=api_key,
        provider=provider,
        system_prompt=system_prompt,
        temperature=0.2,
        max_tokens=1024,
    )

    config = AgentConfig(
        llm=llm,
        tools=tools or [],
        agent_type="react",
        max_iterations=10,
        system_prompt=system_prompt,
    )

    executor = AgentExecutor(config)
    conv_memory = ConversationMemory() if memory else None

    return SimpleAgent(executor=executor, memory=conv_memory)
