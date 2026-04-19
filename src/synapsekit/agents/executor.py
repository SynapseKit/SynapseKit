from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Literal

from .._compat import run_sync
from ..llm.base import BaseLLM
from ..memory.agent_memory import AgentMemory as PersistentAgentMemory
from .base import BaseTool
from .function_calling import FunctionCallingAgent
from .memory import AgentScratchpad
from .react import ReActAgent


@dataclass
class AgentConfig:
    llm: BaseLLM
    tools: list[BaseTool]
    agent_type: Literal["react", "function_calling"] = "react"
    max_iterations: int = 10
    system_prompt: str = "You are a helpful AI assistant."
    verbose: bool = False
    memory: PersistentAgentMemory | None = None
    agent_id: str = "default"
    memory_top_k: int = 5


class AgentExecutor:
    """
    High-level agent runner. Picks ReActAgent or FunctionCallingAgent based on config.

    Usage::

        executor = AgentExecutor(AgentConfig(
            llm=OpenAILLM(config),
            tools=[CalculatorTool(), WebSearchTool()],
            agent_type="function_calling",
        ))

        answer = await executor.run("What is 2 ** 10?")
        answer = executor.run_sync("What is 2 ** 10?")
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._agent = self._build_agent()

    def _build_agent(self) -> ReActAgent | FunctionCallingAgent:
        scratchpad = AgentScratchpad(max_steps=self.config.max_iterations)
        if self.config.agent_type == "react":
            return ReActAgent(
                llm=self.config.llm,
                tools=self.config.tools,
                max_iterations=self.config.max_iterations,
                memory=self.config.memory,
                scratchpad=scratchpad,
                agent_id=self.config.agent_id,
                memory_top_k=self.config.memory_top_k,
            )
        elif self.config.agent_type == "function_calling":
            return FunctionCallingAgent(
                llm=self.config.llm,
                tools=self.config.tools,
                max_iterations=self.config.max_iterations,
                memory=self.config.memory,
                scratchpad=scratchpad,
                system_prompt=self.config.system_prompt,
                agent_id=self.config.agent_id,
                memory_top_k=self.config.memory_top_k,
            )
        else:
            raise ValueError(
                f"Unknown agent_type: {self.config.agent_type!r}. "
                "Use 'react' or 'function_calling'."
            )

    async def run(self, query: str) -> str:
        """Async: run agent and return final answer."""
        return await self._agent.run(query)

    async def stream(self, query: str) -> AsyncGenerator[str]:
        """Async: stream final answer tokens."""
        async for token in self._agent.stream(query):
            yield token

    def run_sync(self, query: str) -> str:
        """Sync: run agent (for scripts / notebooks)."""
        return run_sync(self.run(query))

    @property
    def memory(self) -> AgentScratchpad:
        return self._agent.memory

    @property
    def persistent_memory(self) -> PersistentAgentMemory | None:
        return getattr(self._agent, "persistent_memory", None)
