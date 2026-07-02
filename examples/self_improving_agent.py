"""SelfImprovingAgent demo with no external API keys."""

from __future__ import annotations

import asyncio
from pathlib import Path

from synapsekit import (
    AgentConfig,
    AgentConfigPatch,
    AgentExecutor,
    BaseLLM,
    EvalSuite,
    LLMConfig,
    RolloutPolicy,
    SelfImprovingAgent,
)


class DemoLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(LLMConfig(model="demo", api_key="", provider="local"))

    async def stream(self, prompt: str, **kw):
        del prompt, kw
        yield "demo response"


class DemoAnalyzer:
    async def propose(self, *, samples, snapshot, improvement_targets):
        del samples, snapshot, improvement_targets
        return [
            AgentConfigPatch(
                patch_type="prompt_rewrite",
                description="Prefer clarification on ambiguous arithmetic prompts.",
                changes={
                    "system_prompt": (
                        "You are a careful assistant. Ask one clarification question "
                        "when arithmetic requests omit required operands."
                    )
                },
            )
        ]


async def eval_case(prompt: str = ""):
    score = 0.95 if "clarification" in prompt else 0.70
    return {"score": score, "cost_usd": 0.0}


async def main() -> None:
    executor = AgentExecutor(
        AgentConfig(
            llm=DemoLLM(),
            tools=[],
            agent_type="react",
            system_prompt="You are a helpful assistant.",
        )
    )

    agent = SelfImprovingAgent(
        base_agent=executor,
        eval_suite=EvalSuite.from_cases([("clarification_eval", eval_case)], threshold=0.85),
        rollout=RolloutPolicy(canary_pct=5, min_eval_score=0.85),
        meta_analyzer=DemoAnalyzer(),
        audit_path=Path(".synapsekit_agent_evolution.jsonl"),
        agent_id="demo-agent",
    )

    result = await agent.evolve()
    print(f"cycle status: {result.status}")
    print(f"active prompt: {executor.config.system_prompt}")
    for row in agent.inspect_evolution(limit=5):
        print(f"{row['patch_id'][:8]} {row['status']} score={row['eval_score']}")


if __name__ == "__main__":
    asyncio.run(main())
