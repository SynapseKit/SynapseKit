"""Example demonstrating multi-agent orchestration evaluation (Issue #983).

Detects handoff loops, context loss, and mis-routing over a RunGraph. Runs offline
without external LLM dependencies.
"""

from __future__ import annotations

import asyncio

from synapsekit.agents.multi import Handoff, HandoffChain
from synapsekit.evaluation.orchestration import (
    ContextLossDetector,
    LoopDetector,
    MisroutingDetector,
    OrchestrationEvaluator,
    RunGraph,
)


class DummyExecutor:
    """Offline dummy agent executor for demonstration."""

    def __init__(self, response: str) -> None:
        self.response = response

    async def run(self, input_text: str) -> str:
        return f"{self.response} (received: '{input_text}')"


async def main() -> None:
    print("=== SynapseKit Multi-Agent Orchestration Evaluation Demo ===\n")

    # 1. Build a RunGraph from a HandoffChain execution
    triage_exec = DummyExecutor("Triage completed. Needs specialist billing support.")
    billing_exec = DummyExecutor("Billing team checked account ACC-109.")

    chain = HandoffChain()
    chain.add_agent(
        "triage",
        triage_exec,
        handoffs=[Handoff("billing", condition=lambda r: "billing" in r.lower())],
    )
    chain.add_agent("billing", billing_exec)

    handoff_res = await chain.run("triage", "I have an issue with my recent invoice payment")
    graph = RunGraph.from_handoff_result(
        handoff_res,
        goal="Resolve invoice payment inquiry",
        run_id="demo_run_01",
    )

    print(f"Built RunGraph '{graph.run_id}' for goal: '{graph.goal}'")
    print(f"Agent sequence: {' -> '.join(graph.agent_sequence())}\n")

    # 2. Evaluate with OrchestrationEvaluator
    evaluator = OrchestrationEvaluator(
        loop_detector=LoopDetector(),
        context_loss_detector=ContextLossDetector(),
        misrouting_detector=MisroutingDetector(),
    )

    report = evaluator.evaluate(graph)
    print("--- Evaluation Markdown Report ---")
    print(report.to_markdown())
    print("\n--- End of Demo ---")


if __name__ == "__main__":
    asyncio.run(main())
