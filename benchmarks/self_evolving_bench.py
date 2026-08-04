"""Self-evolving agent benchmark for #732.

Quantifies the core claim — *a `SelfImprovingAgent` autonomously raises its own
accuracy over repeated evolution cycles, and the eval gate stops it from
regressing*. One agent is driven through 5 cycles of the real
observe → diagnose → propose → validate → canary loop:

* **the task suite** is synthetic and behavioural. Answer correctness is a pure
  function of which *directives* ("state the unit", "cite the source", "ask when
  ambiguous", ...) appear in the system prompt the model was handed. A scripted
  :class:`BaseLLM` complies with a directive only when it is literally present in
  its prompt, so a prompt with no directives is measurably bad and each directive
  the agent discovers is worth a fixed, verifiable amount of accuracy;
* **the gate is held out**. The :class:`EvalSuite` scores candidate prompts
  against 8 tasks the proposer never observes — it only ever sees feedback from
  the 15 proposer-visible tasks. The agent therefore cannot grade itself;
* **the gate is load-bearing**. From cycle 2 on the proposer also emits a
  deliberately bad decoy candidate that strips the directives learned so far.
  Every decoy must be *blocked* by the eval gate, which is what proves the gate
  is doing work rather than rubber-stamping.

The proposer (:class:`DirectiveProposer`) is deterministic: it clusters negative
:class:`FeedbackSample`s by the directive governing the failing task and proposes
the single most-implicated directive still missing from the prompt. Deterministic,
offline, no API keys, seeded.
"""

from __future__ import annotations

import asyncio
import random
import tempfile
from collections import Counter
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synapsekit.agents import (
    AgentConfig,
    AgentConfigPatch,
    AgentExecutor,
    BaseTool,
    SelfImprovingAgent,
    ToolResult,
)
from synapsekit.agents.self_improving import AgentConfigSnapshot
from synapsekit.evaluation import EvalSuite
from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.training import FeedbackCollector, RolloutPolicy
from synapsekit.training.types import FeedbackSample

SEED = 732
N_CYCLES = 5
DECOY_FROM_CYCLE = 2  # cycle 1 has no directives to strip, so a decoy can't regress yet

BASELINE_PROMPT = "You are a helpful assistant. Answer the user's question."
DECOY_PROMPT = BASELINE_PROMPT + "\nBe concise and skip unnecessary boilerplate."


@dataclass(frozen=True)
class Directive:
    """A behavioural instruction whose presence in the prompt fixes a task class."""

    key: str
    text: str


# Canonical order — also the deterministic tie-break when two directives are
# implicated in the same number of negative samples.
D_UNITS = Directive("show_units", "Always state the unit alongside every numeric answer.")
D_CITE = Directive("cite_source", "Always cite the source document for every factual claim.")
D_CLARIFY = Directive(
    "refuse_ambiguous",
    "Refuse to guess and ask a clarifying question when the request is ambiguous.",
)
D_STEPS = Directive("show_work", "Show the intermediate steps before stating the final answer.")
D_UNKNOWN = Directive(
    "admit_unknown", "Say you do not know rather than speculating beyond the provided context."
)

DIRECTIVES: tuple[Directive, ...] = (D_UNITS, D_CITE, D_CLARIFY, D_STEPS, D_UNKNOWN)


@dataclass(frozen=True)
class Task:
    """One graded task.

    ``answer`` is what the model always says; ``expected`` is the behavioural
    artefact it only adds when ``directive`` is present in its prompt. Grading is
    ``expected in answer`` — the grader never inspects the model's internals.
    """

    task_id: str
    question: str
    answer: str
    expected: str
    directive: Directive | None


# Tasks whose failures the proposer is allowed to observe. Counts per directive
# are distinct (5/4/3/2/1) so "the single most-implicated missing directive" is
# unambiguous on every cycle.
PROPOSER_TASKS: tuple[Task, ...] = (
    Task("p01", "What is the mass of the shipping crate?", "The mass is 12", "kilograms", D_UNITS),
    Task(
        "p02",
        "How far is the depot from the warehouse?",
        "The distance is 340",
        "kilometres",
        D_UNITS,
    ),
    Task(
        "p03", "How long does the nightly batch job take?", "The runtime is 45", "minutes", D_UNITS
    ),
    Task(
        "p04",
        "What is the storage footprint of the archive?",
        "The footprint is 8",
        "gigabytes",
        D_UNITS,
    ),
    Task(
        "p05",
        "What is the operating temperature of the unit?",
        "The temperature is 65",
        "degrees Celsius",
        D_UNITS,
    ),
    Task(
        "p06",
        "What was the revenue growth last year?",
        "Revenue grew 14 percent",
        "(source: fy24-report)",
        D_CITE,
    ),
    Task(
        "p07",
        "How many active customers are there?",
        "There are 4,200 active customers",
        "(source: crm-export)",
        D_CITE,
    ),
    Task(
        "p08",
        "What is the current churn rate?",
        "Churn is 3.1 percent",
        "(source: retention-memo)",
        D_CITE,
    ),
    Task(
        "p09", "Which region grew fastest?", "EMEA grew fastest", "(source: regional-brief)", D_CITE
    ),
    Task(
        "p10",
        "Can you fix the thing for me?",
        "Here is one possible interpretation",
        "could you clarify",
        D_CLARIFY,
    ),
    Task("p11", "Make it better.", "Here is a generic improvement", "could you clarify", D_CLARIFY),
    Task(
        "p12",
        "Update the report with the new number.",
        "Here is an updated draft",
        "could you clarify",
        D_CLARIFY,
    ),
    Task("p13", "What is 47 times 53 minus 100?", "The result is 2391", "Step 1:", D_STEPS),
    Task(
        "p14",
        "If a train covers 300 km in 4 hours, what is its average speed?",
        "The result is 75",
        "Step 1:",
        D_STEPS,
    ),
    Task(
        "p15",
        "What is the CEO's home address?",
        "Looking at what I was given",
        "I do not know",
        D_UNKNOWN,
    ),
)

# The gate. Never shown to the proposer. Three tasks need no directive at all,
# which is what makes the baseline non-zero but still fallible (3/8 = 37.5%);
# the remaining five are unlocked one per directive, so each accepted patch is
# worth exactly +1/8 held-out accuracy.
HELDOUT_TASKS: tuple[Task, ...] = (
    Task("h01", "What is the capital of France?", "The capital of France is", "Paris", None),
    Task("h02", "How many days are in a leap year?", "A leap year has", "366 days", None),
    Task(
        "h03", "Who wrote the play Hamlet?", "The play was written by", "William Shakespeare", None
    ),
    Task(
        "h04",
        "What is the payload limit of the delivery drone?",
        "The payload limit is 5",
        "kilograms",
        D_UNITS,
    ),
    Task(
        "h05",
        "What was the gross margin in Q3?",
        "Gross margin was 62 percent",
        "(source: q3-financials)",
        D_CITE,
    ),
    Task(
        "h06",
        "Sort out the numbers, please.",
        "Here is a plausible ordering",
        "could you clarify",
        D_CLARIFY,
    ),
    Task(
        "h07",
        "What is 18 percent of 250, rounded to the nearest whole number?",
        "The result is 45",
        "Step 1:",
        D_STEPS,
    ),
    Task(
        "h08",
        "What is the internal codename of the unreleased product?",
        "Checking what I was given",
        "I do not know",
        D_UNKNOWN,
    ),
)

ALL_TASKS: tuple[Task, ...] = PROPOSER_TASKS + HELDOUT_TASKS


def render(system_prompt: str, question: str) -> str:
    """Flatten a system prompt and a question into the text the model receives."""
    return f"{system_prompt}\n\nUser: {question}"


def graded(task: Task, answer: str) -> bool:
    """Whether *answer* exhibits the behaviour *task* is testing for."""
    return task.expected in answer


class ScriptedTaskLLM(BaseLLM):
    """A model that complies with a directive only when it is in its prompt.

    It resolves which task it is answering from the question text it was handed —
    it is driven entirely by the prompt, never by out-of-band state — and appends
    the task's behavioural artefact only when the governing directive is present.
    """

    def __init__(self, tasks: tuple[Task, ...] = ALL_TASKS) -> None:
        super().__init__(
            LLMConfig(
                provider="bench", model="scripted-directive-follower", api_key="", max_retries=0
            )
        )
        self._tasks = tasks

    def answer_for(self, text: str) -> str:
        task = self._resolve(text)
        if task is None:
            return "I have no answer for that."
        complies = task.directive is None or task.directive.text in text
        return f"{task.answer} {task.expected}" if complies else task.answer

    def _resolve(self, text: str) -> Task | None:
        for task in self._tasks:
            if task.question in text:
                return task
        return None

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
        del kw
        yield self.answer_for(prompt)

    async def _call_with_tools_impl(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        del tools
        text = "\n".join(str(m.get("content") or "") for m in messages)
        return {"content": self.answer_for(text), "tool_calls": []}


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo input"
    parameters = {"type": "object", "properties": {"input": {"type": "string"}}}

    async def run(self, **kwargs: Any) -> ToolResult:
        return ToolResult(output=str(kwargs.get("input", "")))


class DirectiveProposer:
    """Deterministic :class:`MetaAnalyzerProtocol` implementation.

    Clusters negative feedback by the directive governing the failing task and
    proposes a ``prompt_rewrite`` adding the single most-implicated directive
    still missing from the prompt. From ``decoy_from_cycle`` onwards it *also*
    emits — first, so the gate has to reject it before reaching the good one — a
    decoy that strips every directive learned so far.
    """

    def __init__(
        self,
        tasks: tuple[Task, ...] = PROPOSER_TASKS,
        *,
        decoy_from_cycle: int = DECOY_FROM_CYCLE,
    ) -> None:
        # Only proposer-visible questions: the held-out slice is unmappable here.
        self._by_question = {t.question: t.directive for t in tasks if t.directive is not None}
        self._decoy_from_cycle = decoy_from_cycle
        self.cycle = 0

    async def propose(
        self,
        *,
        samples: list[FeedbackSample],
        snapshot: AgentConfigSnapshot,
        improvement_targets: list[str],
    ) -> list[AgentConfigPatch]:
        self.cycle += 1
        if "prompt" not in improvement_targets:
            return []

        prompt = snapshot.system_prompt
        counts: Counter[str] = Counter()
        for sample in samples:
            if sample.feedback != "negative":
                continue
            directive = self._by_question.get(sample.query)
            if directive is None or directive.text in prompt:
                continue
            counts[directive.key] += 1

        patches: list[AgentConfigPatch] = []
        if self.cycle >= self._decoy_from_cycle:
            patches.append(self._decoy())

        winner = self._most_implicated(counts)
        if winner is not None:
            patches.append(self._adopt(prompt, winner, counts[winner.key]))
        return patches

    def _most_implicated(self, counts: Counter[str]) -> Directive | None:
        best: Directive | None = None
        for directive in DIRECTIVES:  # canonical order breaks ties deterministically
            if counts[directive.key] > (counts[best.key] if best else 0):
                best = directive
        return best

    def _adopt(self, prompt: str, directive: Directive, implicated: int) -> AgentConfigPatch:
        return AgentConfigPatch(
            patch_type="prompt_rewrite",
            description=(f"Adopt '{directive.key}' — implicated in {implicated} negative samples."),
            changes={"system_prompt": f"{prompt.rstrip()}\n- {directive.text}"},
            metadata={
                "source": "directive-proposer",
                "directive": directive.key,
                "negative_samples": implicated,
                "cycle": self.cycle,
            },
        )

    def _decoy(self) -> AgentConfigPatch:
        return AgentConfigPatch(
            patch_type="prompt_rewrite",
            description="Decoy: compress the prompt by dropping the accumulated directives.",
            changes={"system_prompt": DECOY_PROMPT},
            metadata={"source": "directive-proposer", "decoy": True, "cycle": self.cycle},
        )


def heldout_cases(
    llm: ScriptedTaskLLM, tasks: tuple[Task, ...] = HELDOUT_TASKS
) -> list[tuple[str, Any]]:
    """Build ``EvalSuite`` cases that score a candidate prompt on the held-out slice."""

    def make(task: Task) -> Any:
        async def case(prompt: str = "") -> dict[str, float]:
            answer = await llm.generate(render(prompt, task.question))
            return {"score": 1.0 if graded(task, answer) else 0.0, "cost_usd": 0.0}

        case.__name__ = f"heldout_{task.task_id}"
        return case

    return [(f"heldout_{task.task_id}", make(task)) for task in tasks]


@dataclass
class CycleRecord:
    """What one evolution cycle did, snapshotted at the time it happened."""

    cycle: int
    eval_score: float
    status: str
    patch_id: str | None
    directive: str | None
    blocked_patch_ids: list[str] = field(default_factory=list)


@dataclass
class EvolutionBenchResult:
    baseline_score: float
    cycles: list[CycleRecord]
    history: list[AgentConfigPatch]
    agent: SelfImprovingAgent
    eval_suite: EvalSuite
    audit_path: Path
    heldout_size: int

    @property
    def final_score(self) -> float:
        return self.cycles[-1].eval_score if self.cycles else self.baseline_score

    @property
    def uplift(self) -> float:
        return self.final_score - self.baseline_score

    @property
    def accepted(self) -> list[CycleRecord]:
        return [c for c in self.cycles if c.patch_id is not None]

    def score_current_prompt(self) -> float:
        """Held-out accuracy of the agent's *current* system prompt."""
        prompt = self.agent.base_agent.config.system_prompt
        return asyncio.run(self.eval_suite.score_prompt(prompt)).score


async def run_evolution(audit_path: str | Path) -> EvolutionBenchResult:
    """Drive one agent through :data:`N_CYCLES` real evolution cycles.

    *audit_path* must point somewhere disposable — the audit JSONL is written for
    real so the signature round-trip can be verified, and must never land in the
    repo.
    """
    audit_path = Path(audit_path)
    llm = ScriptedTaskLLM()
    executor = AgentExecutor(
        AgentConfig(
            llm=llm,
            tools=[EchoTool()],
            system_prompt=BASELINE_PROMPT,
            agent_type="function_calling",
        )
    )
    suite = EvalSuite.from_cases(heldout_cases(llm), threshold=None)
    collector = FeedbackCollector()
    collector.start()

    agent = SelfImprovingAgent(
        executor,
        eval_suite=suite,
        # min_eval_score=0.0 makes "must beat the current baseline" the binding
        # gate, rather than an absolute floor the early cycles could never clear.
        rollout=RolloutPolicy(
            min_eval_score=0.0,
            rollback_on_regression=True,
            require_human_approval_for=[],
        ),
        feedback_collector=collector,
        meta_analyzer=DirectiveProposer(),
        improvement_targets=["prompt"],
        agent_id="self-evolving-bench",
        audit_path=audit_path,
    )

    # Seeded: task presentation order is shuffled but reproducible, so the result
    # cannot depend on the order failures happen to arrive in.
    order = list(PROPOSER_TASKS)
    random.Random(SEED).shuffle(order)

    baseline = (await suite.score_prompt(executor.config.system_prompt)).score
    cycles: list[CycleRecord] = []

    for cycle in range(1, N_CYCLES + 1):
        for task in order:
            answer = await agent.arun(task.question)
            ok = graded(task, answer)
            collector.record(
                task.question,
                answer,
                "positive" if ok else "negative",
                corrected_response=None if ok else f"{task.answer} {task.expected}",
                metadata={"task_id": task.task_id, "cycle": cycle},
            )
        await collector.flush()

        seen = len(agent.evolution_history())
        outcome = await agent.evolve()
        appended = agent.evolution_history()[: len(agent.evolution_history()) - seen]
        score = (await suite.score_prompt(executor.config.system_prompt)).score

        cycles.append(
            CycleRecord(
                cycle=cycle,
                eval_score=score,
                status=outcome.status,
                patch_id=outcome.patch.patch_id if outcome.patch else None,
                directive=(outcome.patch.metadata.get("directive") if outcome.patch else None),
                blocked_patch_ids=[p.patch_id for p in appended if p.status == "blocked"],
            )
        )

    await collector.stop()
    return EvolutionBenchResult(
        baseline_score=baseline,
        cycles=cycles,
        history=agent.evolution_history(),
        agent=agent,
        eval_suite=suite,
        audit_path=audit_path,
        heldout_size=len(HELDOUT_TASKS),
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        res = asyncio.run(run_evolution(Path(tmp) / "evolution.jsonl"))
    print(f"held-out slice: {res.heldout_size} tasks the proposer never sees")
    print(f"{'baseline':>12}: {res.baseline_score:.0%}")
    for c in res.cycles:
        blocked = f"  (blocked {len(c.blocked_patch_ids)} decoy)" if c.blocked_patch_ids else ""
        print(
            f"{'cycle ' + str(c.cycle):>12}: {c.eval_score:.0%}  {c.status:<9} +{c.directive}{blocked}"
        )
    print(f"{'uplift':>12}: {res.uplift:+.0%}")
