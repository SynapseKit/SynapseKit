"""Debate -- multi-agent consensus through debate."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...llm.base import BaseLLM
from ..base import BaseTool
from ..executor import AgentConfig, AgentExecutor


@dataclass
class Debater:
    """An agent participating in a debate."""

    name: str
    role: str
    llm: BaseLLM
    tools: list[BaseTool] = field(default_factory=list)


@dataclass
class Judge:
    """The agent that evaluates the debate and makes the final decision."""

    name: str
    role: str
    llm: BaseLLM
    tools: list[BaseTool] = field(default_factory=list)


@dataclass
class DebateResult:
    """Result of a debate execution."""

    final_decision: str
    transcript: list[str] = field(default_factory=list)


class DebateOrchestrator:
    """Multi-agent debate orchestration.

    Two or more agents debate a topic, and a judge makes the final decision.
    This helps reduce hallucinations and improve reasoning for complex tasks.

    Usage::

        debate = DebateOrchestrator(
            topic="Should we use microservices or a monolith for our new app?",
            proposer=Debater("alice", "Microservices Advocate", llm),
            opponent=Debater("bob", "Monolith Advocate", llm),
            judge=Judge("charlie", "Senior Architect", llm),
            rounds=2,
        )
        result = await debate.run()
    """

    def __init__(
        self,
        topic: str,
        proposer: Debater,
        opponent: Debater,
        judge: Judge,
        rounds: int = 2,
    ) -> None:
        self._topic = topic
        self._proposer = proposer
        self._opponent = opponent
        self._judge = judge
        self._rounds = rounds

    def _build_executor(
        self, role: str, name: str, llm: BaseLLM, tools: list[BaseTool]
    ) -> AgentExecutor:
        system_prompt = f"You are {name}, acting as: {role}.\n"
        config = AgentConfig(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
        )
        return AgentExecutor(config)

    async def run(self) -> DebateResult:
        transcript: list[str] = []

        proposer_executor = self._build_executor(
            self._proposer.role, self._proposer.name, self._proposer.llm, self._proposer.tools
        )
        opponent_executor = self._build_executor(
            self._opponent.role, self._opponent.name, self._opponent.llm, self._opponent.tools
        )
        judge_executor = self._build_executor(
            self._judge.role, self._judge.name, self._judge.llm, self._judge.tools
        )

        # Round 1: Proposer makes the opening statement
        prompt = f"The topic is: {self._topic}\nPlease make your opening argument."
        proposer_response = await proposer_executor.run(prompt)
        transcript.append(f"{self._proposer.name} (Proposer):\n{proposer_response}")

        current_argument = proposer_response

        # Subsequent rounds of debate
        for _ in range(self._rounds):
            # Opponent rebuts
            rebuttal_prompt = (
                f"The topic is: {self._topic}\n\n"
                f"Your opponent argued:\n{current_argument}\n\n"
                f"Please provide your rebuttal and counter-arguments."
            )
            opponent_response = await opponent_executor.run(rebuttal_prompt)
            transcript.append(f"{self._opponent.name} (Opponent):\n{opponent_response}")

            # Proposer defends
            defense_prompt = (
                f"The topic is: {self._topic}\n\n"
                f"Your opponent replied:\n{opponent_response}\n\n"
                f"Please defend your position and rebut their claims."
            )
            proposer_response = await proposer_executor.run(defense_prompt)
            transcript.append(f"{self._proposer.name} (Proposer):\n{proposer_response}")
            current_argument = proposer_response

        # Judge decides
        transcript_text = "\n\n---\n\n".join(transcript)
        judge_prompt = (
            f"You are the judge. The topic of the debate was: {self._topic}\n\n"
            f"Here is the transcript of the debate:\n\n{transcript_text}\n\n"
            f"Based on the arguments presented, synthesize the best points from both sides "
            f"and provide a final, well-reasoned decision."
        )
        final_decision = await judge_executor.run(judge_prompt)

        return DebateResult(
            final_decision=final_decision,
            transcript=transcript,
        )
