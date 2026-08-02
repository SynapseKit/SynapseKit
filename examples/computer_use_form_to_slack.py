"""ComputerUseAgent end-to-end demo (#737): fill a legacy form, post to Slack.

    python examples/computer_use_form_to_slack.py

Drives a (simulated) legacy ERP invoice form and then posts the total to Slack,
through the real ``ComputerUseAgent`` loop — observe → decide → safety-check →
act → record. The screen and the action model are in-process fakes so the demo
runs anywhere with no browser, display, or API key; swap in
``BrowserScreenProvider`` + ``AnthropicComputerUseProvider`` for the real thing.

The whole session is recorded to a JSONL file and can be replayed
(see tests/computer_use/test_session_replay.py).
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from synapsekit.computer_use.agent import ComputerUseAgent
from synapsekit.computer_use.recorder import SessionRecorder
from synapsekit.computer_use.safety import SafetyPolicy
from synapsekit.computer_use.types import (
    ComputerAction,
    ComputerActionType,
    ComputerObservation,
)


class LegacyErpScreen:
    """A tiny in-process 'screen': an invoice form + a Slack channel."""

    def __init__(self) -> None:
        self.url = "about:blank"
        self.invoice_total = ""
        self.submitted = False
        self.slack: list[str] = []

    def _focus(self) -> str:
        return "slack" if "slack" in (self.url or "") else "form"

    async def observe(self) -> ComputerObservation:
        text = (
            f"url={self.url} | invoice_total={self.invoice_total!r} "
            f"submitted={self.submitted} | slack={self.slack}"
        )
        return ComputerObservation(text=text, url=self.url)

    async def execute(self, action: ComputerAction) -> str:
        t = action.action_type
        if t == ComputerActionType.NAVIGATE:
            self.url = action.url or self.url
            return f"navigated to {self.url}"
        if t == ComputerActionType.TYPE_TEXT:
            if self._focus() == "slack":
                self.slack.append(action.text or "")
                return f"posted to Slack: {action.text}"
            self.invoice_total = action.text or ""
            return f"typed into form: {action.text}"
        if t == ComputerActionType.CLICK:
            self.submitted = True
            return "clicked Submit — invoice saved"
        return f"noop ({t})"

    async def close(self) -> None:
        return None


class ScriptedProvider:
    """A deterministic action model (stands in for Claude/Operator/OS-Atlas)."""

    def __init__(self, actions: list[ComputerAction]) -> None:
        self._actions = actions
        self._i = 0

    async def next_action(self, task, observation, history) -> ComputerAction:
        if self._i >= len(self._actions):
            return ComputerAction.done("no more steps")
        action = self._actions[self._i]
        self._i += 1
        return action


SCRIPT = [
    ComputerAction(
        type=ComputerActionType.NAVIGATE,
        url="https://legacy-erp.local/invoices/new",
        reason="open the invoice form",
    ),
    ComputerAction(
        type=ComputerActionType.TYPE_TEXT, text="4820.50", reason="enter the Q2 invoice total"
    ),
    ComputerAction(type=ComputerActionType.CLICK, x=120, y=340, reason="submit the invoice"),
    ComputerAction(
        type=ComputerActionType.NAVIGATE,
        url="https://app.slack.com/finance",
        reason="open Slack #finance",
    ),
    ComputerAction(
        type=ComputerActionType.TYPE_TEXT,
        text="Q2 invoice filed — total $4820.50",
        reason="post the total to #finance",
    ),
    ComputerAction.done("invoice filed and total posted to Slack"),
]


def build_agent(recorder: SessionRecorder | str) -> tuple[ComputerUseAgent, LegacyErpScreen]:
    screen = LegacyErpScreen()
    agent = ComputerUseAgent(
        provider=ScriptedProvider(list(SCRIPT)),
        screen=screen,
        # confirm_before=() so the scripted run completes unattended; SafetyPolicy's
        # confirm/block behaviour is covered in tests/computer_use/test_computer_use_agent.py
        safety=SafetyPolicy(confirm_before=()),
        recorder=recorder,
    )
    return agent, screen


async def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "session.jsonl"
        agent, screen = build_agent(str(path))
        result = await agent.run("File the Q2 invoice and post the total to Slack")

        print(f"completed={result.completed}  output={result.output!r}")
        for i, step in enumerate(result.steps, 1):
            print(
                f"  {i}. {step.action.action_type.value:<10} {step.action.reason or ''} → {step.output or step.error or ''}"
            )
        print(f"form submitted={screen.submitted}  slack={screen.slack}")

        replay = SessionRecorder.load(path)
        print(f"recorded {len(replay.actions)} actions — replayable ✓")


if __name__ == "__main__":
    asyncio.run(main())
