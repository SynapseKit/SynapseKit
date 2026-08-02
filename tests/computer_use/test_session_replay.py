"""#737 acceptance: end-to-end ComputerUseAgent run + replayable session.

Runs the real ComputerUseAgent loop against an in-process fake screen (fill a
legacy invoice form, then post the total to Slack), records the session, and
proves the recording is replayable — re-running the recorded actions on a fresh
screen reproduces the same end state.
"""

from __future__ import annotations

import asyncio

from synapsekit.computer_use.agent import ComputerUseAgent
from synapsekit.computer_use.recorder import SessionRecorder
from synapsekit.computer_use.safety import SafetyPolicy
from synapsekit.computer_use.types import ComputerAction, ComputerActionType, ComputerObservation


class _Screen:
    def __init__(self) -> None:
        self.url = "about:blank"
        self.total = ""
        self.submitted = False
        self.slack: list[str] = []

    async def observe(self) -> ComputerObservation:
        return ComputerObservation(text=f"{self.url} total={self.total}", url=self.url)

    async def execute(self, action: ComputerAction) -> str:
        t = action.action_type
        if t == ComputerActionType.NAVIGATE:
            self.url = action.url or self.url
        elif t == ComputerActionType.TYPE_TEXT:
            if "slack" in (self.url or ""):
                self.slack.append(action.text or "")
            else:
                self.total = action.text or ""
        elif t == ComputerActionType.CLICK:
            self.submitted = True
        return "ok"

    async def close(self) -> None:
        return None


class _Provider:
    def __init__(self, actions: list[ComputerAction]) -> None:
        self._actions = actions
        self._i = 0

    async def next_action(self, task, observation, history) -> ComputerAction:
        if self._i >= len(self._actions):
            return ComputerAction.done()
        a = self._actions[self._i]
        self._i += 1
        return a


_SCRIPT = [
    ComputerAction(type=ComputerActionType.NAVIGATE, url="https://erp.local/invoice"),
    ComputerAction(type=ComputerActionType.TYPE_TEXT, text="4820.50"),
    ComputerAction(type=ComputerActionType.CLICK, x=1, y=1),
    ComputerAction(type=ComputerActionType.NAVIGATE, url="https://app.slack.com/finance"),
    ComputerAction(type=ComputerActionType.TYPE_TEXT, text="total $4820.50"),
    ComputerAction.done("filed + posted"),
]


def test_end_to_end_form_then_slack(tmp_path) -> None:
    screen = _Screen()
    agent = ComputerUseAgent(
        provider=_Provider(list(_SCRIPT)),
        screen=screen,
        safety=SafetyPolicy(confirm_before=()),
        recorder=str(tmp_path / "s.jsonl"),
    )
    result = asyncio.run(agent.run("File invoice and post to Slack"))
    assert result.completed
    assert screen.submitted is True
    assert screen.slack == ["total $4820.50"]


def test_session_is_replayable(tmp_path) -> None:
    path = tmp_path / "s.jsonl"
    agent = ComputerUseAgent(
        provider=_Provider(list(_SCRIPT)),
        screen=_Screen(),
        safety=SafetyPolicy(confirm_before=()),
        recorder=str(path),
    )
    asyncio.run(agent.run("File invoice and post to Slack"))

    recorded = SessionRecorder.load(path)
    actions = recorded.actions
    assert len(actions) == len(_SCRIPT)
    assert actions[0].action_type == ComputerActionType.NAVIGATE

    # Replay the recorded actions on a fresh screen → same end state.
    replayed = _Screen()

    async def replay() -> None:
        for action in actions:
            if action.action_type != ComputerActionType.DONE:
                await replayed.execute(action)

    asyncio.run(replay())
    assert replayed.submitted is True
    assert replayed.slack == ["total $4820.50"]
    assert replayed.total == "4820.50"
