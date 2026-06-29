from __future__ import annotations

from pathlib import Path

import pytest

from synapsekit import (
    ComputerAction,
    ComputerActionType,
    ComputerObservation,
    ComputerUseAgent,
    SafetyPolicy,
    SessionRecorder,
)
from synapsekit.computer_use import (
    AnthropicComputerUseProvider,
    BrowserScreenProvider,
    OpenAIComputerUseProvider,
    OpenSourceComputerUseProvider,
    normalize_action,
    requires_confirmation,
    requires_human_confirmation,
)


class FakeScreen:
    def __init__(self, observation: ComputerObservation | None = None) -> None:
        self.observation = observation or ComputerObservation(
            screenshot=b"png",
            text="Ready",
            app="browser",
            window_title="Legacy portal",
            url="https://example.com/form",
        )
        self.executed: list[ComputerAction] = []
        self.closed = False

    async def observe(self) -> ComputerObservation:
        return self.observation

    async def execute(self, action: ComputerAction) -> str:
        self.executed.append(action)
        return f"executed {action.action_type.value}"

    async def close(self) -> None:
        self.closed = True


class FakeProvider:
    def __init__(self, actions: list[ComputerAction]) -> None:
        self.actions = actions

    async def next_action(self, task, observation, history):
        return self.actions.pop(0)


@pytest.mark.asyncio
async def test_agent_runs_until_done() -> None:
    screen = FakeScreen()
    provider = FakeProvider(
        [
            ComputerAction(type=ComputerActionType.CLICK, x=10, y=20),
            ComputerAction.done("form complete"),
        ]
    )

    result = await ComputerUseAgent(provider=provider, screen=screen).run("fill the form")

    assert result.completed is True
    assert result.output == "form complete"
    assert result.error is None
    assert len(result.steps) == 2
    assert screen.executed[0].action_type == ComputerActionType.CLICK
    assert screen.closed is True


@pytest.mark.asyncio
async def test_safety_blocks_forbidden_apps() -> None:
    screen = FakeScreen(ComputerObservation(app="1Password", window_title="Vault"))
    provider = FakeProvider([ComputerAction(type=ComputerActionType.CLICK, x=1, y=2)])

    result = await ComputerUseAgent(provider=provider, screen=screen).run("click continue")

    assert result.completed is False
    assert result.error == "Current app or window is forbidden by SafetyPolicy."
    assert screen.executed == []


@pytest.mark.asyncio
async def test_confirmation_denial_stops_sensitive_action() -> None:
    screen = FakeScreen()
    provider = FakeProvider(
        [ComputerAction(type=ComputerActionType.TYPE_TEXT, text="send invoice")]
    )
    safety = SafetyPolicy(confirm_before=("send",))

    result = await ComputerUseAgent(provider=provider, screen=screen, safety=safety).run(
        "send invoice"
    )

    assert result.error == "Human confirmation denied."
    assert screen.executed == []
    assert safety.audit_log[-1].decision.value == "needs_confirmation"


@pytest.mark.asyncio
async def test_confirmation_approval_executes_action() -> None:
    screen = FakeScreen()
    provider = FakeProvider(
        [ComputerAction(type=ComputerActionType.TYPE_TEXT, text="send invoice")]
    )
    safety = SafetyPolicy(confirm_before=("send",))

    result = await ComputerUseAgent(
        provider=provider,
        screen=screen,
        safety=safety,
        confirm=lambda _action, _observation: True,
        max_steps=1,
    ).run("send invoice")

    assert result.error == "ComputerUseAgent reached max_steps=1."
    assert screen.executed[0].action_type == ComputerActionType.TYPE_TEXT


@pytest.mark.asyncio
async def test_session_recording_is_replayable(tmp_path: Path) -> None:
    recording = tmp_path / "session.jsonl"
    screen = FakeScreen()
    provider = FakeProvider([ComputerAction.done("finished")])

    result = await ComputerUseAgent(
        provider=provider,
        screen=screen,
        recorder=SessionRecorder(recording),
    ).run("finish")

    replay = SessionRecorder.load(recording)
    assert result.recording_path == recording
    assert replay.observations[0].screenshot == b"png"
    assert replay.actions[0].action_type == ComputerActionType.DONE
    assert replay.events[-1]["event"] == "finish"


def test_normalize_action_supports_provider_aliases() -> None:
    action = normalize_action(
        {
            "action": "left_click",
            "coordinate": [400, 200],
            "reason": "press submit",
        }
    )

    assert action.action_type == ComputerActionType.CLICK
    assert action.x == 400
    assert action.y == 200
    assert action.reason == "press submit"


@pytest.mark.asyncio
async def test_open_source_provider_normalizes_model_json() -> None:
    class Model:
        async def generate(self, prompt: str) -> str:
            assert "Return exactly one JSON object" in prompt
            return '{"type": "done", "reason": "ok"}'

    provider = OpenSourceComputerUseProvider(Model())
    action = await provider.next_action("task", ComputerObservation(text="screen"), [])

    assert action.action_type == ComputerActionType.DONE
    assert action.reason == "ok"


@pytest.mark.asyncio
async def test_anthropic_provider_uses_fake_next_action() -> None:
    class Client:
        async def next_action(self, task, observation, history):
            return {"type": "wait", "duration": 0.1}

    action = await AnthropicComputerUseProvider(Client()).next_action(
        "task", ComputerObservation(), []
    )

    assert action.action_type == ComputerActionType.WAIT
    assert action.duration == 0.1


@pytest.mark.asyncio
async def test_openai_provider_uses_fake_next_action() -> None:
    class Client:
        async def next_action(self, task, observation, history):
            return {"type": "navigate", "url": "https://example.com"}

    action = await OpenAIComputerUseProvider(Client()).next_action(
        "task", ComputerObservation(), []
    )

    assert action.action_type == ComputerActionType.NAVIGATE
    assert action.url == "https://example.com"


@pytest.mark.asyncio
async def test_browser_screen_provider_delegates_to_browser_tool() -> None:
    class Browser:
        async def run(self, **kwargs):
            if kwargs["action"] == "get_text":
                return "Name: Ada"
            if kwargs["action"] == "screenshot":
                return "Screenshot captured.\nbase64:cG5n"
            if kwargs["action"] == "click":
                return "clicked"
            if kwargs["action"] == "close":
                return "closed"
            raise AssertionError(kwargs)

    provider = BrowserScreenProvider(browser=Browser())
    observation = await provider.observe()
    output = await provider.execute(
        ComputerAction(type=ComputerActionType.CLICK, text="button:has-text('Save')")
    )

    assert observation.text == "Name: Ada"
    assert observation.screenshot == b"png"
    assert output == "clicked"


def test_requires_human_confirmation_decorator_marks_function() -> None:
    @requires_human_confirmation("payments")
    def pay() -> str:
        return "ok"

    assert pay() == "ok"
    assert requires_confirmation(pay) is True
