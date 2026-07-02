from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from synapsekit import (
    ComputerAction,
    ComputerActionType,
    ComputerObservation,
    ComputerUseAgent,
    ComputerUseResult,
    SafetyPolicy,
    SessionRecorder,
)
from synapsekit.computer_use import (
    AnthropicComputerUseProvider,
    BrowserScreenProvider,
    OpenAIComputerUseProvider,
    OpenSourceComputerUseProvider,
    SafetyCheck,
    SafetyDecision,
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


class FailingScreen:
    """FakeScreen whose execute() always raises RuntimeError."""

    def __init__(self) -> None:
        self.observation = ComputerObservation(
            screenshot=b"png",
            text="Ready",
            app="browser",
            window_title="Test",
            url="https://example.com/",
        )
        self.closed = False

    async def observe(self) -> ComputerObservation:
        return self.observation

    async def execute(self, action: ComputerAction) -> str:
        raise RuntimeError("screen exploded")

    async def close(self) -> None:
        self.closed = True


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

    assert isinstance(result, ComputerUseResult)
    assert isinstance(result.steps, list)
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

    assert isinstance(result, ComputerUseResult)
    assert isinstance(result.steps, list)
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

    assert isinstance(result, ComputerUseResult)
    assert isinstance(result.steps, list)
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

    assert isinstance(result, ComputerUseResult)
    assert isinstance(result.steps, list)
    assert result.error == "ComputerUseAgent reached max_steps=1."
    assert result.completed is False
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

    # Validate every JSONL line is valid JSON with required fields
    lines = recording.read_text().splitlines()
    for line in lines:
        obj = json.loads(line)
        assert "ts" in obj
        assert "event" in obj


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


# ---------------------------------------------------------------------------
# decorator: inspect.iscoroutinefunction assertions
# ---------------------------------------------------------------------------


def test_requires_confirmation_preserves_async_identity() -> None:
    @requires_human_confirmation
    async def my_action() -> str:
        return "done"

    assert inspect.iscoroutinefunction(my_action)
    assert my_action._requires_confirmation is True  # type: ignore[attr-defined]


def test_requires_confirmation_preserves_sync_identity() -> None:
    @requires_human_confirmation
    def my_action() -> str:
        return "done"

    assert not inspect.iscoroutinefunction(my_action)
    assert my_action._requires_confirmation is True  # type: ignore[attr-defined]


def test_requires_confirmation_preserves_async_identity_with_reason() -> None:
    @requires_human_confirmation("some reason")
    async def my_action() -> str:
        return "done"

    assert inspect.iscoroutinefunction(my_action)
    assert my_action._requires_confirmation is True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# URL domain enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_safety_blocks_url_outside_allowed_domains() -> None:
    policy = SafetyPolicy(allowed_domains=["example.com"])
    action = ComputerAction(type=ComputerActionType.NAVIGATE, url="https://evil.com/steal")
    obs = ComputerObservation(
        screenshot=None, app="browser", window_title="", width=1920, height=1080
    )
    check = policy.check(action, obs)
    assert isinstance(check, SafetyCheck)
    assert check.decision in (SafetyDecision.NEEDS_CONFIRMATION, SafetyDecision.BLOCKED)


@pytest.mark.asyncio
async def test_safety_allows_url_inside_allowed_domains() -> None:
    policy = SafetyPolicy(allowed_domains=["example.com"])
    action = ComputerAction(type=ComputerActionType.NAVIGATE, url="https://example.com/page")
    obs = ComputerObservation(
        screenshot=None, app="browser", window_title="", width=1920, height=1080
    )
    check = policy.check(action, obs)
    assert isinstance(check, SafetyCheck)
    assert check.decision == SafetyDecision.ALLOWED


@pytest.mark.asyncio
async def test_safety_blocks_domain_in_blocklist() -> None:
    policy = SafetyPolicy(blocked_domains=["malware.com"])
    action = ComputerAction(type=ComputerActionType.NAVIGATE, url="https://malware.com/")
    obs = ComputerObservation(
        screenshot=None, app="browser", window_title="", width=1920, height=1080
    )
    check = policy.check(action, obs)
    assert isinstance(check, SafetyCheck)
    assert check.decision == SafetyDecision.BLOCKED


# ---------------------------------------------------------------------------
# normalize_action — aliases + negative tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alias,expected_type",
    [
        ("scroll", "SCROLL"),
        ("drag", "DRAG"),
        ("stop", "DONE"),
        ("key", "KEY"),
    ],
)
def test_normalize_action_all_aliases(alias: str, expected_type: str) -> None:
    action = normalize_action({"type": alias})
    assert action.action_type.name == expected_type


def test_normalize_action_raises_on_empty_payload() -> None:
    with pytest.raises(ValueError):
        normalize_action({})


def test_normalize_action_raises_on_unknown_action() -> None:
    with pytest.raises(ValueError):
        normalize_action({"type": "fly_to_mars"})


# ---------------------------------------------------------------------------
# Exception propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_records_error_on_screen_exception() -> None:
    screen = FailingScreen()
    provider = FakeProvider([ComputerAction(type=ComputerActionType.CLICK, x=5, y=5)])

    result = await ComputerUseAgent(provider=provider, screen=screen).run("click something")

    assert isinstance(result, ComputerUseResult)
    assert result.completed is False
    assert result.error is not None


# ---------------------------------------------------------------------------
# confirm_before=[] disables keyword trigger
# ---------------------------------------------------------------------------


def test_safety_empty_confirm_before_disables_keyword_trigger() -> None:
    policy = SafetyPolicy(confirm_before=[], require_confirmation_for_text=False)
    # typing "password" should NOT trigger confirmation since confirm_before is empty list
    action = ComputerAction(type=ComputerActionType.TYPE_TEXT, text="my password")
    obs = ComputerObservation(
        screenshot=None, app="notes", window_title="", width=1920, height=1080
    )
    check = policy.check(action, obs)
    assert isinstance(check, SafetyCheck)
    assert check.decision == SafetyDecision.ALLOWED
