from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from ..audit import EventKind
from .recorder import SessionRecorder
from .safety import SafetyPolicy
from .types import (
    ComputerAction,
    ComputerActionType,
    ComputerObservation,
    ComputerStep,
    ComputerUseProvider,
    ComputerUseResult,
    ConfirmationCallback,
    SafetyDecision,
    ScreenProvider,
)


def _action_audit_payload(action: ComputerAction) -> dict[str, Any]:
    """Return useful action metadata without duplicating typed secret content."""

    return {
        "type": action.action_type.value,
        "x": action.x,
        "y": action.y,
        "end_x": action.end_x,
        "end_y": action.end_y,
        "text_length": len(action.text) if action.text is not None else 0,
        "key": action.key,
        "keys": list(action.keys),
        "delta_x": action.delta_x,
        "delta_y": action.delta_y,
        "url": action.url,
        "duration": action.duration,
        "reason": action.reason,
    }


def _record_environment_event(env: object | None, payload: dict[str, Any]) -> None:
    tracer = getattr(env, "tracer", None)
    session_id = getattr(env, "session_id", None)
    if tracer is None:
        return
    tracer.record(
        EventKind.SYSTEM_EVENT,
        payload,
        actor=f"sandbox:{session_id}" if session_id else "computer-use",
    )


class ComputerUseAgent:
    """Drive a screen with a provider-agnostic computer-use loop."""

    def __init__(
        self,
        *,
        provider: ComputerUseProvider,
        screen: ScreenProvider,
        safety: SafetyPolicy | None = None,
        recorder: SessionRecorder | str | Path | None = None,
        max_steps: int = 20,
        confirm: ConfirmationCallback | None = None,
        close_screen: bool = True,
    ) -> None:
        self.provider = provider
        self.screen = screen
        self.safety = safety or SafetyPolicy()
        self.max_steps = max_steps
        self.confirm = confirm
        self.close_screen = close_screen
        self.recorder: SessionRecorder | None
        if isinstance(recorder, (str, Path)):
            self.recorder = SessionRecorder(recorder)
        else:
            self.recorder = recorder

    async def run(self, task: str, *, env: object | None = None) -> ComputerUseResult:
        """Run a task, optionally using a sandbox environment's screen.

        The environment owns its screen lifecycle. Consequently a screen
        supplied through ``env`` is never closed by this agent.
        """

        active_screen = getattr(env, "screen", None) if env is not None else self.screen
        if active_screen is None:
            raise ValueError("ComputerUseAgent requires a screen or an environment with screen.")

        steps: list[ComputerStep] = []
        output = ""
        error: str | None = None
        completed = False
        if self.recorder is not None:
            self.recorder.start(task)
        _record_environment_event(env, {"event": "computer.run.start", "task": task})

        try:
            for _ in range(self.max_steps):
                observation = await active_screen.observe()
                _record_environment_event(
                    env,
                    {
                        "event": "computer.observe",
                        "app": observation.app,
                        "window_title": observation.window_title,
                        "url": observation.url,
                        "width": observation.width,
                        "height": observation.height,
                        "has_screenshot": observation.screenshot is not None,
                        "text_length": len(observation.text),
                    },
                )
                if self.recorder is not None:
                    self.recorder.record_observation(observation)

                action = await self.provider.next_action(task, observation, steps)
                safety = self.safety.check(action, observation, task=task)
                _record_environment_event(
                    env,
                    {
                        "event": "computer.action",
                        "action": _action_audit_payload(action),
                        "safety": safety.decision.value,
                        "safety_reason": safety.reason,
                    },
                )
                if self.recorder is not None:
                    self.recorder.record_action(action, safety)

                if safety.decision == SafetyDecision.BLOCKED:
                    error = safety.reason
                    step = ComputerStep(
                        observation=observation, action=action, safety=safety, error=error
                    )
                    steps.append(step)
                    if self.recorder is not None:
                        self.recorder.record_step(step)
                    break

                if safety.decision == SafetyDecision.NEEDS_CONFIRMATION:
                    confirmed = await self._confirm(action, observation)
                    _record_environment_event(
                        env,
                        {
                            "event": "computer.confirmation",
                            "action": _action_audit_payload(action),
                            "confirmed": confirmed,
                        },
                    )
                    if not confirmed:
                        error = "Human confirmation denied."
                        step = ComputerStep(
                            observation=observation,
                            action=action,
                            safety=safety,
                            error=error,
                        )
                        steps.append(step)
                        if self.recorder is not None:
                            self.recorder.record_step(step)
                        break

                if action.action_type == ComputerActionType.DONE:
                    completed = True
                    output = action.reason or "done"
                    step = ComputerStep(
                        observation=observation, action=action, safety=safety, output=output
                    )
                    steps.append(step)
                    if self.recorder is not None:
                        self.recorder.record_step(step)
                    break

                execution_output = await active_screen.execute(action)
                _record_environment_event(
                    env,
                    {
                        "event": "computer.execute",
                        "action": _action_audit_payload(action),
                        "output": execution_output[-2000:],
                    },
                )
                step = ComputerStep(
                    observation=observation,
                    action=action,
                    safety=safety,
                    output=execution_output,
                )
                steps.append(step)
                if self.recorder is not None:
                    self.recorder.record_step(step)
            else:
                error = f"ComputerUseAgent reached max_steps={self.max_steps}."
        except Exception as exc:
            error = str(exc)
            _record_environment_event(env, {"event": "computer.run.error", "error": error})
        finally:
            if self.recorder is not None:
                self.recorder.finish(completed=completed, output=output, error=error)
            _record_environment_event(
                env,
                {
                    "event": "computer.run.finish",
                    "completed": completed,
                    "output": output[-2000:],
                    "error": error,
                    "steps": len(steps),
                },
            )
            if self.close_screen and env is None:
                await active_screen.close()

        return ComputerUseResult(
            task=task,
            completed=completed,
            steps=steps,
            output=output,
            error=error,
            recording_path=self.recorder.path if self.recorder is not None else None,
        )

    async def _confirm(self, action: ComputerAction, observation: ComputerObservation) -> bool:
        if self.confirm is None:
            return False
        result = self.confirm(action, observation)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)
