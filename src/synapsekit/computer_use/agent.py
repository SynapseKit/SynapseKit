from __future__ import annotations

import inspect
from pathlib import Path

from .recorder import SessionRecorder
from .safety import SafetyPolicy
from .types import (
    ComputerActionType,
    ComputerStep,
    ComputerUseProvider,
    ComputerUseResult,
    ConfirmationCallback,
    SafetyDecision,
    ScreenProvider,
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

    async def run(self, task: str) -> ComputerUseResult:
        """Run a task until the provider returns ``done`` or a safety/error stop occurs."""

        steps: list[ComputerStep] = []
        output = ""
        error: str | None = None
        completed = False
        if self.recorder is not None:
            self.recorder.start(task)

        try:
            for _ in range(self.max_steps):
                observation = await self.screen.observe()
                if self.recorder is not None:
                    self.recorder.record_observation(observation)

                action = await self.provider.next_action(task, observation, steps)
                safety = self.safety.check(action, observation, task=task)
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

                execution_output = await self.screen.execute(action)
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
        finally:
            if self.recorder is not None:
                self.recorder.finish(completed=completed, output=output, error=error)
            if self.close_screen:
                await self.screen.close()

        return ComputerUseResult(
            task=task,
            completed=completed,
            steps=steps,
            output=output,
            error=error,
            recording_path=self.recorder.path if self.recorder is not None else None,
        )

    async def _confirm(self, action, observation) -> bool:
        if self.confirm is None:
            return False
        result = self.confirm(action, observation)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)
