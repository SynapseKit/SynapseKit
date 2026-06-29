from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .types import ComputerAction, ComputerActionType, ComputerObservation, ComputerStep

_ACTION_ALIASES = {
    "left_click": ComputerActionType.CLICK,
    "click": ComputerActionType.CLICK,
    "double_click": ComputerActionType.DOUBLE_CLICK,
    "right_click": ComputerActionType.RIGHT_CLICK,
    "mouse_move": ComputerActionType.MOVE,
    "move": ComputerActionType.MOVE,
    "drag": ComputerActionType.DRAG,
    "type": ComputerActionType.TYPE_TEXT,
    "type_text": ComputerActionType.TYPE_TEXT,
    "text": ComputerActionType.TYPE_TEXT,
    "key": ComputerActionType.KEY,
    "keypress": ComputerActionType.KEY,
    "hotkey": ComputerActionType.HOTKEY,
    "scroll": ComputerActionType.SCROLL,
    "wait": ComputerActionType.WAIT,
    "navigate": ComputerActionType.NAVIGATE,
    "screenshot": ComputerActionType.SCREENSHOT,
    "done": ComputerActionType.DONE,
    "stop": ComputerActionType.DONE,
}


def normalize_action(payload: Mapping[str, Any] | str) -> ComputerAction:
    """Normalize provider-specific action JSON into ``ComputerAction``."""

    if isinstance(payload, str):
        payload = json.loads(payload)
    raw = dict(payload)
    action_name = str(
        raw.get("type")
        or raw.get("action")
        or raw.get("name")
        or raw.get("tool_name")
        or raw.get("kind")
        or ""
    ).lower()
    if not action_name:
        raise ValueError("Provider action has no action/type/name field.")
    action_type = _ACTION_ALIASES.get(action_name)
    if action_type is None:
        raise ValueError(f"Unsupported computer-use action: {action_name!r}")

    coordinate = raw.get("coordinate") or raw.get("position") or raw.get("point") or ()
    end_coordinate = raw.get("end_coordinate") or raw.get("end_position") or ()
    x = _first_int(raw, "x", "left")
    y = _first_int(raw, "y", "top")
    end_x = _first_int(raw, "end_x")
    end_y = _first_int(raw, "end_y")
    if x is None and isinstance(coordinate, Sequence) and len(coordinate) >= 2:
        x = int(coordinate[0])
        y = int(coordinate[1])
    if end_x is None and isinstance(end_coordinate, Sequence) and len(end_coordinate) >= 2:
        end_x = int(end_coordinate[0])
        end_y = int(end_coordinate[1])

    return ComputerAction(
        type=action_type,
        x=x,
        y=y,
        end_x=end_x,
        end_y=end_y,
        text=_first_str(raw, "text", "value", "input", "content"),
        key=_first_str(raw, "key"),
        keys=tuple(raw.get("keys") or ()),
        delta_x=int(raw.get("delta_x") or raw.get("dx") or 0),
        delta_y=int(raw.get("delta_y") or raw.get("dy") or 0),
        url=_first_str(raw, "url"),
        duration=float(raw["duration"]) if raw.get("duration") is not None else None,
        reason=_first_str(raw, "reason", "message"),
        raw=raw,
    )


class OpenSourceComputerUseProvider:
    """Adapter for local/open-source models that return JSON actions."""

    def __init__(self, model: Any) -> None:
        self.model = model

    async def next_action(
        self,
        task: str,
        observation: ComputerObservation,
        history: Sequence[ComputerStep],
    ) -> ComputerAction:
        prompt = _build_json_action_prompt(task, observation, history)
        if hasattr(self.model, "next_action"):
            result = await self.model.next_action(task, observation, history)
        elif hasattr(self.model, "generate"):
            result = await self.model.generate(prompt)
        else:
            raise TypeError(
                "OpenSourceComputerUseProvider model must expose generate() or next_action()."
            )
        return normalize_action(result)


class AnthropicComputerUseProvider:
    """Thin adapter for Anthropic computer-use compatible clients.

    The adapter expects a client object with an async ``next_action`` method or
    an async ``messages.create`` method. Tests can pass a fake client; real
    clients stay optional.
    """

    def __init__(self, client: Any, *, model: str | None = None) -> None:
        self.client = client
        self.model = model

    async def next_action(
        self,
        task: str,
        observation: ComputerObservation,
        history: Sequence[ComputerStep],
    ) -> ComputerAction:
        if hasattr(self.client, "next_action"):
            return normalize_action(await self.client.next_action(task, observation, history))
        if not hasattr(self.client, "messages"):
            raise TypeError("Anthropic client must expose next_action() or messages.create().")
        response = await self.client.messages.create(
            model=self.model,
            messages=[
                {"role": "user", "content": _build_json_action_prompt(task, observation, history)}
            ],
            tools=[{"type": "computer_20241022", "name": "computer"}],
        )
        return normalize_action(_extract_action_payload(response))


class OpenAIComputerUseProvider:
    """Thin adapter for OpenAI computer-tool compatible clients."""

    def __init__(self, client: Any, *, model: str | None = None) -> None:
        self.client = client
        self.model = model

    async def next_action(
        self,
        task: str,
        observation: ComputerObservation,
        history: Sequence[ComputerStep],
    ) -> ComputerAction:
        if hasattr(self.client, "next_action"):
            return normalize_action(await self.client.next_action(task, observation, history))
        if hasattr(self.client, "responses"):
            response = await self.client.responses.create(
                model=self.model,
                input=_build_json_action_prompt(task, observation, history),
                tools=[{"type": "computer_use_preview"}],
            )
            return normalize_action(_extract_action_payload(response))
        raise TypeError("OpenAI client must expose next_action() or responses.create().")


def _extract_action_payload(response: Any) -> Mapping[str, Any] | str:
    if isinstance(response, Mapping):
        for key in ("action", "computer_call", "output", "content"):
            value = response.get(key)
            if value:
                return value
        return response
    for attr in ("action", "computer_call", "output_text", "content"):
        value = getattr(response, attr, None)
        if value:
            return value
    raise ValueError("Could not extract a computer action from provider response.")


def _build_json_action_prompt(
    task: str,
    observation: ComputerObservation,
    history: Sequence[ComputerStep],
) -> str:
    return (
        "Return exactly one JSON object describing the next computer action.\n"
        "Allowed action types: click, double_click, right_click, move, drag, "
        "type_text, key, hotkey, scroll, wait, navigate, screenshot, done.\n"
        f"Task: {task}\n"
        f"Screen text: {observation.text}\n"
        f"App: {observation.app or ''}\n"
        f"Window title: {observation.window_title or ''}\n"
        f"URL: {observation.url or ''}\n"
        f"Steps already executed: {len(history)}"
    )


def _first_int(data: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return int(value)
    return None


def _first_str(data: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return str(value)
    return None
