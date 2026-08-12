"""Natural-language shell planning with a deterministic safe default."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .types import PlannedStep, ShellContext, ShellKind


class PlanningError(RuntimeError):
    """Raised when natural language cannot be translated safely."""


class ShellPlanner(Protocol):
    async def plan(self, request: str, context: ShellContext) -> list[PlannedStep]: ...


def _shell_command(context: ShellContext, unix: str, powershell: str | None = None) -> str:
    return powershell if context.shell is ShellKind.POWERSHELL and powershell else unix


class RuleBasedPlanner:
    """Predictable offline planner for common Agent OS Shell intents."""

    async def plan(self, request: str, context: ShellContext) -> list[PlannedStep]:
        text = request.strip()
        lowered = text.casefold()
        if not text:
            raise PlanningError("natural-language request is empty")
        if "directory" in lowered and any(
            word in lowered for word in ("large", "size", "gb", "space", "12gb")
        ):
            return [
                PlannedStep(
                    command=_shell_command(
                        context,
                        "du -sh -- */ | sort -h",
                        "Get-ChildItem -Directory | ForEach-Object { $size=(Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum; [pscustomobject]@{Size=$size;Name=$_.Name} } | Sort-Object Size",
                    ),
                    explanation="Measure immediate subdirectory sizes and sort them.",
                    source="rule",
                )
            ]
        if any(
            phrase in lowered for phrase in ("git status", "status of this repo", "what changed")
        ):
            return [
                PlannedStep("git status --short --branch", "Show the repository state.", "rule")
            ]
        if "open the pr" in lowered or "open a pr" in lowered:
            return [
                PlannedStep(
                    "gh pr create --fill --web",
                    "Create a filled GitHub pull request and open it in the browser.",
                    "rule",
                )
            ]
        if any(phrase in lowered for phrase in ("rerun", "re-run", "run the test", "run tests")):
            test_path = next(
                (
                    hit["path"]
                    for hit in context.mesh_hits
                    if str(hit.get("path", "")).endswith(".py")
                    and "test" in str(hit.get("path", "")).casefold()
                ),
                None,
            )
            command = f"pytest -q {test_path}" if test_path else "pytest -q"
            return [PlannedStep(command, "Run the most relevant discovered test target.", "rule")]
        if any(word in lowered for word in ("find", "locate", "search")):
            needle = _extract_quoted_or_last_phrase(text)
            if needle:
                return [
                    PlannedStep(
                        f"rg -n --hidden {needle!r} .", "Search tracked project text.", "rule"
                    )
                ]
            return [PlannedStep("rg --files", "List project files for inspection.", "rule")]
        if lowered.startswith(("list files", "show files", "what files")):
            return [
                PlannedStep(
                    _shell_command(
                        context,
                        "find . -maxdepth 2 -type f",
                        "Get-ChildItem -File -Recurse | Select-Object -First 200",
                    ),
                    "List files near the current directory.",
                    "rule",
                )
            ]
        if lowered.startswith(("explain", "why ", "what is")) and context.mesh_hits:
            paths = ", ".join(str(hit.get("path")) for hit in context.mesh_hits[:3])
            return [
                PlannedStep(
                    f"git grep -n {paths!r}",
                    f"Inspect mesh-ranked sources: {paths}.",
                    "rule",
                )
            ]
        raise PlanningError(
            "could not translate that request offline; use shell syntax or configure an LLM planner"
        )


def _extract_quoted_or_last_phrase(text: str) -> str:
    match = re.search(r"['\"]([^'\"]+)['\"]", text)
    if match:
        return match.group(1)
    words = re.split(r"\s+", text.strip())
    return words[-1] if words and words[-1].casefold() not in {"for", "in", "from"} else ""


class LLMShellPlanner:
    """Strict JSON planner. It never executes or approves its own output."""

    def __init__(self, llm: Any, *, model_name: str = "configured") -> None:
        self.llm = llm
        self.model_name = model_name

    async def plan(self, request: str, context: ShellContext) -> list[PlannedStep]:
        prompt = (
            "Translate the request into safe, reviewable command steps. Return ONLY JSON with "
            '{"steps":[{"command":"...","explanation":"..."}]}. Do not include markdown. '
            "Never use shell=True, command substitution, or destructive operations unless the "
            "user explicitly requested them; the host safety gate still decides execution.\n"
            f"Shell context:\n{json.dumps(context.to_dict(), sort_keys=True)}\n"
            f"Request: {request}"
        )
        response = await self.llm.generate(prompt, temperature=0, max_tokens=800)
        try:
            payload = json.loads(_strip_json_fence(response))
            raw_steps = payload["steps"]
        except (KeyError, TypeError, ValueError) as exc:
            raise PlanningError("LLM planner returned invalid JSON") from exc
        if not isinstance(raw_steps, list) or not raw_steps:
            raise PlanningError("LLM planner returned no command steps")
        steps: list[PlannedStep] = []
        for item in raw_steps:
            if not isinstance(item, dict) or not isinstance(item.get("command"), str):
                raise PlanningError("LLM planner returned a malformed command step")
            command = item["command"].strip()
            if not command or len(command) > 8_192:
                raise PlanningError("LLM planner returned an empty or oversized command")
            steps.append(PlannedStep(command, str(item.get("explanation", "")), "llm", 0.7))
        return steps


def _strip_json_fence(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
    return stripped.strip()


@dataclass
class TranslationCache:
    """Small SQLite cache for repeatable, non-destructive translations."""

    path: Path

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(
            path or Path.home() / ".synapsekit" / "shell" / "translations.sqlite3"
        ).expanduser()
        self._ready = False
        self._lock = asyncio.Lock()

    async def get(
        self, request: str, context: ShellContext, model: str = "rules"
    ) -> list[PlannedStep] | None:
        await self._ensure()
        key = self._key(request, context, model)
        data = await asyncio.to_thread(self._get_sync, key)
        if data is None:
            return None
        return [PlannedStep(**item) for item in data]

    async def put(
        self, request: str, context: ShellContext, steps: list[PlannedStep], model: str = "rules"
    ) -> None:
        if any(
            step.command.casefold().find(marker) >= 0
            for step in steps
            for marker in ("reset", "clean", "--force", "remove", "delete")
        ):
            return
        await self._ensure()
        key = self._key(request, context, model)
        await asyncio.to_thread(self._put_sync, key, [step.__dict__ for step in steps])

    async def _ensure(self) -> None:
        if self._ready:
            return
        async with self._lock:
            if not self._ready:
                await asyncio.to_thread(self._init_sync)
                self._ready = True

    def _init_sync(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS translations (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

    def _get_sync(self, key: str) -> list[dict[str, Any]] | None:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT value FROM translations WHERE key=?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _put_sync(self, key: str, value: list[dict[str, Any]]) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO translations(key,value) VALUES(?,?)",
                (key, json.dumps(value)),
            )

    @staticmethod
    def _key(request: str, context: ShellContext, model: str) -> str:
        material = json.dumps(
            {
                "request": request.strip().casefold(),
                "cwd": context.cwd,
                "shell": context.shell.value,
                "model": model,
            },
            sort_keys=True,
        )
        return hashlib.sha256(material.encode()).hexdigest()


class CachedPlanner:
    def __init__(self, planner: ShellPlanner, cache: TranslationCache, *, model: str) -> None:
        self.planner = planner
        self.cache = cache
        self.model = model

    async def plan(self, request: str, context: ShellContext) -> list[PlannedStep]:
        cached = await self.cache.get(request, context, self.model)
        if cached is not None:
            return cached
        steps = await self.planner.plan(request, context)
        await self.cache.put(request, context, steps, self.model)
        return steps
