from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LearnedPatterns:
    tone: str = "neutral"
    structure: str = "bulleted"
    vocabulary: dict[str, str] = field(
        default_factory=lambda: {
            "deploy": "ship",
            "defect": "bug",
        }
    )
    code_conventions: list[str] = field(
        default_factory=lambda: [
            "prefers explicit types",
            "uses guard clauses",
        ]
    )
    review_style: str = "holistic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedPatterns:
        return cls(
            tone=data.get("tone", "neutral"),
            structure=data.get("structure", "bulleted"),
            vocabulary=data.get("vocabulary", {}),
            code_conventions=data.get("code_conventions", []),
            review_style=data.get("review_style", "holistic"),
        )


class StyleProfile:
    """Manages a human-editable, versioned style profile file (style.md)."""

    def __init__(self, profile_path: str = "~/.synapsekit/twin/style.md") -> None:
        self.profile_path = Path(profile_path).expanduser().resolve()
        self._version: int = 1
        self._patterns: LearnedPatterns = LearnedPatterns()
        if self.profile_path.exists():
            # Synchronous read at construction time only; the public IO
            # methods (load/save/update_from_samples) are async and offload
            # blocking filesystem calls via asyncio.to_thread.
            self._load_sync()

    @property
    def version(self) -> int:
        return self._version

    @property
    def patterns(self) -> LearnedPatterns:
        return self._patterns

    def _load_sync(self) -> LearnedPatterns:
        """Blocking load implementation. Runs off-thread via load()."""
        if not self.profile_path.exists():
            self._patterns = LearnedPatterns()
            self._version = 1
            return self._patterns

        try:
            content = self.profile_path.read_text(encoding="utf-8")
            self._patterns, self._version = self._parse_markdown_profile(content)
        except Exception as err:
            logger.warning("Failed to parse style profile at %s: %s", self.profile_path, err)
            self._patterns = LearnedPatterns()
            self._version = 1
        return self._patterns

    async def load(self) -> LearnedPatterns:
        """Load style patterns and version from profile_path (async, off-thread)."""
        return await asyncio.to_thread(self._load_sync)

    def _save_sync(self, patterns: LearnedPatterns | None = None) -> None:
        """Blocking save implementation. Runs off-thread via save()."""
        if patterns is not None:
            self._patterns = patterns
        self._version += 1
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_content = self._render_markdown_profile(self._patterns, self._version)
        self.profile_path.write_text(markdown_content, encoding="utf-8")

    async def save(self, patterns: LearnedPatterns | None = None) -> None:
        """Save patterns to profile_path, incrementing version (async, off-thread)."""
        await asyncio.to_thread(self._save_sync, patterns)

    async def update_from_samples(self, samples: list[str]) -> LearnedPatterns:
        """Extract patterns from writing samples and update profile."""
        if not samples:
            return self._patterns

        combined = " ".join(samples).lower()

        # Heuristic tone detection
        if any(w in combined for w in ["pls", "thx", "lgtm", "cool", "super"]):
            tone = "casual"
        elif any(w in combined for w in ["furthermore", "accordingly", "whereas", "hereby"]):
            tone = "formal"
        else:
            tone = self._patterns.tone

        # Heuristic structure detection
        bullet_count = combined.count("- ") + combined.count("* ")
        if bullet_count > len(samples):
            structure = "bulleted"
        elif "\n\n" in combined:
            structure = "prose"
        else:
            structure = self._patterns.structure

        # Heuristic vocabulary mapping updates
        vocab = dict(self._patterns.vocabulary)
        if "ship" in combined and "deploy" not in combined:
            vocab["deploy"] = "ship"
        if "defect" in combined:
            vocab["bug"] = "defect"

        new_patterns = LearnedPatterns(
            tone=tone,
            structure=structure,
            vocabulary=vocab,
            code_conventions=list(self._patterns.code_conventions),
            review_style=self._patterns.review_style,
        )
        await self.save(new_patterns)
        return self._patterns

    def _render_markdown_profile(self, patterns: LearnedPatterns, version: int) -> str:
        data_json = json.dumps(patterns.to_dict(), indent=2)
        return (
            f"---\n"
            f"version: {version}\n"
            f"---\n\n"
            f"# Digital Twin Style Profile (v{version})\n\n"
            f"```json\n"
            f"{data_json}\n"
            f"```\n\n"
            f"## Tone\n{patterns.tone}\n\n"
            f"## Structure\n{patterns.structure}\n\n"
            f"## Review Style\n{patterns.review_style}\n"
        )

    def _parse_markdown_profile(self, content: str) -> tuple[LearnedPatterns, int]:
        version = 1
        patterns = LearnedPatterns()

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                header = parts[1]
                for line in header.splitlines():
                    if line.startswith("version:"):
                        try:
                            version = int(line.split(":", 1)[1].strip())
                        except ValueError:
                            version = 1
                body = parts[2]
                if "```json" in body:
                    json_str = body.split("```json", 1)[1].split("```", 1)[0].strip()
                    try:
                        data = json.loads(json_str)
                        patterns = LearnedPatterns.from_dict(data)
                    except Exception:
                        pass
        return patterns, version
