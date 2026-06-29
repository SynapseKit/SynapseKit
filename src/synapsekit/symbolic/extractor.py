from __future__ import annotations

import json
import re
from typing import Any

from ..llm.base import BaseLLM
from .types import (
    ConstraintLanguage,
    ConstraintSet,
    SymbolicBackend,
    VerificationFailure,
)

_EXTRACTION_PROMPT = """\
Convert the user's problem into formal constraints for the {language} solver.

Return only JSON with these fields:
- language: one of smtlib, prolog, minizinc, sympy
- source: the complete formal constraint program or expression
- objective: optional optimization or goal description
- metadata: optional object with extraction notes

Problem:
{problem}
"""


class ConstraintExtractor:
    """LLM-driven natural-language to formal-constraint extractor."""

    def __init__(self, llm: BaseLLM, prompt_template: str | None = None) -> None:
        self._llm = llm
        self._prompt_template = prompt_template or _EXTRACTION_PROMPT

    async def extract(
        self,
        problem: str,
        *,
        backend: SymbolicBackend,
        previous_error: str | None = None,
    ) -> ConstraintSet:
        prompt = self._prompt_template.format(language=backend.language, problem=problem)
        if previous_error:
            prompt = (
                f"{prompt}\n\nPrevious verification failed with this error:\n"
                f"{previous_error}\nReturn a corrected formalization."
            )

        raw = await self._llm.generate(prompt)
        return parse_constraint_response(raw, default_language=backend.language)


def parse_constraint_response(
    response: str | dict[str, Any],
    *,
    default_language: ConstraintLanguage = "text",
) -> ConstraintSet:
    if isinstance(response, dict):
        data = response
    else:
        data = _loads_json_object(response)

    language = data.get("language", default_language)
    if language not in {"smtlib", "prolog", "minizinc", "sympy", "text"}:
        raise VerificationFailure(f"Unsupported constraint language: {language!r}")

    source = data.get("source") or data.get("constraints")
    if not isinstance(source, str) or not source.strip():
        raise VerificationFailure("Constraint extraction did not return non-empty source.")

    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {"notes": metadata}

    objective = data.get("objective")
    if objective is not None and not isinstance(objective, str):
        objective = str(objective)

    return ConstraintSet(
        language=language,
        source=source.strip(),
        objective=objective,
        metadata=metadata,
    )


def _loads_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match is None:
            raise VerificationFailure("Constraint extractor did not return JSON.") from exc
        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise VerificationFailure("Constraint extractor JSON must be an object.")
    return data
