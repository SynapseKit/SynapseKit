from __future__ import annotations

import json
from typing import Any, TypeVar

from .base import BaseLLM

T = TypeVar("T")

_STRUCTURED_PROMPT = """\
Respond with valid JSON that matches this schema:
{schema}

Do not include any text outside the JSON object. Only output the JSON."""


async def generate_structured(
    llm: BaseLLM,
    prompt: str,
    schema: type[T],
    max_retries: int = 3,
) -> T:
    """Generate structured output from an LLM, retrying on parse failure.

    Args:
        llm: The LLM instance to use.
        prompt: The user prompt.
        schema: A Pydantic model class defining the expected output.
        max_retries: Number of retries on parse failure.

    Returns:
        An instance of ``schema`` populated with the LLM's response.

    Raises:
        ValueError: If parsing fails after all retries.
    """
    try:
        from pydantic import BaseModel  # noqa: F401
    except ImportError:
        raise ImportError("pydantic required: pip install pydantic") from None

    json_schema = schema.model_json_schema()  # type: ignore[attr-defined]
    schema_str = json.dumps(json_schema, indent=2)

    messages = [
        {"role": "system", "content": _STRUCTURED_PROMPT.format(schema=schema_str)},
        {"role": "user", "content": prompt},
    ]

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        response = await llm.generate_with_messages(messages)
        try:
            parsed = _extract_json(response)
            return schema(**parsed)  # type: ignore[return-value]
        except Exception as e:
            last_error = e
            # Add feedback for retry
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": f"That response was not valid JSON matching the schema. Error: {e}\nPlease try again with valid JSON only.",
                    }
                )

    raise ValueError(
        f"Failed to generate valid structured output after {max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    # Strip markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return dict(json.loads(text))
