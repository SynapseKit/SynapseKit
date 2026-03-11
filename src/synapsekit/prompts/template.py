from __future__ import annotations

from typing import Any


class PromptTemplate:
    """f-string style prompt template."""

    def __init__(self, template: str) -> None:
        self._template = template

    def format(self, **kwargs: Any) -> str:
        return self._template.format(**kwargs)


class ChatPromptTemplate:
    """Build a List[dict] messages structure for chat LLMs."""

    def __init__(self, messages: list[dict[str, str]]) -> None:
        self._messages = messages

    def format_messages(self, **kwargs: Any) -> list[dict[str, str]]:
        return [
            {"role": m["role"], "content": m["content"].format(**kwargs)} for m in self._messages
        ]


class FewShotPromptTemplate:
    """Render few-shot examples followed by a suffix prompt."""

    def __init__(
        self,
        examples: list[dict[str, str]],
        example_template: str,
        suffix: str,
    ) -> None:
        self._examples = examples
        self._example_template = example_template
        self._suffix = suffix

    def format(self, **kwargs: Any) -> str:
        example_strs = [self._example_template.format(**ex) for ex in self._examples]
        examples_text = "\n\n".join(example_strs)
        suffix = self._suffix.format(**kwargs)
        return f"{examples_text}\n\n{suffix}" if examples_text else suffix
