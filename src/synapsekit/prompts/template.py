from __future__ import annotations

from typing import Any


class PromptTemplate:
    """f-string style prompt template.

    Example:
        >>> tpl = PromptTemplate("Hello {name}")
        >>> tpl.format(name="World")
        'Hello World'
    """

    def __init__(self, template: str) -> None:
        self._template = template

    def format(self, **kwargs: Any) -> str:
        return self._template.format(**kwargs)


class ChatPromptTemplate:
    """Build a List[dict] messages structure for chat LLMs.

    Example:
        >>> tpl = ChatPromptTemplate([
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hello {name}"},
        ... ])
        >>> tpl.format_messages(name="Dhruv")
        [{'role': 'system', 'content': 'You are helpful.'}, {'role': 'user', 'content': 'Hello Dhruv'}]
    """

    def __init__(self, messages: list[dict[str, str]]) -> None:
        self._messages = messages

    def format_messages(self, **kwargs: Any) -> list[dict[str, str]]:
        return [
            {"role": m["role"], "content": m["content"].format(**kwargs)} for m in self._messages
        ]


class FewShotPromptTemplate:
    """Render few-shot examples followed by a suffix prompt.

    Example:
        >>> tpl = FewShotPromptTemplate(
        ...     examples=[
        ...         {"q": "2+2", "a": "4"},
        ...         {"q": "3+3", "a": "6"},
        ...     ],
        ...     example_template="Q: {q}\nA: {a}",
        ...     suffix="Q: {question}\nA:",
        ... )
        >>> tpl.format(question="5+5")
        'Q: 2+2\nA: 4\n\nQ: 3+3\nA: 6\n\nQ: 5+5\nA:'
    """

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
