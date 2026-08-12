"""Lexer for mixed shell and quoted natural-language input."""

from __future__ import annotations

import re

from .types import InputSegment, ParsedInput, SegmentKind

_NL_CUES = re.compile(
    r"^(?:find|locate|search|why|explain|show me|tell me|open|rerun|run|list|summarize|"
    r"check|inspect|where|which|what|help|clean up|look for)\b",
    re.IGNORECASE,
)


def _looks_like_natural_language(value: str, *, at_line_start: bool) -> bool:
    text = value.strip()
    if not text:
        return False
    # Explicit sentinels are always natural language, wherever they appear.
    if text.casefold().startswith(("nl:", "ask:", "natural language:")):
        return True
    # The cue-word and bare-phrase heuristics only apply to a quote that opens
    # a command (line start or right after a connector). A quoted argument in
    # the middle of a command — e.g. ``git commit -m "find and fix the bug"``
    # or ``grep "search term" file`` — is an ordinary shell token, never NL.
    if not at_line_start:
        return False
    if _NL_CUES.match(text):
        return True
    # A standalone quoted phrase with whitespace is the documented shorthand.
    return any(char.isspace() for char in text)


def lex_input(raw: str) -> ParsedInput:
    """Split a line while preserving executable shell text verbatim.

    Quoted natural-language spans are removed from the shell stream, but their
    source offsets are retained for diagnostics. Quotes used as ordinary shell
    arguments (``echo "hello"``) remain shell text.
    """

    segments: list[InputSegment] = []
    shell_start = 0
    index = 0
    length = len(raw)
    at_line_start = True

    def append_shell(end: int) -> None:
        nonlocal shell_start
        if end <= shell_start:
            return
        value = raw[shell_start:end]
        if value.strip():
            segments.append(InputSegment(value, SegmentKind.SHELL, shell_start, end))

    while index < length:
        char = raw[index]
        if char == "\\" and index + 1 < length:
            index += 2
            at_line_start = False
            continue
        if char not in {"'", '"'}:
            at_line_start = at_line_start and char.isspace()
            index += 1
            continue

        quote = char
        closing = index + 1
        escaped = False
        while closing < length:
            current = raw[closing]
            if escaped:
                escaped = False
            elif current == "\\" and quote == '"':
                escaped = True
            elif current == quote:
                break
            closing += 1
        if closing >= length:
            # Let the shell parser produce the precise unmatched-quote error.
            index += 1
            continue

        value = raw[index + 1 : closing]
        before = raw[:index].rstrip()
        quote_at_start = not before or before.endswith(("&&", "||", ";", "|"))
        if _looks_like_natural_language(value, at_line_start=quote_at_start):
            append_shell(index)
            if value.casefold().startswith("nl:"):
                value = value[3:].lstrip()
            elif value.casefold().startswith("ask:"):
                value = value[4:].lstrip()
            segments.append(InputSegment(value, SegmentKind.NATURAL_LANGUAGE, index, closing + 1))
            shell_start = closing + 1
            at_line_start = False
        index = closing + 1

    append_shell(length)
    return ParsedInput(raw=raw, segments=tuple(segments))


def split_shell_commands(text: str, *, shell: str = "bash") -> list[tuple[str, str]]:
    """Split safe shell operators without invoking a shell interpreter."""

    commands: list[tuple[str, str]] = []
    start = 0
    connector = ""
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and quote != "'":
            escaped = True
            index += 1
            continue
        if quote:
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        candidate = ""
        width = 1
        if text.startswith("&&", index) or text.startswith("||", index):
            candidate = text[index : index + 2]
            width = 2
        elif char in {";", "|"}:
            candidate = char
        if candidate:
            value = text[start:index].strip()
            if value:
                commands.append((value, connector))
            connector = candidate
            start = index + width
            index += width
            continue
        index += 1
    if quote:
        raise ValueError("unmatched quote in shell input")
    value = text[start:].strip()
    if value:
        commands.append((value, connector))
    return commands
